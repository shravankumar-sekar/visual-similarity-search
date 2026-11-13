
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from model import EmbeddingNet
from retrieval import evaluate_retrieval
from train import load_cifar100, set_seed


DEFAULT_MODELS = {
    "untrained": None,
    "random_warmup": (
        "checkpoints/random_mining_stage59.pt"
    ),
    "random_continuation": (
        "checkpoints/stage510_v2/stage510_random.pt"
    ),
    "semi_hard_continuation": (
        "checkpoints/stage511_semi_hard.pt"
    ),
}

CSV_FIELDS = [
    "model",
    "queries",
    "gallery_size",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "map",
]


def select_stratified_queries(
    labels,
    samples_per_class=5,
    seed=42,
):
    """
    Select the same number of query images
    from every represented class.
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("Labels must be 1D")

    if samples_per_class < 1:
        raise ValueError(
            "samples_per_class must be positive"
        )

    rng = np.random.default_rng(seed)
    selected = []

    for label in np.unique(labels):
        positions = np.flatnonzero(
            labels == label
        )

        if len(positions) < samples_per_class:
            raise ValueError(
                f"Not enough images for class {label}"
            )

        chosen = rng.choice(
            positions,
            size=samples_per_class,
            replace=False,
        )

        selected.extend(chosen.tolist())

    return np.asarray(
        sorted(selected),
        dtype=np.int64,
    )


def load_model(
    checkpoint_path,
    embedding_dim=128,
    seed=42,
):
    # The untrained baseline uses deterministic
    # initialization matching our earlier experiments.
    set_seed(seed)

    model = EmbeddingNet(
        embedding_dim=embedding_dim
    )

    if checkpoint_path is not None:
        path = Path(checkpoint_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        config = checkpoint.get("config", {})

        if not isinstance(config, dict):
            raise ValueError(
                "Invalid checkpoint configuration"
            )

        saved_dimension = config.get(
            "embedding_dim"
        )

        if saved_dimension is None:
            warmup_config = config.get(
                "warmup_config", {}
            )

            if isinstance(warmup_config, dict):
                saved_dimension = warmup_config.get(
                    "embedding_dim"
                )

        if (
            saved_dimension is not None
            and saved_dimension != embedding_dim
        ):
            raise ValueError(
                "Checkpoint embedding dimension "
                "does not match evaluation model"
            )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        print(
            f"Loaded: {path}",
            flush=True,
        )

    else:
        print(
            "Loaded: untrained ResNet-18",
            flush=True,
        )

    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    expected_dim=128,
):
    """
    Extract embeddings in deterministic dataset order.

    Gallery IDs are assigned according to this order.
    """
    model.eval()

    embedding_batches = []
    label_batches = []

    for images, labels in loader:
        embeddings = model(images)

        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != expected_dim
        ):
            raise ValueError(
                "Unexpected model embedding shape"
            )

        if not torch.isfinite(embeddings).all().item():
            raise RuntimeError(
                "Non-finite embeddings detected"
            )

        embedding_batches.append(
            embeddings.cpu().numpy()
        )
        label_batches.append(
            labels.cpu().numpy()
        )

    if not embedding_batches:
        raise ValueError(
            "No images available for evaluation"
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    ).astype(np.float32)

    labels = np.concatenate(
        label_batches,
        axis=0,
    )

    ids = np.arange(
        len(labels),
        dtype=np.int64,
    )

    return embeddings, labels, ids


def save_comparison(path, rows):
    path = Path(path)

    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
            )

            writer.writeheader()
            writer.writerows(rows)

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def print_result(name, metrics):
    print(f"\n{name}")
    print("-" * 48)

    print(
        f"Queries:          {metrics['queries']}"
    )
    print(
        f"Gallery size:     {metrics['gallery_size']}"
    )

    for k in (1, 5, 10):
        print(
            f"Recall@{k:<2}:        "
            f"{metrics[f'recall_at_{k}']:.4%}"
        )

    for k in (1, 5, 10):
        print(
            f"Hit@{k:<2}:           "
            f"{metrics[f'hit_at_{k}']:.4%}"
        )

    print(
        f"mAP:              {metrics['map']:.4%}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CIFAR-100 image retrieval "
            "using exact NumPy Euclidean search"
        )
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--query-per-class",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_comparison.csv",
    )

    parser.add_argument(
        "--random-warmup",
        default=DEFAULT_MODELS[
            "random_warmup"
        ],
    )
    parser.add_argument(
        "--random-continuation",
        default=DEFAULT_MODELS[
            "random_continuation"
        ],
    )
    parser.add_argument(
        "--semi-hard",
        default=DEFAULT_MODELS[
            "semi_hard_continuation"
        ],
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("Threads must be positive")

    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    if args.query_per_class < 1:
        parser.error(
            "Queries per class must be positive"
        )

    if args.search_batch_size < 1:
        parser.error(
            "Search batch size must be positive"
        )

    output = Path(args.output)

    if output.exists():
        parser.error(
            f"Output already exists: {output}. "
            "Choose another filename."
        )

    models = {
        "untrained": None,
        "random_warmup": args.random_warmup,
        "random_continuation": (
            args.random_continuation
        ),
        "semi_hard_continuation": args.semi_hard,
    }

    for name, checkpoint in models.items():
        if checkpoint is not None:
            if not Path(checkpoint).is_file():
                parser.error(
                    f"{name} checkpoint missing: "
                    f"{checkpoint}"
                )

    torch.set_num_threads(args.threads)
    set_seed(42)

    print("Stage 5.12 — Retrieval evaluation")
    print(f"CPU threads: {torch.get_num_threads()}")

    print(
        "\nLoading validation dataset...",
        flush=True,
    )

    _, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    # Use deterministic, non-augmented dataset
    # order instead of class-balanced batch order.
    gallery_loader = DataLoader(
        val_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    reference_labels = None
    reference_ids = None
    query_indices = None

    for name, checkpoint in models.items():
        print(
            f"\nEvaluating {name}...",
            flush=True,
        )

        model = load_model(
            checkpoint,
            embedding_dim=128,
            seed=42,
        )

        embeddings, labels, ids = extract_embeddings(
            model,
            gallery_loader,
            expected_dim=128,
        )

        if reference_labels is None:
            reference_labels = labels.copy()
            reference_ids = ids.copy()

            query_indices = select_stratified_queries(
                labels,
                samples_per_class=args.query_per_class,
                seed=42,
            )

            print(
                f"Gallery images: {len(labels)}"
            )
            print(
                f"Query images: {len(query_indices)}"
            )
            print(
                f"Classes: {len(np.unique(labels))}"
            )

        else:
            if not np.array_equal(
                labels,
                reference_labels,
            ):
                raise RuntimeError(
                    "Gallery labels changed "
                    "between models"
                )

            if not np.array_equal(
                ids,
                reference_ids,
            ):
                raise RuntimeError(
                    "Gallery IDs changed "
                    "between models"
                )

        metrics = evaluate_retrieval(
            query_embeddings=embeddings[
                query_indices
            ],
            query_labels=labels[
                query_indices
            ],
            query_ids=ids[
                query_indices
            ],
            gallery_embeddings=embeddings,
            gallery_labels=labels,
            gallery_ids=ids,
            ks=(1, 5, 10),
            batch_size=args.search_batch_size,
        )

        row = {
            "model": name,
            **metrics,
        }

        rows.append(row)
        print_result(name, metrics)

        del model
        del embeddings

    save_comparison(output, rows)

    print(
        f"\nResults saved: {output}",
        flush=True,
    )
    print(
        "Evaluation complete.",
        flush=True,
    )


if __name__ == "__main__":
    main()
