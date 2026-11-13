
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from retrieval import evaluate_retrieval
from scripts.evaluate_retrieval import (
    extract_embeddings,
    select_stratified_queries,
)
from train import load_cifar100, set_seed
from model import EmbeddingNet


def compute_triplet_ordering(
    embeddings,
    labels,
    query_positions,
    trials_per_query=10,
    margin=0.3,
    seed=2026,
):
    """
    Evaluate random-triplet ordering and
    full-gallery hardest-triplet ordering.

    The gallery contains the query images.
    Each query image is excluded from its
    own positive candidates.

    Random ordering:
        d(anchor, positive) < d(anchor, negative)

    Margin ordering:
        d(anchor, positive) + margin
        < d(anchor, negative)

    Hardest ordering:
        max positive distance < min negative distance
    """
    embeddings = np.asarray(
        embeddings, dtype=np.float32
    )
    labels = np.asarray(labels)
    query_positions = np.asarray(
        query_positions, dtype=np.int64
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")

    if labels.ndim != 1:
        raise ValueError("Labels must be 1D")

    if len(embeddings) != len(labels):
        raise ValueError("Embedding/label count mismatch")

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")

    if query_positions.ndim != 1 or len(query_positions) == 0:
        raise ValueError("Query positions must be nonempty")

    if (
        (query_positions < 0).any()
        or (query_positions >= len(labels)).any()
    ):
        raise ValueError("Query positions out of range")

    if trials_per_query < 1:
        raise ValueError("trials_per_query must be positive")

    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")

    rng = np.random.default_rng(seed)

    random_correct = 0
    margin_correct = 0
    hard_correct = 0

    total_triplets = 0

    positive_distances = []
    negative_distances = []

    class_positions = {
        label: np.flatnonzero(labels == label)
        for label in np.unique(labels)
    }

    for query_index in query_positions:
        query_index = int(query_index)
        label = labels[query_index]

        positive_candidates = class_positions[label]
        positive_candidates = positive_candidates[
            positive_candidates != query_index
        ]

        negative_candidates = np.flatnonzero(
            labels != label
        )

        if (
            len(positive_candidates) == 0
            or len(negative_candidates) == 0
        ):
            raise ValueError(
                "Every query requires positive "
                "and negative candidates"
            )

        anchor = embeddings[query_index]

        # Exactly one positive and one negative
        # is sampled for each random trial.
        positive_indices = rng.choice(
            positive_candidates,
            size=trials_per_query,
            replace=True,
        )

        negative_indices = rng.choice(
            negative_candidates,
            size=trials_per_query,
            replace=True,
        )

        selected_positive_distances = np.linalg.norm(
            embeddings[positive_indices] - anchor,
            axis=1,
        )

        selected_negative_distances = np.linalg.norm(
            embeddings[negative_indices] - anchor,
            axis=1,
        )

        random_correct += int(np.count_nonzero(
            selected_positive_distances
            < selected_negative_distances
        ))

        margin_correct += int(np.count_nonzero(
            selected_positive_distances + margin
            < selected_negative_distances
        ))

        total_triplets += trials_per_query

        positive_distances.extend(
            selected_positive_distances.tolist()
        )
        negative_distances.extend(
            selected_negative_distances.tolist()
        )

        # Exhaustive hardest-triplet comparison
        # against the full test gallery.
        all_distances = np.linalg.norm(
            embeddings - anchor,
            axis=1,
        )

        hardest_positive = np.max(
            all_distances[positive_candidates]
        )

        hardest_negative = np.min(
            all_distances[negative_candidates]
        )

        hard_correct += int(
            hardest_positive < hardest_negative
        )

    query_count = len(query_positions)

    return {
        "random_triplet_ordering": (
            random_correct / total_triplets
        ),
        "random_triplet_margin": (
            margin_correct / total_triplets
        ),
        "hard_triplet_ordering": (
            hard_correct / query_count
        ),
        "sampled_positive_distance": float(
            np.mean(positive_distances)
        ),
        "sampled_negative_distance": float(
            np.mean(negative_distances)
        ),
        "triplets_evaluated": total_triplets,
    }


def load_best_model(checkpoint_path):
    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint.get("config", {})

    if not isinstance(config, dict):
        raise ValueError("Invalid checkpoint configuration")

    if config.get("embedding_dim") != 128:
        raise ValueError(
            "Expected a 128-dimensional embedding model"
        )

    if (
        config.get("architecture")
        != "resnet18_scratch"
    ):
        raise ValueError(
            "Unexpected checkpoint architecture"
        )

    model = EmbeddingNet(
        embedding_dim=128
    )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.eval()

    return model, checkpoint


def save_results(json_path, csv_path, results):
    json_path = Path(json_path)
    csv_path = Path(csv_path)

    if json_path.resolve() == csv_path.resolve():
        raise ValueError("Output paths must differ")

    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    json_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    csv_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results.keys()),
        )

        writer.writeheader()
        writer.writerow(results)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Independent CIFAR-100 held-out "
            "test evaluation"
        )
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/stage6_random/best.pt",
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
        "--queries-per-class",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--triplets-per-query",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--output",
        default="results/stage6_test.json",
    )

    parser.add_argument(
        "--csv",
        default="results/stage6_test.csv",
    )

    args = parser.parse_args()

    if args.threads < 1 or args.batch_size < 1:
        parser.error(
            "Threads and batch size must be positive"
        )

    if not 1 <= args.queries_per_class <= 60:
        parser.error(
            "Queries per class must be between 1 and 60"
        )

    if args.triplets_per_query < 1:
        parser.error(
            "Triplets per query must be positive"
        )

    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error(
            "Margin must be finite and nonnegative"
        )

    if (
        Path(args.output).exists()
        or Path(args.csv).exists()
    ):
        parser.error(
            "An output file already exists. "
            "Choose new filenames."
        )

    torch.set_num_threads(
        args.threads
    )

    set_seed(42)

    print(
        "Stage 6.4 — Independent test evaluation",
        flush=True,
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    model, checkpoint = load_best_model(
        args.checkpoint
    )

    print(
        f"Selected epoch: {checkpoint['epoch']}"
    )

    print(
        f"Best validation mAP: "
        f"{checkpoint['best_map']:.4%}"
    )

    print(
        "\nLoading held-out CIFAR-100 test split...",
        flush=True,
    )

    _, _, test_loader = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    gallery_loader = DataLoader(
        test_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        "\nExtracting test embeddings...",
        flush=True,
    )

    embeddings, labels, ids = extract_embeddings(
        model,
        gallery_loader,
        expected_dim=128,
    )

    if len(labels) != 6000:
        raise RuntimeError(
            "Expected exactly 6,000 test images"
        )

    query_positions = select_stratified_queries(
        labels,
        samples_per_class=args.queries_per_class,
        seed=42,
    )

    print(
        f"Test gallery: {len(labels)} images"
    )
    print(
        f"Test queries: {len(query_positions)}"
    )

    print(
        "\nComputing retrieval metrics...",
        flush=True,
    )

    retrieval = evaluate_retrieval(
        query_embeddings=embeddings[query_positions],
        query_labels=labels[query_positions],
        query_ids=ids[query_positions],
        gallery_embeddings=embeddings,
        gallery_labels=labels,
        gallery_ids=ids,
        ks=(1, 5, 10),
        batch_size=32,
    )

    print(
        "\nComputing test triplet ordering...",
        flush=True,
    )

    ordering = compute_triplet_ordering(
        embeddings,
        labels,
        query_positions,
        trials_per_query=args.triplets_per_query,
        margin=args.margin,
        seed=2026,
    )

    results = {
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_best_map": checkpoint["best_map"],
        **retrieval,
        **ordering,
    }

    print("\nHeld-out test results")
    print("=" * 52)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")

    save_results(
        args.output,
        args.csv,
        results,
    )

    print(
        f"\nJSON saved: {args.output}",
        flush=True,
    )
    print(
        f"CSV saved: {args.csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
