
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from data import stratified_split, CIFAR100Subset
from epoch_sampler import EpochBalancedBatchSampler
from loss import batch_hard_triplet_loss, pairwise_distances
from metrics import save_metrics
from model import EmbeddingNet
from optimizers import MomentumSGD, Adam, NAdam


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-100 similarity embedding model"
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/controlled_test.pt",
    )
    parser.add_argument(
        "--metrics",
        default="results/controlled_test.csv",
    )

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--embedding-dim", type=int, default=128
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "nadam"],
        default="adam",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=0.3)

    parser.add_argument(
        "--classes-per-batch", type=int, default=16
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=4
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Training batches per epoch; 0 means full epoch",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=20,
        help="Validation batches; 0 means full validation",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    return parser


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("Epochs must be positive")

    if args.threads < 1:
        raise ValueError("Threads must be positive")

    if args.seed < 0:
        raise ValueError("Seed must be non-negative")

    if args.embedding_dim < 1:
        raise ValueError("Embedding dimension must be positive")

    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")

    if args.margin < 0:
        raise ValueError("Margin cannot be negative")

    if not 2 <= args.classes_per_batch <= 100:
        raise ValueError(
            "Classes per batch must be between 2 and 100"
        )

    if args.samples_per_class < 2:
        raise ValueError(
            "Samples per class must be at least 2"
        )

    # Training has 480 samples per class.
    # Validation has 60 samples per class.
    if 480 % args.samples_per_class != 0:
        raise ValueError(
            "Training samples per class must divide evenly"
        )

    if 60 % args.samples_per_class != 0:
        raise ValueError(
            "Validation samples per class must divide evenly"
        )

    if args.max_batches < 0 or args.val_batches < 0:
        raise ValueError("Batch limits cannot be negative")

    if args.log_every < 1:
        raise ValueError("Logging interval must be positive")


def make_config(args):
    return {
        "embedding_dim": args.embedding_dim,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "margin": args.margin,
        "classes_per_batch": args.classes_per_batch,
        "samples_per_class": args.samples_per_class,
        "max_batches": args.max_batches,
        "val_batches": args.val_batches,
        "seed": args.seed,
    }


def load_cifar100(
    root="./data",
    seed=42,
    classes_per_batch=16,
    samples_per_class=4,
):
    train = CIFAR100(
        root=root,
        train=True,
        download=True,
    )
    test = CIFAR100(
        root=root,
        train=False,
        download=True,
    )

    images = np.concatenate(
        [train.data, test.data],
        axis=0,
    )

    labels = np.asarray(train.targets + test.targets)

    train_idx, val_idx, test_idx = stratified_split(
        labels,
        seed=seed,
    )

    train_set = CIFAR100Subset(
        images,
        labels,
        train_idx,
        training=True,
    )
    val_set = CIFAR100Subset(
        images,
        labels,
        val_idx,
        training=False,
    )
    test_set = CIFAR100Subset(
        images,
        labels,
        test_idx,
        training=False,
    )

    def make_loader(dataset, indices, sampler_seed):
        sampler = EpochBalancedBatchSampler(
            labels,
            indices,
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class,
            seed=sampler_seed,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
        )

    train_loader = make_loader(train_set, train_idx, seed)
    val_loader = make_loader(val_set, val_idx, seed + 1)
    test_loader = make_loader(test_set, test_idx, seed + 2)

    return train_loader, val_loader, test_loader


def make_optimizer(name, parameters, lr):
    name = name.lower()

    if name == "sgd":
        return MomentumSGD(
            parameters,
            lr=lr,
            momentum=0.9,
        )

    if name == "adam":
        return Adam(parameters, lr=lr)

    if name == "nadam":
        return NAdam(parameters, lr=lr)

    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def triplet_metrics(embeddings, labels, margin=0.3):
    distances = pairwise_distances(embeddings)

    same_class = labels[:, None] == labels[None, :]

    positive_mask = same_class.clone()
    positive_mask.fill_diagonal_(False)

    negative_mask = ~same_class

    valid = (
        positive_mask.any(dim=1)
        & negative_mask.any(dim=1)
    )

    if not valid.any():
        raise ValueError("No valid triplets in batch")

    hardest_positive = distances.masked_fill(
        ~positive_mask,
        float("-inf"),
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        ~negative_mask,
        float("inf"),
    ).min(dim=1).values

    positive_dist = hardest_positive[valid]
    negative_dist = hardest_negative[valid]

    ordering_accuracy = (
        positive_dist < negative_dist
    ).float().mean().item()

    margin_accuracy = (
        positive_dist + margin < negative_dist
    ).float().mean().item()

    # Measures variation between embeddings in a batch.
    spread = embeddings.std(
        dim=0,
        unbiased=False,
    ).norm().item()

    return {
        "ordering_accuracy": ordering_accuracy,
        "margin_accuracy": margin_accuracy,
        "embedding_spread": spread,
    }


def run_epoch(
    model,
    loader,
    optimizer=None,
    margin=0.3,
    max_batches=None,
    log_every=10,
    name="Training",
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_ordering = 0.0
    total_margin = 0.0
    total_spread = 0.0
    total_samples = 0

    total_batches = (
        len(loader)
        if max_batches is None
        else min(len(loader), max_batches)
    )

    if total_batches < 1:
        raise ValueError("No batches selected")

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= total_batches:
            break

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            embeddings = model(images)

            loss = batch_hard_triplet_loss(
                embeddings,
                labels,
                margin=margin,
            )

            if training:
                loss.backward()
                optimizer.step()

        metrics = triplet_metrics(
            embeddings.detach(),
            labels,
            margin=margin,
        )

        batch_size = len(labels)

        total_loss += loss.item() * batch_size
        total_ordering += (
            metrics["ordering_accuracy"] * batch_size
        )
        total_margin += (
            metrics["margin_accuracy"] * batch_size
        )
        total_spread += (
            metrics["embedding_spread"] * batch_size
        )
        total_samples += batch_size

        completed = batch_idx + 1

        if (
            completed == 1
            or completed % log_every == 0
            or completed == total_batches
        ):
            print(
                f"{name} {completed}/{total_batches} | "
                f"loss={loss.item():.4f} | "
                f"ordering={metrics['ordering_accuracy']:.4f} | "
                f"margin={metrics['margin_accuracy']:.4f} | "
                f"spread={metrics['embedding_spread']:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("No samples processed")

    return {
        "loss": total_loss / total_samples,
        "ordering_accuracy": total_ordering / total_samples,
        "margin_accuracy": total_margin / total_samples,
        "embedding_spread": total_spread / total_samples,
        "samples": total_samples,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    margin=0.3,
    max_batches=5,
    log_every=10,
):
    return run_epoch(
        model,
        loader,
        optimizer=optimizer,
        margin=margin,
        max_batches=max_batches,
        log_every=log_every,
        name="Training",
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    margin=0.3,
    max_batches=5,
    name="Validation",
    log_every=10,
):
    return run_epoch(
        model,
        loader,
        optimizer=None,
        margin=margin,
        max_batches=max_batches,
        log_every=log_every,
        name=name,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    history,
    config=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    numpy_state = np.random.get_state()

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "config": config,
        "python_rng": random.getstate(),
        "numpy_rng": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_rng": torch.get_rng_state(),
    }

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def load_checkpoint(
    path,
    model,
    optimizer,
    config=None,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if config is not None:
        if checkpoint.get("config") != config:
            raise ValueError(
                "Checkpoint configuration does not match "
                "the current experiment"
            )

    model.load_state_dict(
        checkpoint["model_state"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    if "python_rng" in checkpoint:
        random.setstate(checkpoint["python_rng"])

    if "numpy_rng" in checkpoint:
        state = checkpoint["numpy_rng"]

        np.random.set_state((
            state[0],
            np.array(state[1], dtype=np.uint32),
            state[2],
            state[3],
            state[4],
        ))

    if "torch_rng" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng"]
        )

    return checkpoint["epoch"], checkpoint["history"]


def print_summary(epoch, train_metrics, val_metrics):
    print(f"\nEpoch {epoch} summary")
    print("-" * 54)

    print(
        f"Training loss:      {train_metrics['loss']:.4f}"
    )
    print(
        f"Validation loss:    {val_metrics['loss']:.4f}"
    )
    print(
        f"Train ordering:     "
        f"{train_metrics['ordering_accuracy']:.4f}"
    )
    print(
        f"Validation ordering:"
        f" {val_metrics['ordering_accuracy']:.4f}"
    )
    print(
        f"Validation margin:  "
        f"{val_metrics['margin_accuracy']:.4f}"
    )
    print(
        f"Validation spread:  "
        f"{val_metrics['embedding_spread']:.4f}"
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    config = make_config(args)

    train_limit = (
        None if args.max_batches == 0
        else args.max_batches
    )
    val_limit = (
        None if args.val_batches == 0
        else args.val_batches
    )

    print("Controlled learning experiment", flush=True)
    print(f"CPU threads: {torch.get_num_threads()}")
    print(
        f"Batch size: "
        f"{args.classes_per_batch * args.samples_per_class}"
    )
    print(
        f"Embedding dimension: {args.embedding_dim}"
    )
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr}")
    print(f"Margin: {args.margin}")

    print("\nLoading CIFAR-100...", flush=True)

    train_loader, val_loader, _ = load_cifar100(
        seed=args.seed,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
    )

    print(
        f"Training batches available: {len(train_loader)}"
    )
    print(
        f"Validation batches available: {len(val_loader)}"
    )

    model = EmbeddingNet(
        embedding_dim=args.embedding_dim
    )

    optimizer = make_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
    )

    checkpoint_path = Path(args.checkpoint)

    start_epoch = 1
    history = []

    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        last_epoch, history = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config=config,
        )

        start_epoch = last_epoch + 1

        print(
            f"Resuming from epoch {start_epoch}",
            flush=True,
        )
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}. "
                "Use --resume or another checkpoint path."
            )

        print("Starting fresh experiment.", flush=True)

    if start_epoch > args.epochs:
        print("Requested epochs already completed.")
        return

    print(
        "\nMeasuring validation before further training...",
        flush=True,
    )

    baseline = evaluate(
        model,
        val_loader,
        margin=args.margin,
        max_batches=val_limit,
        name="Baseline validation",
        log_every=args.log_every,
    )

    print(
        f"\nStarting validation loss: "
        f"{baseline['loss']:.4f}"
    )
    print(
        f"Starting validation ordering: "
        f"{baseline['ordering_accuracy']:.4f}"
    )
    print(
        f"Starting embedding spread: "
        f"{baseline['embedding_spread']:.4f}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader.batch_sampler.set_epoch(epoch)

        print(
            f"\nStarting epoch {epoch}/{args.epochs}",
            flush=True,
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            margin=args.margin,
            max_batches=train_limit,
            log_every=args.log_every,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            margin=args.margin,
            max_batches=val_limit,
            name="Validation",
            log_every=args.log_every,
        )

        print_summary(
            epoch,
            train_metrics,
            val_metrics,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ordering": train_metrics["ordering_accuracy"],
            "train_margin": train_metrics["margin_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_ordering": val_metrics["ordering_accuracy"],
            "val_margin": val_metrics["margin_accuracy"],
        }

        history.append(row)

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            history,
            config=config,
        )

        save_metrics(
            history,
            args.metrics,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}",
            flush=True,
        )
        print(
            f"Metrics saved: {args.metrics}",
            flush=True,
        )

    print("\nControlled experiment completed.", flush=True)


if __name__ == "__main__":
    main()
