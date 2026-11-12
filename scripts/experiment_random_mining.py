
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from model import EmbeddingNet
from optimizers import Adam
from random_mining import random_triplet_loss
from train import load_cifar100, set_seed
from scripts.diagnose_embeddings import diagnose


CSV_FIELDS = [
    "epoch",
    "train_loss",
    "random_ordering",
    "hard_ordering",
    "positive_distance",
    "negative_distance",
    "embedding_spread",
]


def make_experiment_config(
    lr=0.0001,
    steps=100,
    seed=42,
    embedding_dim=128,
    classes_per_batch=16,
    samples_per_class=4,
    margin=0.3,
    diagnostic_batches=10,
):
    return {
        "optimizer": "adam",
        "lr": float(lr),
        "steps": int(steps),
        "seed": int(seed),
        "embedding_dim": int(embedding_dim),
        "classes_per_batch": int(classes_per_batch),
        "samples_per_class": int(samples_per_class),
        "margin": float(margin),
        "diagnostic_batches": int(diagnostic_batches),
        "mining": "random",
    }


def save_experiment_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    history,
    config,
    baseline,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    numpy_state = np.random.get_state()

    checkpoint = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "config": config,
        "baseline": baseline,
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

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        torch.save(
            checkpoint,
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_experiment_checkpoint(
    path,
    model,
    optimizer,
    config,
):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("format_version") != 1:
        raise ValueError(
            "Unsupported checkpoint format"
        )

    if checkpoint.get("config") != config:
        raise ValueError(
            "Checkpoint configuration does not "
            "match the current experiment"
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    random.setstate(
        checkpoint["python_rng"]
    )

    numpy_state = checkpoint["numpy_rng"]

    np.random.set_state((
        numpy_state[0],
        np.asarray(
            numpy_state[1],
            dtype=np.uint32,
        ),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    ))

    torch.set_rng_state(
        checkpoint["torch_rng"]
    )

    return {
        "epoch": checkpoint["epoch"],
        "history": checkpoint["history"],
        "baseline": checkpoint["baseline"],
    }


def save_experiment_metrics(path, history):
    path = Path(path)
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
            writer.writerows(history)

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def train_epoch(
    model,
    loader,
    optimizer,
    steps,
    epoch,
    margin=0.3,
    log_every=20,
):
    if steps < 1 or steps > len(loader):
        raise ValueError(
            "Steps must be between 1 and "
            "the number of batches"
        )

    if log_every < 1:
        raise ValueError(
            "Logging interval must be positive"
        )

    model.train()

    loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    completed = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= steps:
            break

        optimizer.zero_grad()

        embeddings = model(images)

        loss = random_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite training loss detected"
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        completed += 1

        if (
            completed == 1
            or completed % log_every == 0
            or completed == steps
        ):
            print(
                f"Epoch {epoch} | "
                f"batch {completed}/{steps} | "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    if completed != steps:
        raise RuntimeError(
            f"Expected {steps} batches, "
            f"processed {completed}"
        )

    return total_loss / completed


def print_diagnostics(title, result):
    print(f"\n{title}")
    print("-" * 46)

    for key in (
        "random_ordering",
        "hard_ordering",
        "positive_distance",
        "negative_distance",
        "embedding_spread",
    ):
        print(
            f"{key:22s}: {result[key]:.6f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Checkpointed random-triplet "
            "training experiment"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--diagnostic-batches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/random_mining_stage59.pt",
    )
    parser.add_argument(
        "--output",
        default="results/random_mining_stage59.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error(
            "--epochs must be positive"
        )

    if args.steps < 1:
        parser.error(
            "--steps must be positive"
        )

    if args.threads < 1:
        parser.error(
            "--threads must be positive"
        )

    if args.lr <= 0:
        parser.error(
            "--lr must be positive"
        )

    if args.seed < 0:
        parser.error(
            "--seed must be non-negative"
        )

    if args.diagnostic_batches < 1:
        parser.error(
            "--diagnostic-batches must be positive"
        )

    if args.log_every < 1:
        parser.error(
            "--log-every must be positive"
        )

    checkpoint_path = Path(
        args.checkpoint
    )
    output_path = Path(
        args.output
    )

    if not args.resume and checkpoint_path.exists():
        parser.error(
            f"Checkpoint already exists: "
            f"{checkpoint_path}. "
            "Use --resume or a different path."
        )

    if not args.resume and output_path.exists():
        parser.error(
            f"Metrics file already exists: "
            f"{output_path}. "
            "Choose a different output path."
        )

    if args.resume and not checkpoint_path.is_file():
        parser.error(
            f"Cannot resume: "
            f"{checkpoint_path} does not exist"
        )

    torch.set_num_threads(
        args.threads
    )
    set_seed(
        args.seed
    )

    config = make_experiment_config(
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
    )

    print(
        "Checkpointed random-mining experiment",
        flush=True,
    )
    print(
        f"CPU threads: {torch.get_num_threads()}"
    )
    print(
        f"Training steps per epoch: {args.steps}"
    )
    print(
        f"Learning rate: {args.lr}"
    )
    print(
        f"Target epochs: {args.epochs}"
    )

    print(
        "\nLoading CIFAR-100...",
        flush=True,
    )

    train_loader, val_loader, _ = load_cifar100(
        seed=args.seed,
        classes_per_batch=config["classes_per_batch"],
        samples_per_class=config["samples_per_class"],
    )

    if args.steps > len(train_loader):
        parser.error(
            "--steps exceeds the number "
            "of training batches"
        )

    if args.diagnostic_batches > len(val_loader):
        parser.error(
            "--diagnostic-batches exceeds "
            "the validation loader length"
        )

    model = EmbeddingNet(
        embedding_dim=config["embedding_dim"]
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
    )

    history = []
    start_epoch = 1

    if args.resume:
        restored = load_experiment_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config,
        )

        last_epoch = restored["epoch"]
        history = restored["history"]
        baseline = restored["baseline"]

        start_epoch = last_epoch + 1

        print(
            f"\nRestored checkpoint from "
            f"epoch {last_epoch}",
            flush=True,
        )

        # Recover the CSV from checkpoint history,
        # including when the earlier process stopped
        # between checkpoint and CSV saving.
        save_experiment_metrics(
            output_path,
            history,
        )

    else:
        print(
            "\nMeasuring initial validation...",
            flush=True,
        )

        baseline = diagnose(
            model,
            val_loader,
            batches=args.diagnostic_batches,
            seed=args.seed,
        )

    print_diagnostics(
        "Original baseline",
        baseline,
    )

    if start_epoch > args.epochs:
        print(
            "\nAll requested epochs already completed."
        )
        return

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print(
            f"\nStarting epoch "
            f"{epoch}/{args.epochs}",
            flush=True,
        )

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            steps=args.steps,
            epoch=epoch,
            margin=config["margin"],
            log_every=args.log_every,
        )

        diagnostic = diagnose(
            model,
            val_loader,
            batches=args.diagnostic_batches,
            seed=args.seed,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **diagnostic,
        }

        print_diagnostics(
            f"Epoch {epoch} validation",
            diagnostic,
        )

        print(
            f"Average training loss: "
            f"{train_loss:.6f}",
            flush=True,
        )

        history.append(row)

        save_experiment_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            history,
            config,
            baseline,
        )

        save_experiment_metrics(
            output_path,
            history,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}",
            flush=True,
        )
        print(
            f"Metrics saved: {output_path}",
            flush=True,
        )

    print(
        "\nExperiment completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    main()
