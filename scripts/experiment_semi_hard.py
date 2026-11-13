
import argparse
import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from scripts.compare_mining import (
    detect_collapse,
    load_warmup_branch,
)
from scripts.diagnose_embeddings import diagnose
from scripts.experiment_random_mining import (
    save_experiment_checkpoint,
)
from semi_hard_mining import semi_hard_triplet_loss
from train import load_cifar100, set_seed


FIELDS = [
    "epoch",
    "train_loss",
    "semi_hard_fraction",
    "fallback_fraction",
    "active_fraction",
    "random_ordering",
    "hard_ordering",
    "positive_distance",
    "negative_distance",
    "embedding_spread",
    "collapsed",
]


def train_semi_hard_epoch(
    model,
    loader,
    optimizer,
    epoch,
    steps=100,
    margin=0.3,
    log_every=20,
):
    if not 1 <= steps <= len(loader):
        raise ValueError("Invalid training step count")

    if log_every < 1:
        raise ValueError("Invalid log interval")

    model.train()
    loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    totals = {
        "semi_hard_fraction": 0.0,
        "fallback_fraction": 0.0,
        "active_fraction": 0.0,
    }

    completed = 0

    for images, labels in loader:
        if completed >= steps:
            break

        optimizer.zero_grad()

        embeddings = model(images)

        loss, stats = semi_hard_triplet_loss(
            embeddings,
            labels,
            margin=margin,
            return_stats=True,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite semi-hard training loss"
            )

        loss.backward()
        optimizer.step()

        completed += 1
        total_loss += loss.item()

        for key in totals:
            totals[key] += stats[key]

        if (
            completed == 1
            or completed % log_every == 0
            or completed == steps
        ):
            print(
                f"Epoch {epoch} | "
                f"batch {completed}/{steps} | "
                f"loss={loss.item():.4f} | "
                f"semi-hard={stats['semi_hard_fraction']:.3f} | "
                f"fallback={stats['fallback_fraction']:.3f}",
                flush=True,
            )

    if completed != steps:
        raise RuntimeError(
            "Training ended before all requested steps"
        )

    return {
        "train_loss": total_loss / completed,
        **{
            key: value / completed
            for key, value in totals.items()
        },
    }


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temporary.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS,
            )

            writer.writeheader()
            writer.writerows(rows)

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def make_row(
    epoch,
    diagnostic,
    training=None,
    collapsed=False,
):
    row = {
        "epoch": epoch,
        "train_loss": None,
        "semi_hard_fraction": None,
        "fallback_fraction": None,
        "active_fraction": None,
        **diagnostic,
        "collapsed": collapsed,
    }

    if training is not None:
        row.update(training)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5.11 semi-hard mining experiment"
    )

    parser.add_argument(
        "--warmup",
        default="checkpoints/random_mining_stage59.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
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
        default=0.00003,
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
        default="checkpoints/stage511_semi_hard.pt",
    )
    parser.add_argument(
        "--output",
        default="results/stage511_semi_hard.csv",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.steps < 1:
        parser.error("Epochs and steps must be positive")

    if args.threads < 1:
        parser.error("Threads must be positive")

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be finite and positive")

    if args.diagnostic_batches < 1:
        parser.error("Diagnostic batches must be positive")

    if args.log_every < 1:
        parser.error("Log interval must be positive")

    warmup = Path(args.warmup)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)

    if not warmup.is_file():
        parser.error(
            f"Warm-up checkpoint not found: {warmup}"
        )

    resolved_paths = [
        path.resolve()
        for path in (warmup, checkpoint, output)
    ]

    if len(set(resolved_paths)) != 3:
        parser.error(
            "Warm-up and output paths must be different"
        )

    for path in (checkpoint, output):
        if path.exists():
            parser.error(
                f"Output already exists: {path}. "
                "Use a new filename."
            )

    torch.set_num_threads(args.threads)
    set_seed(42)

    print("Stage 5.11 — semi-hard mining")
    print(f"Warm-up: {warmup}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Learning rate: {args.lr}")

    train_loader, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    if args.steps > len(train_loader):
        parser.error("Too many training steps")

    if args.diagnostic_batches > len(val_loader):
        parser.error("Too many diagnostic batches")

    (
        model,
        optimizer,
        restored,
        warmup_config,
    ) = load_warmup_branch(
        warmup,
        new_lr=args.lr,
    )

    baseline = diagnose(
        model,
        val_loader,
        batches=args.diagnostic_batches,
        seed=42,
    )

    baseline_spread = baseline["embedding_spread"]

    print("\nWarm-up baseline:")
    print(baseline, flush=True)

    rows = [
        make_row(
            epoch=0,
            diagnostic=baseline,
        )
    ]

    history = []

    config = {
        "strategy": "semi-hard",
        "source_checkpoint": str(warmup.resolve()),
        "source_epoch": restored["epoch"],
        "warmup_config": warmup_config,
        "optimizer": "adam",
        "continuation_lr": args.lr,
        "steps": args.steps,
        "seed": 42,
        "margin": 0.3,
        "diagnostic_batches": args.diagnostic_batches,
    }

    for epoch in range(1, args.epochs + 1):
        sampler_epoch = restored["epoch"] + epoch

        print(
            f"\nContinuation epoch {epoch}/{args.epochs}",
            flush=True,
        )

        training = train_semi_hard_epoch(
            model,
            train_loader,
            optimizer,
            epoch=sampler_epoch,
            steps=args.steps,
            margin=0.3,
            log_every=args.log_every,
        )

        diagnostic = diagnose(
            model,
            val_loader,
            batches=args.diagnostic_batches,
            seed=42,
        )

        collapsed = detect_collapse(
            diagnostic["embedding_spread"],
            baseline_spread,
            threshold=0.25,
        )

        row = make_row(
            epoch=epoch,
            diagnostic=diagnostic,
            training=training,
            collapsed=collapsed,
        )

        rows.append(row)
        history.append(row)

        print("\nEpoch results:")
        for key, value in row.items():
            print(f"{key}: {value}", flush=True)

        save_experiment_checkpoint(
            checkpoint,
            model,
            optimizer,
            epoch=epoch,
            history=history,
            config=config,
            baseline=baseline,
        )

        save_csv(output, rows)

        print(
            f"Saved checkpoint: {checkpoint}",
            flush=True,
        )
        print(
            f"Saved metrics: {output}",
            flush=True,
        )

        if collapsed:
            print(
                "\nEARLY STOP: embedding spread "
                "fell below 25% of baseline.",
                flush=True,
            )
            break

    print("\nStage 5.11 experiment complete.")


if __name__ == "__main__":
    main()
