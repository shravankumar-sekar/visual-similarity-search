
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

from loss import batch_hard_triplet_loss
from model import EmbeddingNet
from optimizers import Adam
from random_mining import random_triplet_loss
from scripts.diagnose_embeddings import diagnose
from scripts.experiment_random_mining import (
    load_experiment_checkpoint,
    save_experiment_checkpoint,
)
from train import load_cifar100, set_seed


RESULT_FIELDS = [
    "branch",
    "epoch",
    "train_loss",
    "random_ordering",
    "hard_ordering",
    "positive_distance",
    "negative_distance",
    "embedding_spread",
    "collapsed",
]


def detect_collapse(
    current_spread,
    baseline_spread,
    threshold=0.25,
):
    if baseline_spread <= 0:
        raise ValueError(
            "Baseline spread must be positive"
        )

    if threshold <= 0 or threshold >= 1:
        raise ValueError(
            "Threshold must be between 0 and 1"
        )

    return (
        current_spread
        < baseline_spread * threshold
    )


def calculate_loss(
    embeddings,
    labels,
    strategy,
    margin=0.3,
):
    if strategy == "random":
        return random_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

    if strategy == "hard":
        return batch_hard_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

    raise ValueError(
        f"Unknown mining strategy: {strategy}"
    )


def train_branch_epoch(
    model,
    loader,
    optimizer,
    strategy,
    steps,
    epoch,
    margin=0.3,
    log_every=20,
):
    if strategy not in ("random", "hard"):
        raise ValueError(
            f"Unknown mining strategy: {strategy}"
        )

    if steps < 1 or steps > len(loader):
        raise ValueError(
            "Invalid number of training steps"
        )

    if log_every < 1:
        raise ValueError(
            "Logging interval must be positive"
        )

    model.train()

    loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    completed = 0

    for images, labels in loader:
        if completed >= steps:
            break

        optimizer.zero_grad()

        embeddings = model(images)

        loss = calculate_loss(
            embeddings,
            labels,
            strategy=strategy,
            margin=margin,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Non-finite training loss"
            )

        loss.backward()
        optimizer.step()

        completed += 1
        total_loss += loss.item()

        if (
            completed == 1
            or completed % log_every == 0
            or completed == steps
        ):
            print(
                f"{strategy} | "
                f"epoch {epoch} | "
                f"batch {completed}/{steps} | "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    if completed != steps:
        raise RuntimeError(
            "DataLoader ended unexpectedly"
        )

    return total_loss / completed


def save_results(path, rows):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temp_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=RESULT_FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)

        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def make_result_row(
    branch,
    epoch,
    diagnostic,
    train_loss=None,
    collapsed=False,
):
    return {
        "branch": branch,
        "epoch": epoch,
        "train_loss": train_loss,
        "random_ordering": diagnostic[
            "random_ordering"
        ],
        "hard_ordering": diagnostic[
            "hard_ordering"
        ],
        "positive_distance": diagnostic[
            "positive_distance"
        ],
        "negative_distance": diagnostic[
            "negative_distance"
        ],
        "embedding_spread": diagnostic[
            "embedding_spread"
        ],
        "collapsed": collapsed,
    }


def load_warmup_branch(
    checkpoint_path,
    new_lr,
    expected_seed=42,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint.get("config")

    if not isinstance(config, dict):
        raise ValueError(
            "Warm-up checkpoint has no configuration"
        )

    if config.get("mining") != "random":
        raise ValueError(
            "Expected a random-mining checkpoint"
        )

    if config.get("optimizer") != "adam":
        raise ValueError(
            "Expected a custom Adam checkpoint"
        )

    if config.get("seed") != expected_seed:
        raise ValueError(
            "Warm-up checkpoint seed mismatch"
        )

    if config.get("embedding_dim") != 128:
        raise ValueError(
            "Expected 128-dimensional embeddings"
        )

    if (
        config.get("classes_per_batch") != 16
        or config.get("samples_per_class") != 4
    ):
        raise ValueError(
            "Expected a 16 x 4 warm-up batch"
        )

    model = EmbeddingNet(
        embedding_dim=128
    )

    optimizer = Adam(
        model.parameters(),
        lr=config["lr"],
    )

    restored = load_experiment_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config=config,
    )

    if restored["epoch"] < 1:
        raise ValueError(
            "Warm-up checkpoint has no completed epoch"
        )

    if not hasattr(optimizer, "lr"):
        raise AttributeError(
            "Custom Adam must expose a writable "
            "'lr' attribute for fine-tuning"
        )

    # Preserve Adam moments while reducing its
    # learning rate for the continuation experiment.
    optimizer.lr = new_lr

    return (
        model,
        optimizer,
        restored,
        config,
    )


def print_result(row):
    print(
        f"\n{row['branch']} | "
        f"continuation epoch {row['epoch']}"
    )
    print("-" * 48)

    for field in (
        "train_loss",
        "random_ordering",
        "hard_ordering",
        "positive_distance",
        "negative_distance",
        "embedding_spread",
    ):
        value = row[field]

        if value is not None:
            print(
                f"{field:22s}: {value:.6f}"
            )

    print(
        f"Collapse warning: {row['collapsed']}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare random and hard mining "
            "from the same warm-up checkpoint"
        )
    )

    parser.add_argument(
        "--warmup",
        default=(
            "checkpoints/"
            "random_mining_stage59.pt"
        ),
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
        "--output",
        default="results/mining_comparison.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
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

    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error(
            "--lr must be finite and positive"
        )

    if args.diagnostic_batches < 1:
        parser.error(
            "--diagnostic-batches must be positive"
        )

    if args.log_every < 1:
        parser.error(
            "--log-every must be positive"
        )

    warmup_path = Path(args.warmup)

    if not warmup_path.is_file():
        parser.error(
            f"Warm-up checkpoint missing: {warmup_path}"
        )

    output_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)

    random_checkpoint = (
        checkpoint_dir / "stage510_random.pt"
    )
    hard_checkpoint = (
        checkpoint_dir / "stage510_hard.pt"
    )

    protected_paths = [
        warmup_path.resolve(),
        output_path.resolve(),
        random_checkpoint.resolve(),
        hard_checkpoint.resolve(),
    ]

    if len(set(protected_paths)) != len(protected_paths):
        parser.error(
            "Input and output paths must be distinct"
        )

    for path in (
        output_path,
        random_checkpoint,
        hard_checkpoint,
    ):
        if path.exists():
            parser.error(
                f"Output already exists: {path}. "
                "Choose unused paths before running."
            )

    torch.set_num_threads(args.threads)
    set_seed(42)

    print(
        "Stage 5.10: controlled mining comparison"
    )
    print(
        f"Warm-up checkpoint: {warmup_path}"
    )
    print(
        f"CPU threads: {torch.get_num_threads()}"
    )
    print(
        f"Continuation learning rate: {args.lr}"
    )

    train_loader, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    if args.steps > len(train_loader):
        parser.error(
            "Steps exceed full training batches"
        )

    if args.diagnostic_batches > len(val_loader):
        parser.error(
            "Diagnostic batch count exceeds "
            "validation batches"
        )

    rows = []

    # Branches run sequentially to avoid keeping
    # two ResNet-18 models in memory simultaneously.
    for strategy, checkpoint_path in (
        ("random", random_checkpoint),
        ("hard", hard_checkpoint),
    ):
        print(
            f"\n{'=' * 55}\n"
            f"STARTING {strategy.upper()} BRANCH\n"
            f"{'=' * 55}",
            flush=True,
        )

        # Reloading the same checkpoint ensures
        # identical starting weights, Adam moments
        # and RNG state for the two branches.
        (
            model,
            optimizer,
            restored,
            warmup_config,
        ) = load_warmup_branch(
            warmup_path,
            new_lr=args.lr,
            expected_seed=42,
        )

        baseline = diagnose(
            model,
            val_loader,
            batches=args.diagnostic_batches,
            seed=42,
        )

        baseline_spread = baseline[
            "embedding_spread"
        ]

        baseline_row = make_result_row(
            strategy,
            epoch=0,
            diagnostic=baseline,
        )

        rows.append(baseline_row)
        print_result(baseline_row)

        branch_history = []

        branch_config = {
            "source_checkpoint": str(
                warmup_path.resolve()
            ),
            "source_epoch": restored["epoch"],
            "warmup_config": warmup_config,
            "strategy": strategy,
            "continuation_lr": args.lr,
            "steps": args.steps,
            "seed": 42,
            "margin": 0.3,
            "diagnostic_batches": (
                args.diagnostic_batches
            ),
        }

        for continuation_epoch in range(
            1,
            args.epochs + 1,
        ):
            # Continue the original sampler's
            # epoch sequence after warm-up.
            sampler_epoch = (
                restored["epoch"]
                + continuation_epoch
            )

            loss = train_branch_epoch(
                model,
                train_loader,
                optimizer,
                strategy=strategy,
                steps=args.steps,
                epoch=sampler_epoch,
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

            result = make_result_row(
                strategy,
                epoch=continuation_epoch,
                diagnostic=diagnostic,
                train_loss=loss,
                collapsed=collapsed,
            )

            rows.append(result)
            branch_history.append(result)

            print_result(result)

            save_experiment_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=continuation_epoch,
                history=branch_history,
                config=branch_config,
                baseline=baseline,
            )

            save_results(
                output_path,
                rows,
            )

            print(
                f"Saved branch checkpoint: "
                f"{checkpoint_path}",
                flush=True,
            )

            if collapsed:
                print(
                    "\nSTOPPING THIS BRANCH: "
                    "embedding spread fell below "
                    "25% of its warm-up baseline.",
                    flush=True,
                )
                break

        del model
        del optimizer

    save_results(output_path, rows)

    print(
        "\nComparison completed.",
        flush=True,
    )
    print(
        f"Results: {output_path}"
    )
    print(
        f"Random branch: {random_checkpoint}"
    )
    print(
        f"Hard branch: {hard_checkpoint}"
    )


if __name__ == "__main__":
    main()
