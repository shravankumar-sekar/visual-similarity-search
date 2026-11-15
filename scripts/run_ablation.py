
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from loss import batch_hard_triplet_loss
from model import EmbeddingNet
from optimizers import Adam, MomentumSGD, NAdam
from random_mining import random_triplet_loss
from retrieval import evaluate_retrieval
from semi_hard_mining import semi_hard_triplet_loss
from scripts.evaluate_retrieval import (
    extract_embeddings,
    select_stratified_queries,
)
from scripts.train_full import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from train import load_cifar100, set_seed


FIELDS = [
    "epoch",
    "train_loss",
    "train_images",
    "val_map",
    "val_hit_at_1",
    "val_hit_at_5",
    "val_hit_at_10",
    "val_spread",
    "spread_ratio",
    "collapsed",
    "epoch_seconds",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-epoch CIFAR-100 ablation runner"
    )

    parser.add_argument(
        "--dimension",
        type=int,
        choices=[64, 128, 256],
        default=128,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "nadam"],
        default="adam",
    )
    parser.add_argument(
        "--mining",
        choices=["random", "hard", "semi-hard"],
        default="random",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--queries-per-class",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser


def validate_args(args):
    if args.dimension not in (64, 128, 256):
        raise ValueError("Unsupported embedding dimension")

    if not math.isfinite(args.margin) or args.margin <= 0:
        raise ValueError("Margin must be finite and positive")

    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive")

    if args.optimizer not in ("sgd", "adam", "nadam"):
        raise ValueError("Unknown optimizer")

    if args.mining not in ("random", "hard", "semi-hard"):
        raise ValueError("Unknown mining method")

    for field in (
        "epochs",
        "threads",
        "queries_per_class",
        "eval_batch_size",
        "search_batch_size",
        "log_every",
    ):
        if getattr(args, field) < 1:
            raise ValueError(f"{field} must be positive")

    if args.queries_per_class > 60:
        raise ValueError(
            "Validation has only 60 images per class"
        )

    if args.seed < 0:
        raise ValueError("Seed must be non-negative")


def make_config(args):
    # The number of target epochs is excluded so
    # a short run can later be resumed.
    return {
        "architecture": "resnet18_scratch",
        "embedding_dim": args.dimension,
        "margin": float(args.margin),
        "optimizer": args.optimizer,
        "mining": args.mining,
        "lr": float(args.lr),
        "seed": args.seed,
        "classes_per_batch": 16,
        "samples_per_class": 4,
        "queries_per_class": args.queries_per_class,
    }


def make_optimizer(name, parameters, lr):
    if name == "adam":
        return Adam(parameters, lr=lr)

    if name == "nadam":
        return NAdam(parameters, lr=lr)

    if name == "sgd":
        return MomentumSGD(
            parameters,
            lr=lr,
            momentum=0.9,
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def calculate_loss(
    embeddings,
    labels,
    mining,
    margin,
):
    if mining == "random":
        return random_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

    if mining == "hard":
        return batch_hard_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

    if mining == "semi-hard":
        return semi_hard_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

    raise ValueError(f"Unsupported mining: {mining}")


def train_ablation_epoch(
    model,
    loader,
    optimizer,
    epoch,
    mining,
    margin,
    log_every=100,
):
    if log_every < 1:
        raise ValueError("log_every must be positive")

    model.train()
    loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_images = 0

    start = time.perf_counter()

    for batch_number, (images, labels) in enumerate(
        loader,
        start=1,
    ):
        optimizer.zero_grad()

        embeddings = model(images)

        loss = calculate_loss(
            embeddings,
            labels,
            mining,
            margin,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, "
                f"batch {batch_number}"
            )

        loss.backward()
        optimizer.step()

        count = len(labels)
        total_loss += loss.item() * count
        total_images += count

        if (
            batch_number == 1
            or batch_number % log_every == 0
            or batch_number == len(loader)
        ):
            elapsed = time.perf_counter() - start

            print(
                f"Epoch {epoch} | "
                f"batch {batch_number}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    if total_images != len(loader.dataset):
        raise RuntimeError(
            f"Incomplete epoch: {total_images} "
            f"of {len(loader.dataset)} images"
        )

    return {
        "loss": total_loss / total_images,
        "images": total_images,
    }


@torch.no_grad()
def evaluate_ablation(
    model,
    gallery_loader,
    dimension,
    queries_per_class=5,
    search_batch_size=32,
    seed=42,
):
    model.eval()

    embeddings, labels, ids = extract_embeddings(
        model,
        gallery_loader,
        expected_dim=dimension,
    )

    indices = select_stratified_queries(
        labels,
        samples_per_class=queries_per_class,
        seed=seed,
    )

    retrieval = evaluate_retrieval(
        query_embeddings=embeddings[indices],
        query_labels=labels[indices],
        query_ids=ids[indices],
        gallery_embeddings=embeddings,
        gallery_labels=labels,
        gallery_ids=ids,
        ks=(1, 5, 10),
        batch_size=search_batch_size,
    )

    # All embeddings are L2-normalized.
    # A small spread warns of possible collapse.
    spread = float(
        np.linalg.norm(
            np.std(embeddings, axis=0)
        )
    )

    return {
        **retrieval,
        "spread": spread,
    }


def is_collapsed(
    current_spread,
    baseline_spread,
    threshold=0.25,
):
    if baseline_spread <= 0:
        raise ValueError("Baseline spread must be positive")

    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")

    return current_spread < baseline_spread * threshold


def make_row(
    epoch,
    training,
    validation,
    baseline_spread,
    seconds,
):
    ratio = (
        validation["spread"] / baseline_spread
    )

    return {
        "epoch": epoch,
        "train_loss": training["loss"],
        "train_images": training["images"],
        "val_map": validation["map"],
        "val_hit_at_1": validation["hit_at_1"],
        "val_hit_at_5": validation["hit_at_5"],
        "val_hit_at_10": validation["hit_at_10"],
        "val_spread": validation["spread"],
        "spread_ratio": ratio,
        "collapsed": is_collapsed(
            validation["spread"],
            baseline_spread,
        ),
        "epoch_seconds": seconds,
    }


def save_csv(path, history):
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
            writer.writerows(history)

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    run_dir = Path(args.run_dir)

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    if args.resume:
        if not last_path.is_file():
            parser.error(
                f"Missing resume checkpoint: {last_path}"
            )
    elif run_dir.exists() and any(run_dir.iterdir()):
        parser.error(
            f"Run directory is not empty: {run_dir}"
        )

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    config = make_config(args)

    print("\nStage 7 — Ablation training")
    print("-" * 55)
    print(f"Embedding dimension: {args.dimension}")
    print(f"Margin: {args.margin}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Mining: {args.mining}")
    print(f"Learning rate: {args.lr}")
    print(f"Threads: {args.threads}")
    print(f"Target epochs: {args.epochs}")
    print(f"Output directory: {run_dir}")

    train_loader, val_loader, _ = load_cifar100(
        seed=args.seed,
        classes_per_batch=16,
        samples_per_class=4,
    )

    if len(train_loader.dataset) != 48000:
        raise RuntimeError(
            "Expected 48,000 training images"
        )

    if len(val_loader.dataset) != 6000:
        raise RuntimeError(
            "Expected 6,000 validation images"
        )

    gallery_loader = DataLoader(
        val_loader.dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = EmbeddingNet(
        embedding_dim=args.dimension
    )

    optimizer = make_optimizer(
        args.optimizer,
        model.parameters(),
        args.lr,
    )

    history = []
    best_map = float("-inf")
    best_epoch = 0
    start_epoch = 1
    baseline = None

    if args.resume:
        restored = load_training_checkpoint(
            last_path,
            model,
            optimizer,
            expected_config=config,
        )

        history = restored["history"]
        start_epoch = restored["epoch"] + 1
        best_map = restored["best_map"]
        best_epoch = restored["best_epoch"]
        baseline = restored["baseline_metrics"]

        save_csv(metrics_path, history)

        print(
            f"\nResuming at epoch {start_epoch}",
            flush=True,
        )

    else:
        print(
            "\nEvaluating untrained baseline...",
            flush=True,
        )

        baseline = evaluate_ablation(
            model,
            gallery_loader,
            dimension=args.dimension,
            queries_per_class=args.queries_per_class,
            search_batch_size=args.search_batch_size,
            seed=args.seed,
        )

        print(
            f"Baseline mAP: {baseline['map']:.4%}"
        )
        print(
            f"Baseline spread: {baseline['spread']:.6f}"
        )

    if start_epoch > args.epochs:
        print("All requested epochs already completed.")
        return

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        start = time.perf_counter()

        print(
            f"\nStarting full epoch {epoch}/{args.epochs}",
            flush=True,
        )

        training = train_ablation_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            mining=args.mining,
            margin=args.margin,
            log_every=args.log_every,
        )

        print(
            "\nEvaluating validation retrieval...",
            flush=True,
        )

        validation = evaluate_ablation(
            model,
            gallery_loader,
            dimension=args.dimension,
            queries_per_class=args.queries_per_class,
            search_batch_size=args.search_batch_size,
            seed=args.seed,
        )

        row = make_row(
            epoch,
            training,
            validation,
            baseline_spread=baseline["spread"],
            seconds=time.perf_counter() - start,
        )

        history.append(row)

        improved = (
            not row["collapsed"]
            and validation["map"] > best_map
        )

        if improved:
            best_map = validation["map"]
            best_epoch = epoch

            save_training_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                history,
                config,
                best_map,
                best_epoch,
                baseline,
            )

        save_training_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            history,
            config,
            best_map,
            best_epoch,
            baseline,
        )

        save_csv(
            metrics_path,
            history,
        )

        print("\nEpoch summary")
        print("-" * 55)
        print(f"Train loss: {row['train_loss']:.6f}")
        print(f"Validation mAP: {row['val_map']:.4%}")
        print(f"Hit@1: {row['val_hit_at_1']:.4%}")
        print(f"Hit@5: {row['val_hit_at_5']:.4%}")
        print(f"Embedding spread: {row['val_spread']:.6f}")
        print(f"Spread ratio: {row['spread_ratio']:.3f}")
        print(f"Collapse warning: {row['collapsed']}")

        if best_epoch:
            print(
                f"Best mAP: {best_map:.4%} "
                f"(epoch {best_epoch})"
            )
        else:
            print("No non-collapsed best checkpoint yet")

        if row["collapsed"]:
            print(
                "\nEARLY STOP: embedding spread "
                "fell below 25% of baseline.",
                flush=True,
            )
            break

    print("\nAblation run finished.")
    print(f"Metrics: {metrics_path}")
    print(f"Last checkpoint: {last_path}")

    if best_path.exists():
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
