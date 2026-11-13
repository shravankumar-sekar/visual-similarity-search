
import argparse
import csv
import math
import random
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

from model import EmbeddingNet
from optimizers import Adam
from random_mining import random_triplet_loss
from retrieval import evaluate_retrieval
from scripts.evaluate_retrieval import (
    extract_embeddings,
    select_stratified_queries,
)
from train import load_cifar100, set_seed


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_images",
    "val_recall_at_1",
    "val_recall_at_5",
    "val_recall_at_10",
    "val_hit_at_1",
    "val_hit_at_5",
    "val_hit_at_10",
    "val_map",
    "epoch_seconds",
]


def make_config(args):
    """
    Settings that must remain unchanged
    when resuming an experiment.

    The target number of epochs is deliberately
    excluded, allowing a 3-epoch experiment
    to continue to 30 epochs.
    """
    return {
        "architecture": "resnet18_scratch",
        "embedding_dim": 128,
        "optimizer": "adam",
        "mining": "random",
        "lr": float(args.lr),
        "margin": float(args.margin),
        "classes_per_batch": 16,
        "samples_per_class": 4,
        "seed": int(args.seed),
        "queries_per_class": int(args.queries_per_class),
        "eval_every": int(args.eval_every),
    }


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    if args.threads < 1:
        raise ValueError("threads must be positive")

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    if args.search_batch_size < 1:
        raise ValueError(
            "search-batch-size must be positive"
        )

    if args.log_every < 1:
        raise ValueError("log-every must be positive")

    if args.eval_every < 1:
        raise ValueError("eval-every must be positive")

    if args.queries_per_class < 1:
        raise ValueError(
            "queries-per-class must be positive"
        )

    if args.queries_per_class > 60:
        raise ValueError(
            "validation has only 60 images per class"
        )

    if args.seed < 0:
        raise ValueError("seed must be non-negative")

    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError(
            "learning rate must be finite and positive"
        )

    if not math.isfinite(args.margin) or args.margin <= 0:
        raise ValueError(
            "margin must be finite and positive"
        )


def capture_rng_state():
    numpy_state = np.random.get_state()

    return {
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


def restore_rng_state(state):
    random.setstate(state["python_rng"])

    numpy_state = state["numpy_rng"]

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
        state["torch_rng"]
    )


def atomic_save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    history,
    config,
    best_map,
    best_epoch,
    baseline_metrics,
):
    checkpoint = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "config": config,
        "best_map": float(best_map),
        "best_epoch": int(best_epoch),
        "baseline_metrics": baseline_metrics,
        "rng_state": capture_rng_state(),
    }

    atomic_save_checkpoint(
        path,
        checkpoint,
    )


def load_training_checkpoint(
    path,
    model,
    optimizer,
    expected_config,
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

    if checkpoint.get("config") != expected_config:
        raise ValueError(
            "Checkpoint configuration mismatch"
        )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    restore_rng_state(
        checkpoint["rng_state"]
    )

    return {
        "epoch": checkpoint["epoch"],
        "history": checkpoint["history"],
        "best_map": checkpoint["best_map"],
        "best_epoch": checkpoint["best_epoch"],
        "baseline_metrics": checkpoint[
            "baseline_metrics"
        ],
    }


def save_history(path, history):
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
                fieldnames=METRIC_FIELDS,
            )

            writer.writeheader()
            writer.writerows(history)

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def train_full_epoch(
    model,
    loader,
    optimizer,
    epoch,
    margin=0.3,
    log_every=100,
):
    """
    Process every training batch exactly once.

    The class-balanced epoch sampler supplies
    750 complete batches of 64 examples.
    """
    model.train()
    loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_images = 0

    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(
        loader,
        start=1,
    ):
        optimizer.zero_grad()

        embeddings = model(images)

        loss = random_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss in epoch {epoch}, "
                f"batch {batch_index}"
            )

        loss.backward()
        optimizer.step()

        batch_images = len(labels)

        total_loss += (
            loss.item() * batch_images
        )
        total_images += batch_images

        if (
            batch_index == 1
            or batch_index % log_every == 0
            or batch_index == len(loader)
        ):
            elapsed = time.perf_counter() - start

            print(
                f"Epoch {epoch} | "
                f"batch {batch_index}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    expected_images = len(loader.dataset)

    if total_images != expected_images:
        raise RuntimeError(
            f"Incomplete epoch: processed "
            f"{total_images} of {expected_images} images"
        )

    return {
        "loss": total_loss / total_images,
        "images": total_images,
    }


@torch.no_grad()
def evaluate_model_retrieval(
    model,
    gallery_loader,
    queries_per_class=5,
    search_batch_size=32,
    seed=42,
):
    model.eval()

    embeddings, labels, ids = extract_embeddings(
        model,
        gallery_loader,
        expected_dim=128,
    )

    query_indices = select_stratified_queries(
        labels,
        samples_per_class=queries_per_class,
        seed=seed,
    )

    return evaluate_retrieval(
        query_embeddings=embeddings[query_indices],
        query_labels=labels[query_indices],
        query_ids=ids[query_indices],
        gallery_embeddings=embeddings,
        gallery_labels=labels,
        gallery_ids=ids,
        ks=(1, 5, 10),
        batch_size=search_batch_size,
    )


def make_history_row(
    epoch,
    training,
    validation,
    epoch_seconds,
):
    row = {
        "epoch": epoch,
        "train_loss": training["loss"],
        "train_images": training["images"],
        "val_recall_at_1": None,
        "val_recall_at_5": None,
        "val_recall_at_10": None,
        "val_hit_at_1": None,
        "val_hit_at_5": None,
        "val_hit_at_10": None,
        "val_map": None,
        "epoch_seconds": epoch_seconds,
    }

    if validation is not None:
        for k in (1, 5, 10):
            row[f"val_recall_at_{k}"] = validation[
                f"recall_at_{k}"
            ]
            row[f"val_hit_at_{k}"] = validation[
                f"hit_at_{k}"
            ]

        row["val_map"] = validation["map"]

    return row


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 6 full CIFAR-100 random-mining "
            "training"
        )
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
        "--lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation embedding extraction batch size",
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--queries-per-class",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--run-dir",
        default="checkpoints/stage6_random",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser


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

    run_dir = Path(args.run_dir)
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    if not args.resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            parser.error(
                f"Run directory is not empty: {run_dir}. "
                "Use --resume or a new run directory."
            )
    elif not last_path.is_file():
        parser.error(
            f"Cannot resume; missing: {last_path}"
        )

    print("\nStage 6 — Full training")
    print("-" * 55)
    print(f"CPU threads: {torch.get_num_threads()}")
    print("Training batch: 16 classes x 4 images = 64")
    print("Training batches per epoch: 750")
    print(f"Target complete epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Run directory: {run_dir}")

    print("\nLoading CIFAR-100...", flush=True)

    train_loader, val_loader, _ = load_cifar100(
        seed=args.seed,
        classes_per_batch=16,
        samples_per_class=4,
    )

    gallery_loader = DataLoader(
        val_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if len(train_loader.dataset) != 48000:
        raise RuntimeError(
            "Expected exactly 48,000 training images"
        )

    if len(val_loader.dataset) != 6000:
        raise RuntimeError(
            "Expected exactly 6,000 validation images"
        )

    model = EmbeddingNet(
        embedding_dim=128
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
    )

    start_epoch = 1
    history = []
    best_map = float("-inf")
    best_epoch = 0
    baseline_metrics = None

    if args.resume:
        restored = load_training_checkpoint(
            last_path,
            model,
            optimizer,
            expected_config=config,
        )

        start_epoch = restored["epoch"] + 1
        history = restored["history"]
        best_map = restored["best_map"]
        best_epoch = restored["best_epoch"]
        baseline_metrics = restored["baseline_metrics"]

        # Recover the CSV from the checkpoint if
        # the previous run stopped during CSV saving.
        save_history(
            metrics_path,
            history,
        )

        print(
            f"\nResuming from epoch {start_epoch}",
            flush=True,
        )
        print(
            f"Previous best validation mAP: "
            f"{best_map:.4%} at epoch {best_epoch}",
            flush=True,
        )

    else:
        print(
            "\nEvaluating untrained model...",
            flush=True,
        )

        baseline_metrics = evaluate_model_retrieval(
            model,
            gallery_loader,
            queries_per_class=args.queries_per_class,
            search_batch_size=args.search_batch_size,
            seed=args.seed,
        )

        print(
            f"Untrained validation mAP: "
            f"{baseline_metrics['map']:.4%}"
        )
        print(
            f"Untrained Hit@1: "
            f"{baseline_metrics['hit_at_1']:.4%}"
        )

    if start_epoch > args.epochs:
        print(
            "\nRequested epochs already completed."
        )
        return

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.perf_counter()

        print(
            f"\n{'=' * 55}\n"
            f"FULL EPOCH {epoch}/{args.epochs}\n"
            f"{'=' * 55}",
            flush=True,
        )

        training = train_full_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            margin=args.margin,
            log_every=args.log_every,
        )

        should_evaluate = (
            epoch % args.eval_every == 0
        )

        validation = None
        improved = False

        if should_evaluate:
            print(
                "\nEvaluating validation retrieval...",
                flush=True,
            )

            validation = evaluate_model_retrieval(
                model,
                gallery_loader,
                queries_per_class=args.queries_per_class,
                search_batch_size=args.search_batch_size,
                seed=args.seed,
            )

            if validation["map"] > best_map:
                best_map = validation["map"]
                best_epoch = epoch
                improved = True

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        row = make_history_row(
            epoch,
            training,
            validation,
            epoch_seconds,
        )

        history.append(row)

        # Save best before last, so the last
        # checkpoint never points to a best model
        # that has not yet been written.
        if improved:
            save_training_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                history,
                config,
                best_map,
                best_epoch,
                baseline_metrics,
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
            baseline_metrics,
        )

        save_history(
            metrics_path,
            history,
        )

        print(
            f"\nEpoch {epoch} summary"
        )
        print("-" * 55)
        print(
            f"Training loss: {training['loss']:.6f}"
        )
        print(
            f"Images processed: {training['images']}"
        )

        if validation is not None:
            print(
                f"Validation Hit@1: "
                f"{validation['hit_at_1']:.4%}"
            )
            print(
                f"Validation Hit@5: "
                f"{validation['hit_at_5']:.4%}"
            )
            print(
                f"Validation mAP: "
                f"{validation['map']:.4%}"
            )

        print(
            f"Best validation mAP: "
            f"{best_map:.4%} "
            f"(epoch {best_epoch})"
        )
        print(
            f"Epoch duration: "
            f"{epoch_seconds / 60:.2f} minutes"
        )

        print(
            f"Saved: {last_path}",
            flush=True,
        )

        if improved:
            print(
                f"New best model: {best_path}",
                flush=True,
            )

    print("\nTraining target completed.")
    print(f"Last checkpoint: {last_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Metric history: {metrics_path}")


if __name__ == "__main__":
    main()
