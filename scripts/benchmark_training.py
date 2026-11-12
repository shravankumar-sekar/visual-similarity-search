
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss import batch_hard_triplet_loss
from model import EmbeddingNet
from optimizers import Adam
from train import load_cifar100, set_seed


def benchmark_steps(
    model,
    loader,
    optimizer,
    steps=5,
    warmup=2,
    margin=0.3,
):
    if steps < 1:
        raise ValueError("Steps must be positive")

    if warmup < 0:
        raise ValueError("Warmup cannot be negative")

    if len(loader) == 0:
        raise ValueError("DataLoader cannot be empty")

    model.train()
    iterator = iter(loader)

    timings = {
        "data": [],
        "forward": [],
        "loss": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }

    losses = []
    batch_sizes = []

    for step in range(warmup + steps):
        total_start = time.perf_counter()

        data_start = time.perf_counter()

        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)

        data_end = time.perf_counter()

        optimizer.zero_grad()

        forward_start = time.perf_counter()
        embeddings = model(images)
        forward_end = time.perf_counter()

        loss_start = time.perf_counter()
        loss = batch_hard_triplet_loss(
            embeddings,
            labels,
            margin=margin,
        )
        loss_end = time.perf_counter()

        backward_start = time.perf_counter()
        loss.backward()
        backward_end = time.perf_counter()

        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer_end = time.perf_counter()

        if step < warmup:
            continue

        timings["data"].append(data_end - data_start)
        timings["forward"].append(
            forward_end - forward_start
        )
        timings["loss"].append(loss_end - loss_start)
        timings["backward"].append(
            backward_end - backward_start
        )
        timings["optimizer"].append(
            optimizer_end - optimizer_start
        )
        timings["total"].append(
            optimizer_end - total_start
        )

        losses.append(loss.item())
        batch_sizes.append(len(labels))

        print(
            f"Measured batch {step - warmup + 1}/{steps} | "
            f"time={timings['total'][-1]:.3f}s | "
            f"loss={loss.item():.4f}",
            flush=True,
        )

    averages = {
        key: statistics.mean(values)
        for key, values in timings.items()
    }

    mean_batch_size = statistics.mean(batch_sizes)
    images_per_second = (
        mean_batch_size / averages["total"]
    )

    return {
        "steps": steps,
        "batch_size": mean_batch_size,
        "mean_loss": statistics.mean(losses),
        "timings": averages,
        "images_per_second": images_per_second,
    }


def print_report(result, batches_per_epoch):
    timings = result["timings"]
    step_time = timings["total"]

    epoch_seconds = step_time * batches_per_epoch
    thirty_epoch_hours = epoch_seconds * 30 / 3600

    print("\n" + "=" * 46)
    print("CPU TRAINING BENCHMARK")
    print("=" * 46)

    print(f"Measured steps:     {result['steps']}")
    print(f"Batch size:         {result['batch_size']:.0f}")
    print(f"Mean loss:          {result['mean_loss']:.4f}")

    print("\nAverage time per batch")
    print("-" * 46)

    for name in (
        "data",
        "forward",
        "loss",
        "backward",
        "optimizer",
        "total",
    ):
        print(f"{name:15s}: {timings[name]:.4f} sec")

    print("\nThroughput")
    print("-" * 46)

    print(
        f"Images/second:      "
        f"{result['images_per_second']:.2f}"
    )
    print(f"Batches per epoch:  {batches_per_epoch}")
    print(
        f"Estimated epoch:    "
        f"{epoch_seconds / 60:.1f} minutes"
    )
    print(
        f"Estimated 30 epochs: "
        f"{thirty_epoch_hours:.1f} hours"
    )

    print("\nEstimates exclude validation and checkpointing.")
    print("Short benchmarks may not reflect sustained speed.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU triplet training"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--classes-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU threads; 0 keeps the default",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.threads < 0:
        parser.error("--threads cannot be negative")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    set_seed(42)

    print("Preparing CPU training benchmark...", flush=True)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    train_loader, _, _ = load_cifar100(
        seed=42,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
    )

    model = EmbeddingNet(
        embedding_dim=args.embedding_dim
    ).cpu()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    print(f"Training batches per epoch: {len(train_loader)}")
    print(
        f"Batch size: "
        f"{args.classes_per_batch * args.samples_per_class}"
    )
    print(
        f"Warmup steps: {args.warmup} | "
        f"Measured steps: {args.steps}",
        flush=True,
    )

    result = benchmark_steps(
        model,
        train_loader,
        optimizer,
        steps=args.steps,
        warmup=args.warmup,
    )

    print_report(
        result,
        batches_per_epoch=len(train_loader),
    )


if __name__ == "__main__":
    main()
