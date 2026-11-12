
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import EmbeddingNet
from optimizers import Adam
from random_mining import random_triplet_loss
from train import load_cifar100, set_seed
from scripts.diagnose_embeddings import diagnose


def train_epoch(model, loader, optimizer, steps, epoch):
    model.train()
    loader.batch_sampler.set_epoch(epoch)

    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= steps:
            break

        optimizer.zero_grad()
        embeddings = model(images)

        loss = random_triplet_loss(
            embeddings, labels, margin=0.3
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch {epoch} | "
                f"batch {batch_idx + 1}/{steps} | "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--output",
        default="results/random_mining_test.csv",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.steps < 1 or args.threads < 1:
        parser.error("Epochs, steps and threads must be positive")

    if args.lr <= 0:
        parser.error("Learning rate must be positive")

    torch.set_num_threads(args.threads)
    set_seed(42)

    train_loader, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    if args.steps > len(train_loader):
        parser.error("Steps cannot exceed batches per epoch")

    model = EmbeddingNet(embedding_dim=128)
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = []

    print("Initial validation diagnostics", flush=True)
    baseline = diagnose(model, val_loader, batches=10)

    print(baseline, flush=True)

    for epoch in range(1, args.epochs + 1):
        average_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            steps=args.steps,
            epoch=epoch,
        )

        result = diagnose(
            model, val_loader, batches=10
        )

        row = {
            "epoch": epoch,
            "train_loss": average_loss,
            **result,
        }
        history.append(row)

        print(f"\nEpoch {epoch} results:")
        for name, value in row.items():
            print(f"{name}: {value:.6f}" if name != "epoch"
                  else f"{name}: {value}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0]),
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"\nResults saved: {output}")


if __name__ == "__main__":
    main()
