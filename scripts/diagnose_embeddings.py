
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss import pairwise_distances
from model import EmbeddingNet
from train import load_cifar100, set_seed, triplet_metrics


@torch.no_grad()
def diagnose(model, loader, batches=10, seed=42):
    model.eval()
    generator = torch.Generator().manual_seed(seed)

    random_correct = 0
    hard_correct = 0
    total = 0

    positive_distances = []
    negative_distances = []
    spreads = []

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= batches:
            break

        embeddings = model(images)
        distances = pairwise_distances(embeddings)
        metrics = triplet_metrics(embeddings, labels)

        spreads.append(metrics["embedding_spread"])

        same = labels[:, None] == labels[None, :]
        positive_mask = same.clone()
        positive_mask.fill_diagonal_(False)
        negative_mask = ~same

        hardest_positive = distances.masked_fill(
            ~positive_mask, float("-inf")
        ).max(dim=1).values

        hardest_negative = distances.masked_fill(
            ~negative_mask, float("inf")
        ).min(dim=1).values

        hard_correct += (
            hardest_positive < hardest_negative
        ).sum().item()

        for i in range(len(labels)):
            positive_indices = torch.where(
                positive_mask[i]
            )[0]

            negative_indices = torch.where(
                negative_mask[i]
            )[0]

            positive_index = positive_indices[
                torch.randint(
                    len(positive_indices),
                    (1,),
                    generator=generator,
                ).item()
            ]

            negative_index = negative_indices[
                torch.randint(
                    len(negative_indices),
                    (1,),
                    generator=generator,
                ).item()
            ]

            dp = distances[i, positive_index].item()
            dn = distances[i, negative_index].item()

            positive_distances.append(dp)
            negative_distances.append(dn)

            if dp < dn:
                random_correct += 1

            total += 1

    if total == 0:
        raise ValueError("No validation samples processed")

    return {
        "random_ordering": random_correct / total,
        "hard_ordering": hard_correct / total,
        "positive_distance": sum(positive_distances) / total,
        "negative_distance": sum(negative_distances) / total,
        "embedding_spread": sum(spreads) / len(spreads),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if args.batches < 1 or args.threads < 1:
        parser.error("Batches and threads must be positive")

    torch.set_num_threads(args.threads)
    set_seed(42)

    model = EmbeddingNet(embedding_dim=128)

    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state"])

        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("Model: randomly initialized")

    _, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )

    result = diagnose(
        model,
        val_loader,
        batches=args.batches,
    )

    print("\nEmbedding diagnostics")
    print("-" * 40)

    for name, value in result.items():
        print(f"{name:22s}: {value:.6f}")


if __name__ == "__main__":
    main()
