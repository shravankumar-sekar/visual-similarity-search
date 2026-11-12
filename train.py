
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from data import (
    stratified_split,
    CIFAR100Subset,
    ClassBalancedBatchSampler,
)
from model import EmbeddingNet
from loss import batch_hard_triplet_loss, pairwise_distances
from optimizers import MomentumSGD, Adam, NAdam


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cifar100(
    root="./data",
    seed=42,
    classes_per_batch=8,
    samples_per_class=4,
):
    train = CIFAR100(root=root, train=True, download=True)
    test = CIFAR100(root=root, train=False, download=True)

    images = np.concatenate([train.data, test.data], axis=0)
    labels = np.asarray(train.targets + test.targets)

    train_idx, val_idx, test_idx = stratified_split(
        labels, seed=seed
    )

    train_set = CIFAR100Subset(
        images, labels, train_idx, training=True
    )
    val_set = CIFAR100Subset(
        images, labels, val_idx, training=False
    )
    test_set = CIFAR100Subset(
        images, labels, test_idx, training=False
    )

    def make_loader(dataset, indices, sampler_seed):
        sampler = ClassBalancedBatchSampler(
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
            parameters, lr=lr, momentum=0.9
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
        ~positive_mask, float("-inf")
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        ~negative_mask, float("inf")
    ).min(dim=1).values

    positive_dist = hardest_positive[valid]
    negative_dist = hardest_negative[valid]

    ordering_accuracy = (
        positive_dist < negative_dist
    ).float().mean().item()

    margin_accuracy = (
        positive_dist + margin < negative_dist
    ).float().mean().item()

    return {
        "ordering_accuracy": ordering_accuracy,
        "margin_accuracy": margin_accuracy,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    margin=0.3,
    max_batches=5,
):
    model.train()

    total_loss = 0.0
    total_ordering = 0.0
    total_margin = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        optimizer.zero_grad()

        embeddings = model(images)

        loss = batch_hard_triplet_loss(
            embeddings, labels, margin=margin
        )

        loss.backward()
        optimizer.step()

        metrics = triplet_metrics(
            embeddings.detach(),
            labels,
            margin=margin,
        )

        total_loss += loss.item()
        total_ordering += metrics["ordering_accuracy"]
        total_margin += metrics["margin_accuracy"]
        num_batches += 1

        print(
            f"Training batch {batch_idx + 1}/{max_batches} | "
            f"loss={loss.item():.4f} | "
            f"ordering={metrics['ordering_accuracy']:.4f} | "
            f"margin={metrics['margin_accuracy']:.4f}",
            flush=True,
        )

    return {
        "loss": total_loss / num_batches,
        "ordering_accuracy": total_ordering / num_batches,
        "margin_accuracy": total_margin / num_batches,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    margin=0.3,
    max_batches=5,
    name="Validation",
):
    model.eval()

    total_loss = 0.0
    total_ordering = 0.0
    total_margin = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        embeddings = model(images)

        loss = batch_hard_triplet_loss(
            embeddings, labels, margin=margin
        )

        metrics = triplet_metrics(
            embeddings, labels, margin=margin
        )

        total_loss += loss.item()
        total_ordering += metrics["ordering_accuracy"]
        total_margin += metrics["margin_accuracy"]
        num_batches += 1

        print(
            f"{name} batch {batch_idx + 1}/{max_batches} | "
            f"loss={loss.item():.4f} | "
            f"ordering={metrics['ordering_accuracy']:.4f} | "
            f"margin={metrics['margin_accuracy']:.4f}",
            flush=True,
        )

    return {
        "loss": total_loss / num_batches,
        "ordering_accuracy": total_ordering / num_batches,
        "margin_accuracy": total_margin / num_batches,
    }


def save_checkpoint(path, model, optimizer, epoch, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def main():
    set_seed(42)

    # Temporary CPU smoke-test settings.
    config = {
        "embedding_dim": 128,
        "optimizer": "adam",
        "lr": 0.001,
        "margin": 0.3,
        "epochs": 1,
        "classes_per_batch": 8,
        "samples_per_class": 4,
        "max_batches": 5,
    }

    print("Starting CIFAR-100 smoke test...", flush=True)

    train_loader, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=config["classes_per_batch"],
        samples_per_class=config["samples_per_class"],
    )

    print("Dataset loaded successfully.", flush=True)
    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Validation batches: {len(val_loader)}", flush=True)

    model = EmbeddingNet(
        embedding_dim=config["embedding_dim"]
    )

    optimizer = make_optimizer(
        config["optimizer"],
        model.parameters(),
        lr=config["lr"],
    )

    print("Model and optimizer initialized.", flush=True)

    history = []

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nStarting epoch {epoch}", flush=True)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            margin=config["margin"],
            max_batches=config["max_batches"],
        )

        val_metrics = evaluate(
            model,
            val_loader,
            margin=config["margin"],
            max_batches=config["max_batches"],
            name="Validation",
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

        print(
            f"\nEpoch {epoch:02d} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_ordering={row['train_ordering']:.4f} | "
            f"train_margin={row['train_margin']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_ordering={row['val_ordering']:.4f} | "
            f"val_margin={row['val_margin']:.4f}",
            flush=True,
        )

        save_checkpoint(
            "checkpoints/smoke_test.pt",
            model,
            optimizer,
            epoch,
            history,
        )

        print("Checkpoint saved.", flush=True)

    print("\nSmoke test completed successfully.", flush=True)
    print(
        "Test split remains reserved for final evaluation.",
        flush=True,
    )


if __name__ == "__main__":
    main()
