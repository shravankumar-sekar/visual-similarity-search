
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (
    stratified_split,
    CIFAR100Subset,
    ClassBalancedBatchSampler,
)


def main():
    train = CIFAR100(root="./data", train=True, download=False)
    test = CIFAR100(root="./data", train=False, download=False)

    images = np.concatenate([train.data, test.data])
    labels = np.asarray(train.targets + test.targets)

    train_idx, val_idx, test_idx = stratified_split(labels)

    train_set = CIFAR100Subset(
        images, labels, train_idx, training=True
    )

    sampler = ClassBalancedBatchSampler(
        labels,
        train_idx,
        classes_per_batch=8,
        samples_per_class=4,
    )

    loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=0,
    )

    batch_images, batch_labels = next(iter(loader))

    unique_labels, counts = np.unique(
        batch_labels.numpy(),
        return_counts=True,
    )

    print("Train samples:", len(train_set))
    print("Validation samples:", len(val_idx))
    print("Test samples:", len(test_idx))
    print("Image batch:", tuple(batch_images.shape))
    print("Classes in batch:", len(unique_labels))
    print("Images per class:", counts.tolist())

    assert len(train_set) == 48000
    assert len(val_idx) == 6000
    assert len(test_idx) == 6000
    assert tuple(batch_images.shape) == (32, 3, 32, 32)
    assert len(unique_labels) == 8
    assert np.all(counts == 4)

    print("DataLoader integration PASSED")


if __name__ == "__main__":
    main()
