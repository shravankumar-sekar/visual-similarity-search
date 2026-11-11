
import sys
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100

# Allow importing data.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import stratified_split


def main():
    train_dataset = CIFAR100(
        root="./data",
        train=True,
        download=False,
    )

    test_dataset = CIFAR100(
        root="./data",
        train=False,
        download=False,
    )

    labels = np.asarray(
        train_dataset.targets + test_dataset.targets
    )

    train_idx, val_idx, test_idx = stratified_split(
        labels,
        seed=42,
    )

    print("Training:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    assert len(train_idx) == 48000
    assert len(val_idx) == 6000
    assert len(test_idx) == 6000

    assert len(np.unique(np.concatenate(
        [train_idx, val_idx, test_idx]
    ))) == 60000

    print("Real CIFAR-100 split verification PASSED")


if __name__ == "__main__":
    main()
