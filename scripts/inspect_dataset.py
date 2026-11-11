
from collections import Counter

import numpy as np
from torchvision.datasets import CIFAR100


def main():
   

    print("Loading CIFAR-100...")

    train_dataset = CIFAR100(
        root="./data",
        train=True,
        download=True,
    )

    test_dataset = CIFAR100(
        root="./data",
        train=False,
        download=True,
    )

    labels = train_dataset.targets + test_dataset.targets
    class_counts = Counter(labels)

    print("\nDATASET INFORMATION")
    print("--------------------")
    print("Training images:", len(train_dataset))
    print("Original test images:", len(test_dataset))
    print("Total images:", len(train_dataset) + len(test_dataset))
    print("Number of classes:", len(class_counts))

    image, label = train_dataset[0]

    print("\nIMAGE INFORMATION")
    print("--------------------")
    print("Image size:", image.size)
    print("Image mode:", image.mode)
    print("Image array shape:", np.asarray(image).shape)
    print("First image label:", label)
    print("First image class:", train_dataset.classes[label])

    print("\nCLASS DISTRIBUTION")
    print("--------------------")
    print("Minimum images per class:", min(class_counts.values()))
    print("Maximum images per class:", max(class_counts.values()))

    assert len(train_dataset) == 50000
    assert len(test_dataset) == 10000
    assert len(class_counts) == 100
    assert all(count == 600 for count in class_counts.values())

    print("\nAll dataset checks PASSED")


if __name__ == "__main__":
    main()
