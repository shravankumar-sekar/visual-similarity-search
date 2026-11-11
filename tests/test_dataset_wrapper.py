
import numpy as np
import torch

from data import CIFAR100Subset


def test_dataset_length():
    images = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(10)

    dataset = CIFAR100Subset(
        images,
        labels,
        indices=[1, 3, 5],
        training=False,
    )

    assert len(dataset) == 3


def test_image_shape_and_label():
    images = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(10)

    dataset = CIFAR100Subset(
        images,
        labels,
        indices=[1, 3, 5],
        training=False,
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 1
