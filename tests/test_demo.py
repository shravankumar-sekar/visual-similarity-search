
"""Tests for the Streamlit visual-search demo."""

import numpy as np
import pytest
import torch

from demo import (
    CIFAR100_CLASSES,
    class_name,
    search_top_k,
    tensor_to_image,
)


def test_class_names():
    assert len(CIFAR100_CLASSES) == 100
    assert class_name(0) == "apple"
    assert class_name(99) == "worm"


def test_invalid_class_label():
    with pytest.raises(ValueError):
        class_name(100)


def test_nearest_neighbor_order():
    gallery = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    indices, distances = search_top_k(
        query, gallery, k=3
    )

    assert indices.tolist() == [0, 1, 2]
    assert distances[0] == pytest.approx(0.0)
    assert np.all(np.diff(distances) >= 0)


def test_exact_top_five():
    gallery = np.eye(10, dtype=np.float32)
    query = gallery[3]

    indices, distances = search_top_k(
        query, gallery, k=5
    )

    assert len(indices) == 5
    assert len(distances) == 5
    assert indices[0] == 3
    assert distances[0] == pytest.approx(0.0)


def test_deterministic_ties():
    gallery = np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float32)

    query = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    indices, _ = search_top_k(
        query, gallery, k=2
    )

    assert indices.tolist() == [0, 1]


def test_dimension_mismatch():
    gallery = np.zeros(
        (10, 128),
        dtype=np.float32,
    )

    query = np.zeros(
        64,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        search_top_k(query, gallery, k=5)


def test_invalid_top_k():
    gallery = np.zeros(
        (10, 128),
        dtype=np.float32,
    )

    query = np.zeros(
        128,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        search_top_k(query, gallery, k=11)


def test_nonfinite_embedding():
    gallery = np.zeros(
        (10, 128),
        dtype=np.float32,
    )

    query = np.zeros(
        128,
        dtype=np.float32,
    )

    query[0] = np.nan

    with pytest.raises(ValueError):
        search_top_k(query, gallery, k=5)


def test_tensor_to_image():
    tensor = torch.full(
        (3, 32, 32),
        0.5,
        dtype=torch.float32,
    )

    image = tensor_to_image(tensor)

    assert image.size == (32, 32)
    assert image.mode == "RGB"
