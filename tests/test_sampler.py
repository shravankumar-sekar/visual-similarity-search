
import numpy as np
import pytest

from data import ClassBalancedBatchSampler


@pytest.fixture
def sample_data():
    labels = np.repeat(np.arange(10), 20)
    indices = np.arange(len(labels))
    return labels, indices


def test_batch_structure(sample_data):
    labels, indices = sample_data

    sampler = ClassBalancedBatchSampler(
        labels, indices, classes_per_batch=4,
        samples_per_class=3,
    )

    batch = next(iter(sampler))
    batch_labels = labels[indices[batch]]

    classes, counts = np.unique(
        batch_labels, return_counts=True
    )

    assert len(batch) == 12
    assert len(classes) == 4
    assert np.all(counts == 3)
    assert len(set(batch)) == 12


def test_reproducibility(sample_data):
    labels, indices = sample_data

    sampler_a = ClassBalancedBatchSampler(
        labels, indices, seed=42
    )

    sampler_b = ClassBalancedBatchSampler(
        labels, indices, seed=42
    )

    assert list(sampler_a) == list(sampler_b)


def test_invalid_configuration(sample_data):
    labels, indices = sample_data

    with pytest.raises(ValueError):
        ClassBalancedBatchSampler(
            labels, indices, classes_per_batch=1
        )


def test_subset_positions():
    labels = np.repeat(np.arange(5), 10)
    indices = np.arange(0, 50, 2)

    sampler = ClassBalancedBatchSampler(
        labels, indices,
        classes_per_batch=2,
        samples_per_class=2,
    )

    batch = next(iter(sampler))

    assert all(0 <= idx < len(indices) for idx in batch)
