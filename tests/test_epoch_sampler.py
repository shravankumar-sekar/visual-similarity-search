
import numpy as np
import pytest

from epoch_sampler import EpochBalancedBatchSampler


def make_sampler(seed=42):
    labels = np.repeat(np.arange(100), 480)
    indices = np.arange(48000)

    return EpochBalancedBatchSampler(
        labels,
        indices,
        classes_per_batch=8,
        samples_per_class=4,
        seed=seed,
    )


def test_every_image_visited_once():
    sampler = make_sampler()
    batches = list(sampler)

    visited = [
        position
        for batch in batches
        for position in batch
    ]

    assert len(batches) == 1500
    assert len(visited) == 48000
    assert len(set(visited)) == 48000
    assert set(visited) == set(range(48000))


def test_batch_class_balance():
    sampler = make_sampler()
    labels = np.repeat(np.arange(100), 480)

    for batch in sampler:
        batch_labels = labels[batch]
        unique, counts = np.unique(
            batch_labels,
            return_counts=True,
        )

        assert len(batch) == 32
        assert len(unique) == 8
        assert all(count == 4 for count in counts)


def test_sampler_reproducibility():
    sampler_a = make_sampler(seed=42)
    sampler_b = make_sampler(seed=42)

    assert list(sampler_a) == list(sampler_b)


def test_different_epoch_order():
    sampler = make_sampler()

    first_epoch = list(sampler)

    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert first_epoch != second_epoch

    first_positions = {
        position
        for batch in first_epoch
        for position in batch
    }

    second_positions = {
        position
        for batch in second_epoch
        for position in batch
    }

    assert first_positions == second_positions
    assert len(first_positions) == 48000


def test_validation_partial_batch():
    labels = np.repeat(np.arange(100), 60)
    indices = np.arange(6000)

    sampler = EpochBalancedBatchSampler(
        labels,
        indices,
        classes_per_batch=8,
        samples_per_class=4,
        seed=42,
    )

    batches = list(sampler)

    visited = [
        position
        for batch in batches
        for position in batch
    ]

    assert len(batches) == 188
    assert len(visited) == 6000
    assert len(set(visited)) == 6000
    assert len(batches[-1]) == 16

    for batch in batches:
        batch_labels = labels[batch]
        _, counts = np.unique(
            batch_labels,
            return_counts=True,
        )

        assert len(counts) >= 2
        assert all(count == 4 for count in counts)


def test_subset_positions():
    labels = np.repeat(np.arange(4), 8)
    indices = np.array([
        8, 9, 10, 11,
        16, 17, 18, 19,
        24, 25, 26, 27,
        0, 1, 2, 3,
    ])

    sampler = EpochBalancedBatchSampler(
        labels,
        indices,
        classes_per_batch=2,
        samples_per_class=2,
        seed=42,
    )

    visited = [
        position
        for batch in sampler
        for position in batch
    ]

    assert sorted(visited) == list(range(len(indices)))


def test_invalid_batch_configuration():
    labels = np.repeat(np.arange(4), 8)
    indices = np.arange(32)

    with pytest.raises(ValueError):
        EpochBalancedBatchSampler(
            labels,
            indices,
            classes_per_batch=1,
        )

    with pytest.raises(ValueError):
        EpochBalancedBatchSampler(
            labels,
            indices,
            classes_per_batch=5,
        )

    with pytest.raises(ValueError):
        EpochBalancedBatchSampler(
            labels,
            indices,
            samples_per_class=3,
        )


def test_resume_epoch_reproducibility():
    sampler_a = make_sampler(seed=42)
    sampler_b = make_sampler(seed=42)

    sampler_a.set_epoch(2)
    sampler_b.set_epoch(2)

    assert list(sampler_a) == list(sampler_b)
