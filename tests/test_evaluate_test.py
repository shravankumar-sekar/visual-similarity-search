
import numpy as np
import pytest

from scripts.evaluate_test import (
    compute_triplet_ordering,
    save_results,
)


def make_separated_examples():
    embeddings = np.array([
        [0.00, 0.00],
        [0.10, 0.00],
        [0.20, 0.00],
        [10.00, 0.00],
        [10.10, 0.00],
        [10.20, 0.00],
    ], dtype=np.float32)

    labels = np.array([
        0, 0, 0,
        1, 1, 1,
    ])

    return embeddings, labels


def test_perfect_triplet_ordering():
    embeddings, labels = make_separated_examples()

    result = compute_triplet_ordering(
        embeddings,
        labels,
        query_positions=np.array([0, 3]),
        trials_per_query=10,
        margin=0.3,
    )

    assert result["random_triplet_ordering"] == 1.0
    assert result["random_triplet_margin"] == 1.0
    assert result["hard_triplet_ordering"] == 1.0
    assert result["triplets_evaluated"] == 20


def test_overlapping_classes_fail_hard_ordering():
    embeddings = np.array([
        [0.0],
        [2.0],
        [1.0],
        [3.0],
    ], dtype=np.float32)

    labels = np.array([0, 0, 1, 1])

    result = compute_triplet_ordering(
        embeddings,
        labels,
        query_positions=np.arange(4),
        trials_per_query=5,
    )

    assert result["hard_triplet_ordering"] == 0.0


def test_identical_embeddings_have_zero_ordering():
    embeddings = np.ones(
        (6, 4),
        dtype=np.float32,
    )

    labels = np.array([
        0, 0, 0,
        1, 1, 1,
    ])

    result = compute_triplet_ordering(
        embeddings,
        labels,
        query_positions=np.arange(6),
    )

    assert result["random_triplet_ordering"] == 0.0
    assert result["random_triplet_margin"] == 0.0
    assert result["hard_triplet_ordering"] == 0.0


def test_triplet_sampling_reproducible():
    embeddings, labels = make_separated_examples()

    first = compute_triplet_ordering(
        embeddings,
        labels,
        query_positions=np.arange(6),
        seed=42,
    )

    second = compute_triplet_ordering(
        embeddings,
        labels,
        query_positions=np.arange(6),
        seed=42,
    )

    assert first == second


def test_reject_missing_positive():
    embeddings = np.array([
        [0.0],
        [1.0],
        [2.0],
    ], dtype=np.float32)

    labels = np.array([0, 1, 1])

    with pytest.raises(ValueError):
        compute_triplet_ordering(
            embeddings,
            labels,
            query_positions=np.array([0]),
        )


def test_reject_invalid_query_position():
    embeddings, labels = make_separated_examples()

    with pytest.raises(ValueError):
        compute_triplet_ordering(
            embeddings,
            labels,
            query_positions=np.array([99]),
        )


def test_reject_nonfinite_embeddings():
    embeddings, labels = make_separated_examples()

    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError):
        compute_triplet_ordering(
            embeddings,
            labels,
            query_positions=np.array([0]),
        )


def test_results_are_saved(tmp_path):
    json_path = tmp_path / "test.json"
    csv_path = tmp_path / "test.csv"

    results = {
        "map": 0.15,
        "random_triplet_ordering": 0.8,
    }

    save_results(
        json_path,
        csv_path,
        results,
    )

    assert json_path.is_file()
    assert csv_path.is_file()

    with pytest.raises(FileExistsError):
        save_results(
            json_path,
            csv_path,
            results,
        )
