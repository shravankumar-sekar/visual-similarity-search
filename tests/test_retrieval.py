
import numpy as np
import pytest

from retrieval import (
    brute_force_search,
    evaluate_retrieval,
)

from scripts.evaluate_retrieval import (
    select_stratified_queries,
)


def make_perfect_gallery():
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [10.0, 0.0],
        [10.1, 0.0],
        [10.2, 0.0],
    ], dtype=np.float32)

    labels = np.array([
        0, 0, 0,
        1, 1, 1,
    ])

    ids = np.arange(6)

    return embeddings, labels, ids


def test_nearest_neighbor_ranking():
    gallery = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
    ], dtype=np.float32)

    queries = np.array([
        [0.8, 0.0],
    ], dtype=np.float32)

    indices, distances = brute_force_search(
        queries,
        gallery,
        top_k=3,
    )

    assert indices.tolist() == [[1, 0, 2]]

    np.testing.assert_allclose(
        distances[0],
        [0.2, 0.8, 2.2],
        atol=1e-5,
    )


def test_self_match_is_excluded():
    gallery, _, ids = make_perfect_gallery()

    indices, distances = brute_force_search(
        gallery[[0]],
        gallery,
        top_k=5,
        query_ids=ids[[0]],
        gallery_ids=ids,
    )

    assert 0 not in indices[0]
    assert indices[0, 0] == 1
    assert np.isfinite(distances).all()


def test_perfect_retrieval_metrics():
    gallery, labels, ids = make_perfect_gallery()

    result = evaluate_retrieval(
        query_embeddings=gallery[[0, 3]],
        query_labels=labels[[0, 3]],
        query_ids=ids[[0, 3]],
        gallery_embeddings=gallery,
        gallery_labels=labels,
        gallery_ids=ids,
    )

    assert result["queries"] == 2
    assert result["gallery_size"] == 6

    # Two relevant items exist for each query.
    # The first retrieval finds one of those two.
    assert result["recall_at_1"] == pytest.approx(0.5)
    assert result["recall_at_5"] == pytest.approx(1.0)
    assert result["recall_at_10"] == pytest.approx(1.0)

    assert result["hit_at_1"] == pytest.approx(1.0)
    assert result["hit_at_5"] == pytest.approx(1.0)
    assert result["hit_at_10"] == pytest.approx(1.0)

    assert result["map"] == pytest.approx(1.0)


def test_mean_average_precision_manual_example():
    gallery = np.array([
        [0.0],
        [3.0],
        [1.0],
        [2.0],
    ], dtype=np.float32)

    labels = np.array([0, 0, 1, 1])
    ids = np.arange(4)

    # Query ID 0:
    #   nearest -> class 1 (incorrect)
    #   next    -> class 1 (incorrect)
    #   third   -> class 0 (correct)
    #
    # Only one relevant item remains after
    # excluding the query, so AP = 1/3.
    result = evaluate_retrieval(
        query_embeddings=gallery[[0]],
        query_labels=labels[[0]],
        query_ids=ids[[0]],
        gallery_embeddings=gallery,
        gallery_labels=labels,
        gallery_ids=ids,
    )

    assert result["map"] == pytest.approx(1.0 / 3.0)
    assert result["recall_at_1"] == pytest.approx(0.0)
    assert result["recall_at_5"] == pytest.approx(1.0)
    assert result["hit_at_1"] == pytest.approx(0.0)
    assert result["hit_at_5"] == pytest.approx(1.0)


def test_search_batch_size_does_not_change_results():
    gallery, _, ids = make_perfect_gallery()

    small_indices, small_distances = brute_force_search(
        gallery[[0, 3]],
        gallery,
        top_k=3,
        query_ids=ids[[0, 3]],
        gallery_ids=ids,
        batch_size=1,
    )

    large_indices, large_distances = brute_force_search(
        gallery[[0, 3]],
        gallery,
        top_k=3,
        query_ids=ids[[0, 3]],
        gallery_ids=ids,
        batch_size=8,
    )

    np.testing.assert_array_equal(
        small_indices,
        large_indices,
    )

    np.testing.assert_allclose(
        small_distances,
        large_distances,
    )


def test_rejects_invalid_top_k():
    gallery, _, ids = make_perfect_gallery()

    with pytest.raises(ValueError):
        brute_force_search(
            gallery[[0]],
            gallery,
            top_k=6,
            query_ids=ids[[0]],
            gallery_ids=ids,
        )

    with pytest.raises(ValueError):
        brute_force_search(
            gallery[[0]],
            gallery,
            top_k=0,
        )


def test_rejects_incompatible_dimensions():
    with pytest.raises(ValueError):
        brute_force_search(
            np.zeros((2, 3)),
            np.zeros((4, 2)),
        )


def test_rejects_missing_query_identity():
    gallery, _, ids = make_perfect_gallery()

    with pytest.raises(ValueError):
        brute_force_search(
            gallery[[0]],
            gallery,
            top_k=1,
            query_ids=np.array([999]),
            gallery_ids=ids,
        )


def test_rejects_duplicate_gallery_ids():
    gallery, _, ids = make_perfect_gallery()

    invalid_ids = ids.copy()
    invalid_ids[1] = invalid_ids[0]

    with pytest.raises(ValueError):
        brute_force_search(
            gallery[[0]],
            gallery,
            top_k=1,
            query_ids=ids[[0]],
            gallery_ids=invalid_ids,
        )


def test_rejects_nonfinite_embeddings():
    gallery, _, _ = make_perfect_gallery()
    invalid = gallery.copy()
    invalid[0, 0] = np.nan

    with pytest.raises(ValueError):
        brute_force_search(
            invalid,
            gallery,
            top_k=1,
        )


def test_rejects_queries_without_relevant_images():
    gallery = np.array([
        [0.0],
        [1.0],
        [2.0],
    ], dtype=np.float32)

    labels = np.array([0, 1, 2])
    ids = np.arange(3)

    with pytest.raises(ValueError):
        evaluate_retrieval(
            gallery[[0]],
            labels[[0]],
            ids[[0]],
            gallery,
            labels,
            ids,
        )


def test_stratified_query_selection():
    labels = np.array(
        [0] * 10 + [1] * 10 + [2] * 10
    )

    first = select_stratified_queries(
        labels,
        samples_per_class=3,
        seed=42,
    )

    second = select_stratified_queries(
        labels,
        samples_per_class=3,
        seed=42,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    assert len(first) == 9
    assert len(np.unique(first)) == 9

    selected_labels = labels[first]

    for class_id in (0, 1, 2):
        assert np.sum(
            selected_labels == class_id
        ) == 3
