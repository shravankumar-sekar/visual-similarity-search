
import numpy as np


def _validate_embeddings(embeddings, name):
    array = np.asarray(embeddings, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")

    return array


def _validate_ids(ids, count, name):
    array = np.asarray(ids)

    if array.ndim != 1 or len(array) != count:
        raise ValueError(f"{name} has incorrect shape")

    if len(np.unique(array)) != count:
        raise ValueError(f"{name} must contain unique IDs")

    return array


def brute_force_search(
    query_embeddings,
    gallery_embeddings,
    top_k=5,
    query_ids=None,
    gallery_ids=None,
    batch_size=32,
):
    """
    Exact NumPy Euclidean nearest-neighbor search.

    Returns:
        indices: gallery row indices, shape (queries, top_k)
        distances: Euclidean distances, same shape

    When IDs are provided, matching query/gallery IDs
    are excluded to prevent self-retrieval.

    No approximate index or external search library
    is used.
    """
    queries = _validate_embeddings(
        query_embeddings, "Query embeddings"
    )
    gallery = _validate_embeddings(
        gallery_embeddings, "Gallery embeddings"
    )

    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("Embedding dimensions do not match")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if (query_ids is None) != (gallery_ids is None):
        raise ValueError(
            "Provide both query_ids and gallery_ids, "
            "or neither"
        )

    if query_ids is not None:
        query_ids = _validate_ids(
            query_ids, len(queries), "query_ids"
        )
        gallery_ids = _validate_ids(
            gallery_ids, len(gallery), "gallery_ids"
        )

        if not np.isin(query_ids, gallery_ids).all():
            raise ValueError(
                "All query IDs must exist in the gallery"
            )

        maximum_k = len(gallery) - 1

    else:
        maximum_k = len(gallery)

    if top_k > maximum_k:
        raise ValueError(
            f"top_k={top_k} exceeds available "
            f"gallery items ({maximum_k})"
        )

    gallery_norms = np.sum(
        gallery * gallery, axis=1
    )

    all_indices = []
    all_distances = []

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))

        query_batch = queries[start:end]

        query_norms = np.sum(
            query_batch * query_batch, axis=1
        )

        squared_distances = (
            query_norms[:, None]
            + gallery_norms[None, :]
            - 2.0 * (query_batch @ gallery.T)
        )

        np.maximum(
            squared_distances,
            0.0,
            out=squared_distances,
        )

        distances = np.sqrt(squared_distances)

        if query_ids is not None:
            same_identity = (
                query_ids[start:end, None]
                == gallery_ids[None, :]
            )

            distances[same_identity] = np.inf

        # Stable sorting gives deterministic results
        # when multiple candidates have equal distances.
        ranked = np.argsort(
            distances,
            axis=1,
            kind="stable",
        )[:, :top_k]

        ranked_distances = np.take_along_axis(
            distances,
            ranked,
            axis=1,
        )

        all_indices.append(ranked)
        all_distances.append(ranked_distances)

    return (
        np.concatenate(all_indices, axis=0),
        np.concatenate(all_distances, axis=0),
    )


def evaluate_retrieval(
    query_embeddings,
    query_labels,
    query_ids,
    gallery_embeddings,
    gallery_labels,
    gallery_ids,
    ks=(1, 5, 10),
    batch_size=32,
):
    """
    Evaluate complete-gallery retrieval.

    Recall@K:
        relevant results in top K / all relevant
        gallery results for the query.

    Hit@K:
        fraction of queries with at least one
        relevant result in top K.

    mAP:
        mean average precision over the complete
        ranked gallery, excluding the query image.

    A relevant result has the same CIFAR-100 class.
    """
    queries = _validate_embeddings(
        query_embeddings, "Query embeddings"
    )
    gallery = _validate_embeddings(
        gallery_embeddings, "Gallery embeddings"
    )

    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if query_labels.ndim != 1:
        raise ValueError("query_labels must be 1D")

    if gallery_labels.ndim != 1:
        raise ValueError("gallery_labels must be 1D")

    if len(query_labels) != len(queries):
        raise ValueError("Query label count mismatch")

    if len(gallery_labels) != len(gallery):
        raise ValueError("Gallery label count mismatch")

    if len(gallery) < 2:
        raise ValueError(
            "Gallery needs at least two images"
        )

    if not ks:
        raise ValueError("ks cannot be empty")

    if any(
        not isinstance(k, (int, np.integer)) or k < 1
        for k in ks
    ):
        raise ValueError("All K values must be positive")

    query_ids = _validate_ids(
        query_ids, len(queries), "query_ids"
    )
    gallery_ids = _validate_ids(
        gallery_ids, len(gallery), "gallery_ids"
    )

    if not np.isin(query_ids, gallery_ids).all():
        raise ValueError(
            "Queries must be members of the gallery"
        )

    # Rank every candidate except the query itself.
    rankings, _ = brute_force_search(
        queries,
        gallery,
        top_k=len(gallery) - 1,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        batch_size=batch_size,
    )

    recall_totals = {int(k): 0.0 for k in ks}
    hit_totals = {int(k): 0.0 for k in ks}
    average_precisions = []

    for query_row, ranked_indices in enumerate(rankings):
        query_label = query_labels[query_row]
        query_id = query_ids[query_row]

        relevant_gallery = (
            (gallery_labels == query_label)
            & (gallery_ids != query_id)
        )

        relevant_count = int(
            relevant_gallery.sum()
        )

        if relevant_count == 0:
            raise ValueError(
                "At least one query has no other "
                "same-class image in the gallery"
            )

        ranked_relevant = relevant_gallery[
            ranked_indices
        ].astype(np.int64)

        cumulative_relevant = np.cumsum(
            ranked_relevant
        )

        ranks = np.arange(
            1,
            len(ranked_relevant) + 1,
            dtype=np.float64,
        )

        precision_at_rank = (
            cumulative_relevant / ranks
        )

        average_precision = (
            np.sum(
                precision_at_rank * ranked_relevant
            )
            / relevant_count
        )

        average_precisions.append(
            float(average_precision)
        )

        for k in recall_totals:
            effective_k = min(
                k, len(ranked_relevant)
            )

            correct = int(
                cumulative_relevant[effective_k - 1]
            )

            recall_totals[k] += (
                correct / relevant_count
            )

            hit_totals[k] += float(correct > 0)

    query_count = len(queries)

    result = {
        "queries": query_count,
        "gallery_size": len(gallery),
        "map": float(
            np.mean(average_precisions)
        ),
    }

    for k in recall_totals:
        result[f"recall_at_{k}"] = (
            recall_totals[k] / query_count
        )
        result[f"hit_at_{k}"] = (
            hit_totals[k] / query_count
        )

    return result
