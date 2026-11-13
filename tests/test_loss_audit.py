
import pytest
import torch

from loss import (
    batch_hard_triplet_loss,
    pairwise_distances,
)


def test_pairwise_distance_geometry():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [3.0, 0.0],
        [0.0, 4.0],
    ])

    distances = pairwise_distances(embeddings)

    assert distances.shape == (3, 3)

    assert distances[0, 1].item() == pytest.approx(
        3.0, abs=1e-4
    )
    assert distances[0, 2].item() == pytest.approx(
        4.0, abs=1e-4
    )
    assert distances[1, 2].item() == pytest.approx(
        5.0, abs=1e-4
    )

    assert torch.allclose(
        distances,
        distances.T,
        atol=1e-4,
    )


def test_batch_hard_matches_manual_example():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    # Hardest positive distances: [2, 2, 2, 2]
    # Nearest negatives:          [1, 1, 1, 1]
    # Each hinge loss = 2 - 1 + 0.3 = 1.3
    actual = batch_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
    )

    assert actual.item() == pytest.approx(
        1.3, abs=1e-4
    )


def test_batch_hard_produces_finite_gradients():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
    ], requires_grad=True)

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
    )

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_batch_hard_handles_collapsed_embeddings():
    embeddings = torch.ones(
        4,
        3,
        requires_grad=True,
    )

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(
        0.3, abs=1e-4
    )

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
