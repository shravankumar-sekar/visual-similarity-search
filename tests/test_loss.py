
import torch
import pytest

from loss import pairwise_distances, batch_hard_triplet_loss


def test_pairwise_distance():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [3.0, 4.0],
    ])

    distances = pairwise_distances(embeddings)

    assert torch.allclose(
        distances[0, 1],
        torch.tensor(5.0),
        atol=1e-5,
    )


def test_triplet_loss_zero():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 0.0],
        [5.1, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(
        embeddings, labels, margin=0.3
    )

    assert torch.isclose(loss, torch.tensor(0.0))


def test_triplet_loss_positive():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0],
        [1.1, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(
        embeddings, labels, margin=0.3
    )

    assert loss.item() > 0


def test_triplet_loss_gradients():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0],
        [1.1, 0.0],
    ], requires_grad=True)

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(embeddings, labels)

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()




def test_hard_mining_selection():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [1.0, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(
        embeddings, labels, margin=0.3
    )

    # Hardest positives are 0.2 apart.
    # Hardest negatives are 0.6, 0.6, 0.6, 0.8 apart.
    # All triplets satisfy the margin.
    assert torch.isclose(
        loss, torch.tensor(0.0), atol=1e-5
    )


def test_missing_positive_raises_error():
    embeddings = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 1])

    with pytest.raises(ValueError):
        batch_hard_triplet_loss(embeddings, labels)


def test_single_class_raises_error():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 0, 0])

    with pytest.raises(ValueError):
        batch_hard_triplet_loss(embeddings, labels)


def test_batch_hard_loss_expected_value():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.5, 0.0],
        [0.6, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    loss = batch_hard_triplet_loss(
        embeddings, labels, margin=0.3
    )

    expected = torch.tensor(0.675)
    assert torch.allclose(loss, expected, atol=1e-5)


