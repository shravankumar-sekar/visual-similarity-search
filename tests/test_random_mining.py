
import pytest
import torch

from random_mining import random_triplet_loss


def test_random_triplet_zero_loss():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [3.0, 0.0],
        [3.1, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])

    loss = random_triplet_loss(
        embeddings, labels, margin=0.3
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_random_triplet_positive_loss():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.5, 0.0],
        [0.6, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])

    generator = torch.Generator().manual_seed(42)

    loss = random_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
        generator=generator,
    )

    assert loss.item() > 0


def test_random_triplet_gradients():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.5, 0.0],
        [0.6, 0.0],
    ], requires_grad=True)

    labels = torch.tensor([0, 0, 1, 1])

    loss = random_triplet_loss(
        embeddings, labels, margin=0.3
    )
    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_random_triplet_reproducibility():
    embeddings = torch.arange(16).float().reshape(8, 2)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    a = torch.Generator().manual_seed(42)
    b = torch.Generator().manual_seed(42)

    loss_a = random_triplet_loss(
        embeddings, labels, generator=a
    )
    loss_b = random_triplet_loss(
        embeddings, labels, generator=b
    )

    assert torch.equal(loss_a, loss_b)


def test_random_triplet_invalid_batch():
    embeddings = torch.randn(4, 3)

    with pytest.raises(ValueError):
        random_triplet_loss(
            embeddings, torch.tensor([0, 1, 2, 3])
        )

    with pytest.raises(ValueError):
        random_triplet_loss(
            embeddings, torch.tensor([0, 0, 0, 0])
        )

    with pytest.raises(ValueError):
        random_triplet_loss(
            embeddings,
            torch.tensor([0, 0, 1, 1]),
            margin=-0.1,
        )
