
import pytest
import torch

from semi_hard_mining import semi_hard_triplet_loss


def test_semi_hard_negative_is_selected():
    embeddings = torch.tensor([
        [0.00, 0.0],
        [0.20, 0.0],
        [0.35, 0.0],
        [0.55, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    loss, stats = semi_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
        generator=torch.Generator().manual_seed(42),
        return_stats=True,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0
    assert stats["semi_hard_fraction"] > 0


def test_random_fallback_when_no_semi_hard_exists():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 0.0],
        [5.1, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    loss, stats = semi_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
        return_stats=True,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    assert stats["semi_hard_fraction"] == 0.0
    assert stats["fallback_fraction"] == 1.0


def test_semi_hard_gradients():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.2, 0.0],
        [0.35, 0.0],
        [0.55, 0.0],
    ], requires_grad=True)

    labels = torch.tensor([0, 0, 1, 1])

    loss = semi_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.5,
    )

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_semi_hard_reproducible():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],
        [0.4, 0.0],
        [0.5, 0.0],
        [0.6, 0.0],
        [0.7, 0.0],
    ])

    labels = torch.tensor([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    first, first_stats = semi_hard_triplet_loss(
        embeddings,
        labels,
        generator=torch.Generator().manual_seed(42),
        return_stats=True,
    )

    second, second_stats = semi_hard_triplet_loss(
        embeddings,
        labels,
        generator=torch.Generator().manual_seed(42),
        return_stats=True,
    )

    assert torch.equal(first, second)
    assert first_stats == second_stats


def test_semi_hard_statistics():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.2, 0.0],
        [0.35, 0.0],
        [0.55, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    _, stats = semi_hard_triplet_loss(
        embeddings,
        labels,
        return_stats=True,
    )

    assert (
        stats["semi_hard_fraction"]
        + stats["fallback_fraction"]
    ) == pytest.approx(1.0)

    for name in (
        "semi_hard_fraction",
        "fallback_fraction",
        "active_fraction",
    ):
        assert 0 <= stats[name] <= 1


@pytest.mark.parametrize(
    "labels",
    [
        [0, 1, 2, 3],
        [0, 0, 0, 0],
        [0, 0, 1],
    ],
)
def test_invalid_labels(labels):
    embeddings = torch.randn(4, 3)

    with pytest.raises(ValueError):
        semi_hard_triplet_loss(
            embeddings,
            torch.tensor(labels),
        )


def test_invalid_margin():
    embeddings = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError):
        semi_hard_triplet_loss(
            embeddings,
            labels,
            margin=0,
        )


def test_all_identical_embeddings_stay_finite():
    embeddings = torch.ones(
        4,
        3,
        requires_grad=True,
    )

    labels = torch.tensor([0, 0, 1, 1])

    loss, stats = semi_hard_triplet_loss(
        embeddings,
        labels,
        margin=0.3,
        return_stats=True,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.3, abs=1e-4)
    assert stats["semi_hard_fraction"] == 0.0
