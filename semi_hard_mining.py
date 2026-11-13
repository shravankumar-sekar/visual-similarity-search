
import torch
from torch.nn import functional as F

from loss import pairwise_distances


def semi_hard_triplet_loss(
    embeddings,
    labels,
    margin=0.3,
    generator=None,
    return_stats=False,
):
    """
    Sample one positive for every anchor.

    Select the closest negative satisfying:
        d(a,p) < d(a,n) < d(a,p) + margin

    When none exists, select a random negative.

    The selected distances remain differentiable.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")

    if not embeddings.is_floating_point():
        raise ValueError("Embeddings must be floating point")

    if labels.ndim != 1:
        raise ValueError("Labels must be 1D")

    if len(embeddings) != len(labels):
        raise ValueError("Labels and embeddings must match")

    if len(labels) < 3:
        raise ValueError("Batch is too small")

    if margin <= 0:
        raise ValueError("Margin must be positive")

    if not torch.isfinite(embeddings).all().item():
        raise ValueError("Embeddings must be finite")

    labels = labels.to(embeddings.device)

    distances = pairwise_distances(embeddings)

    same = labels[:, None] == labels[None, :]

    positive_mask = same.clone()
    positive_mask.fill_diagonal_(False)

    negative_mask = ~same

    if not positive_mask.any(dim=1).all().item():
        raise ValueError(
            "Every anchor needs another same-class sample"
        )

    if not negative_mask.any(dim=1).all().item():
        raise ValueError(
            "Every anchor needs a different-class sample"
        )

    positive_indices = torch.multinomial(
        positive_mask.float(),
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    anchors = torch.arange(
        len(labels),
        device=embeddings.device,
    )

    positive_distances = distances[
        anchors,
        positive_indices,
    ]

    # We choose indices using detached distances.
    # Actual selected distances retain their gradients.
    candidate_distances = distances.detach()

    positive_reference = (
        positive_distances.detach().unsqueeze(1)
    )

    eligible = (
        negative_mask
        & (candidate_distances > positive_reference)
        & (
            candidate_distances
            < positive_reference + margin
        )
    )

    has_semi_hard = eligible.any(dim=1)

    closest_semi_hard = (
        candidate_distances
        .masked_fill(~eligible, float("inf"))
        .argmin(dim=1)
    )

    fallback_negative = torch.multinomial(
        negative_mask.float(),
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    negative_indices = torch.where(
        has_semi_hard,
        closest_semi_hard,
        fallback_negative,
    )

    negative_distances = distances[
        anchors,
        negative_indices,
    ]

    individual_losses = F.relu(
        positive_distances
        - negative_distances
        + margin
    )

    loss = individual_losses.mean()

    if not return_stats:
        return loss

    stats = {
        "semi_hard_fraction": (
            has_semi_hard.float().mean().item()
        ),
        "fallback_fraction": (
            (~has_semi_hard).float().mean().item()
        ),
        "active_fraction": (
            (individual_losses.detach() > 0)
            .float()
            .mean()
            .item()
        ),
        "positive_distance": (
            positive_distances.detach().mean().item()
        ),
        "negative_distance": (
            negative_distances.detach().mean().item()
        ),
    }

    return loss, stats
