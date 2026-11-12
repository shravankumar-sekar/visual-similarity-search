
import torch
from torch.nn import functional as F


def random_triplet_loss(
    embeddings, labels, margin=0.3, generator=None
):
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")

    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("Invalid labels")

    if margin < 0:
        raise ValueError("Margin cannot be negative")

    same_class = labels[:, None] == labels[None, :]

    positive_mask = same_class.clone()
    positive_mask.fill_diagonal_(False)

    negative_mask = ~same_class

    if not positive_mask.any(dim=1).all():
        raise ValueError("Every anchor needs a positive sample")

    if not negative_mask.any(dim=1).all():
        raise ValueError("Every anchor needs a negative sample")

    positive_indices = torch.multinomial(
        positive_mask.float(),
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    negative_indices = torch.multinomial(
        negative_mask.float(),
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    positives = embeddings[positive_indices]
    negatives = embeddings[negative_indices]

    positive_distances = torch.norm(
        embeddings - positives, dim=1
    )
    negative_distances = torch.norm(
        embeddings - negatives, dim=1
    )

    losses = F.relu(
        positive_distances - negative_distances + margin
    )

    return losses.mean()
