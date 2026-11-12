
import torch
from torch.nn import functional as F


def pairwise_distances(embeddings):
    squared_norms = (embeddings ** 2).sum(dim=1)

    squared_distances = (
        squared_norms.unsqueeze(1)
        + squared_norms.unsqueeze(0)
        - 2 * embeddings @ embeddings.T
    )

    squared_distances = squared_distances.clamp(min=0)

    return torch.sqrt(squared_distances + 1e-12)



def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D tensor")

    if labels.ndim != 1 or len(labels) != len(embeddings):
        raise ValueError("Invalid labels")

    distances = pairwise_distances(embeddings)

    same_class = labels[:, None] == labels[None, :]
    positive_mask = same_class.clone()
    positive_mask.fill_diagonal_(False)
    negative_mask = ~same_class

    if not positive_mask.any(dim=1).all():
        raise ValueError("Every anchor needs a positive sample")

    if not negative_mask.any(dim=1).all():
        raise ValueError("Every anchor needs a negative sample")

    hardest_positive = distances.masked_fill(
        ~positive_mask, float("-inf")
    ).max(dim=1).values

    hardest_negative = distances.masked_fill(
        ~negative_mask, float("inf")
    ).min(dim=1).values

    losses = torch.relu(
        hardest_positive - hardest_negative + margin
    )

    return losses.mean()

