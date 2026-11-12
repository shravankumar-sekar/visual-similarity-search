
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            in_features, embedding_dim
        )

    def forward(self, x):
        embeddings = self.backbone(x)
        return F.normalize(embeddings, p=2, dim=1)
