
import torch

from model import EmbeddingNet


def test_output_shape():
    model = EmbeddingNet(embedding_dim=128)
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 128)


def test_embedding_normalization():
    model = EmbeddingNet()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        embeddings = model(x)

    norms = torch.linalg.vector_norm(embeddings, dim=1)

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
    )


def test_embedding_dimensions():
    for dim in (64, 128, 256):
        model = EmbeddingNet(embedding_dim=dim)
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, dim)


def test_backward_pass():
    model = EmbeddingNet(embedding_dim=128)
    model.train()

    x = torch.randn(4, 3, 32, 32)
    embeddings = model(x)

    target = torch.randn_like(embeddings)
    loss = (embeddings - target).square().mean()
    loss.backward()

    backbone_grad = model.backbone.conv1.weight.grad
    head_grad = model.backbone.fc.weight.grad

    assert backbone_grad is not None
    assert head_grad is not None

    assert torch.isfinite(backbone_grad).all()
    assert torch.isfinite(head_grad).all()

    assert backbone_grad.abs().sum() > 0
    assert head_grad.abs().sum() > 0

