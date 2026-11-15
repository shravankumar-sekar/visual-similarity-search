
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from scripts.run_ablation import (
    build_parser,
    calculate_loss,
    is_collapsed,
    make_config,
    make_optimizer,
    make_row,
    save_csv,
    train_ablation_epoch,
    validate_args,
)


def make_args():
    return SimpleNamespace(
        dimension=128,
        margin=0.3,
        optimizer="adam",
        mining="random",
        lr=0.0001,
        epochs=3,
        threads=8,
        seed=42,
        queries_per_class=5,
        eval_batch_size=64,
        search_batch_size=32,
        log_every=10,
    )


class TinySampler:
    def __init__(self):
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch


class TinyLoader:
    def __init__(self):
        self.batch_sampler = TinySampler()

        self.images = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.7, 0.3],
        ])

        self.labels = torch.tensor([
            0, 0, 1, 1,
            0, 0, 1, 1,
        ])

        self.dataset = list(range(8))

    def __len__(self):
        return 2

    def __iter__(self):
        for start in (0, 4):
            yield (
                self.images[start:start + 4],
                self.labels[start:start + 4],
            )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, images):
        return F.normalize(
            self.linear(images),
            dim=1,
        )


def test_parser_defaults():
    parser = build_parser()

    args = parser.parse_args([
        "--run-dir",
        "checkpoints/test_ablation",
    ])

    validate_args(args)

    assert args.dimension == 128
    assert args.margin == 0.3
    assert args.optimizer == "adam"
    assert args.mining == "random"


def test_configuration_allows_epoch_extension():
    args = make_args()

    first = make_config(args)

    args.epochs = 30
    second = make_config(args)

    assert first == second
    assert first["embedding_dim"] == 128


@pytest.mark.parametrize(
    "name",
    ["sgd", "adam", "nadam"],
)
def test_custom_optimizer_updates(name):
    torch.manual_seed(42)

    model = torch.nn.Linear(3, 2)

    optimizer = make_optimizer(
        name,
        model.parameters(),
        lr=0.0001,
    )

    original = model.weight.detach().clone()

    optimizer.zero_grad()

    loss = model(
        torch.ones(2, 3)
    ).square().mean()

    loss.backward()
    optimizer.step()

    assert not torch.equal(
        original,
        model.weight.detach(),
    )


@pytest.mark.parametrize(
    "mining",
    ["random", "hard", "semi-hard"],
)
def test_all_mining_losses_are_differentiable(mining):
    torch.manual_seed(42)

    embeddings = torch.randn(
        8, 4, requires_grad=True
    )

    labels = torch.tensor([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    loss = calculate_loss(
        embeddings,
        labels,
        mining=mining,
        margin=3.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_full_tiny_epoch():
    torch.manual_seed(42)

    model = TinyModel()
    loader = TinyLoader()

    optimizer = make_optimizer(
        "adam",
        model.parameters(),
        lr=0.0001,
    )

    result = train_ablation_epoch(
        model,
        loader,
        optimizer,
        epoch=2,
        mining="random",
        margin=3.0,
        log_every=2,
    )

    assert result["images"] == 8
    assert np.isfinite(result["loss"])
    assert loader.batch_sampler.epoch == 2


def test_collapse_guard():
    assert is_collapsed(0.05, 0.50)
    assert not is_collapsed(0.30, 0.50)
    assert not is_collapsed(0.125, 0.50)

    with pytest.raises(ValueError):
        is_collapsed(0.1, 0.0)


def test_history_row_and_csv(tmp_path):
    training = {
        "loss": 0.2,
        "images": 48000,
    }

    validation = {
        "map": 0.1,
        "hit_at_1": 0.2,
        "hit_at_5": 0.4,
        "hit_at_10": 0.5,
        "spread": 0.4,
    }

    row = make_row(
        epoch=1,
        training=training,
        validation=validation,
        baseline_spread=0.5,
        seconds=500.0,
    )

    path = tmp_path / "metrics.csv"

    save_csv(path, [row])

    assert row["train_images"] == 48000
    assert row["collapsed"] is False
    assert row["spread_ratio"] == pytest.approx(0.8)

    content = path.read_text()

    assert "val_map" in content
    assert "0.1" in content


def test_invalid_configuration():
    args = make_args()
    args.margin = -0.3

    with pytest.raises(ValueError):
        validate_args(args)

    args = make_args()
    args.dimension = 512

    with pytest.raises(ValueError):
        validate_args(args)

    args = make_args()
    args.lr = float("nan")

    with pytest.raises(ValueError):
        validate_args(args)


def test_unknown_optimizer_and_mining():
    with pytest.raises(ValueError):
        make_optimizer(
            "unknown",
            torch.nn.Linear(2, 2).parameters(),
            lr=0.001,
        )

    embeddings = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError):
        calculate_loss(
            embeddings,
            labels,
            mining="unknown",
            margin=0.3,
        )
