
import csv

import pytest
import torch
from torch.nn import functional as F

from model import EmbeddingNet
from optimizers import Adam

from scripts.compare_mining import (
    calculate_loss,
    detect_collapse,
    load_warmup_branch,
    make_result_row,
    save_results,
    train_branch_epoch,
)

from scripts.experiment_random_mining import (
    make_experiment_config,
    save_experiment_checkpoint,
)


class TinySampler:
    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch


class TinyLoader:
    def __init__(self):
        self.batch_sampler = TinySampler()

        self.images = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.8, 0.2],
            [0.0, 0.7, 0.3],
        ])

        self.labels = torch.tensor([
            0, 0, 0, 0,
            1, 1, 1, 1,
        ])

    def __len__(self):
        return 2

    def __iter__(self):
        for _ in range(2):
            yield self.images, self.labels


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(3, 4)

    def forward(self, images):
        return F.normalize(
            self.linear(images),
            dim=1,
        )


def make_tiny_components():
    torch.manual_seed(42)

    model = TinyEmbeddingModel()
    optimizer = Adam(
        model.parameters(),
        lr=0.0001,
    )

    return model, TinyLoader(), optimizer


def test_collapse_detection():
    assert detect_collapse(
        0.05,
        0.50,
    )

    assert not detect_collapse(
        0.40,
        0.50,
    )

    assert not detect_collapse(
        0.125,
        0.50,
    )

    with pytest.raises(ValueError):
        detect_collapse(0.1, 0.0)


def test_mining_loss_selection():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [2.0, 0.0],
        [2.1, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    for strategy in ("random", "hard"):
        loss = calculate_loss(
            embeddings,
            labels,
            strategy=strategy,
        )

        assert torch.isfinite(loss)
        assert loss.ndim == 0

    with pytest.raises(ValueError):
        calculate_loss(
            embeddings,
            labels,
            strategy="unknown",
        )


@pytest.mark.parametrize(
    "strategy",
    ["random", "hard"],
)
def test_branch_training_updates_model(strategy):
    model, loader, optimizer = (
        make_tiny_components()
    )

    original = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    # Test-only margin:
    # Normalized embeddings have L2 distances <= 2.
    # A margin of 3 guarantees active triplets.
    # Production training still uses margin=0.3.
    test_margin = 3.0

    loss = train_branch_epoch(
        model,
        loader,
        optimizer,
        strategy=strategy,
        steps=2,
        epoch=4,
        margin=test_margin,
        log_every=2,
    )

    assert torch.isfinite(
        torch.tensor(loss)
    )

    # The test must exercise a nonzero loss.
    assert loss > 0.0

    assert loader.batch_sampler.epoch == 4

    changed = any(
        not torch.equal(
            before,
            after.detach(),
        )
        for before, after in zip(
            original,
            model.parameters(),
        )
    )

    assert changed, (
        f"{strategy} mining produced active triplets "
        "but did not update model parameters"
    )


def test_results_csv(tmp_path):
    diagnostic = {
        "random_ordering": 0.65,
        "hard_ordering": 0.01,
        "positive_distance": 0.4,
        "negative_distance": 0.6,
        "embedding_spread": 0.45,
    }

    row = make_result_row(
        "random",
        epoch=1,
        diagnostic=diagnostic,
        train_loss=0.23,
        collapsed=False,
    )

    path = tmp_path / "comparison.csv"

    save_results(path, [row])

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["branch"] == "random"
    assert rows[0]["epoch"] == "1"

    assert float(
        rows[0]["random_ordering"]
    ) == 0.65


def test_warmup_checkpoint_starts_branch(tmp_path):
    torch.manual_seed(42)

    model = EmbeddingNet(
        embedding_dim=128
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.0001,
    )

    config = make_experiment_config()

    path = tmp_path / "warmup.pt"

    save_experiment_checkpoint(
        path,
        model,
        optimizer,
        epoch=3,
        history=[],
        config=config,
        baseline={
            "embedding_spread": 0.44,
        },
    )

    original_weights = {
        key: tensor.detach().clone()
        for key, tensor in model.state_dict().items()
    }

    (
        restored_model,
        restored_optimizer,
        restored,
        restored_config,
    ) = load_warmup_branch(
        path,
        new_lr=0.00003,
    )

    assert restored["epoch"] == 3
    assert restored_config == config

    assert restored_optimizer.lr == pytest.approx(
        0.00003
    )

    for key, original in original_weights.items():
        assert torch.equal(
            restored_model.state_dict()[key],
            original,
        )
