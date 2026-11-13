
import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from optimizers import Adam
from scripts.train_full import (
    capture_rng_state,
    load_training_checkpoint,
    make_config,
    make_history_row,
    restore_rng_state,
    save_history,
    save_training_checkpoint,
    train_full_epoch,
    validate_args,
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


def make_args():
    return SimpleNamespace(
        epochs=3,
        threads=8,
        batch_size=64,
        search_batch_size=32,
        log_every=100,
        eval_every=1,
        queries_per_class=5,
        seed=42,
        lr=0.0001,
        margin=0.3,
    )


def make_components():
    torch.manual_seed(42)

    model = TinyModel()
    optimizer = Adam(
        model.parameters(),
        lr=0.0001,
    )

    return model, optimizer


def test_full_epoch_processes_every_image():
    model, optimizer = make_components()
    loader = TinyLoader()

    result = train_full_epoch(
        model,
        loader,
        optimizer,
        epoch=4,
        log_every=2,
    )

    assert loader.batch_sampler.epoch == 4
    assert result["images"] == 8
    assert np.isfinite(result["loss"])


def test_config_excludes_target_epochs():
    args = make_args()

    first = make_config(args)

    args.epochs = 30
    second = make_config(args)

    assert first == second
    assert first["mining"] == "random"
    assert first["embedding_dim"] == 128


def test_invalid_training_arguments():
    args = make_args()

    args.threads = 0

    with pytest.raises(ValueError):
        validate_args(args)

    args = make_args()
    args.lr = -0.1

    with pytest.raises(ValueError):
        validate_args(args)

    args = make_args()
    args.queries_per_class = 61

    with pytest.raises(ValueError):
        validate_args(args)


def test_checkpoint_round_trip(tmp_path):
    model, optimizer = make_components()
    config = make_config(make_args())

    path = tmp_path / "last.pt"

    original_weights = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    baseline = {
        "map": 0.02,
        "hit_at_1": 0.08,
    }

    history = [{
        "epoch": 1,
        "train_loss": 0.22,
    }]

    save_training_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=history,
        config=config,
        best_map=0.04,
        best_epoch=1,
        baseline_metrics=baseline,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)

    restored = load_training_checkpoint(
        path,
        model,
        optimizer,
        expected_config=config,
    )

    assert restored["epoch"] == 1
    assert restored["best_epoch"] == 1
    assert restored["best_map"] == 0.04
    assert restored["history"] == history
    assert restored["baseline_metrics"] == baseline

    for key, original in original_weights.items():
        assert torch.equal(
            model.state_dict()[key],
            original,
        )


def test_checkpoint_rejects_changed_lr(tmp_path):
    model, optimizer = make_components()
    config = make_config(make_args())

    path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=[],
        config=config,
        best_map=0.03,
        best_epoch=1,
        baseline_metrics={"map": 0.02},
    )

    wrong_config = config.copy()
    wrong_config["lr"] = 0.001

    with pytest.raises(
        ValueError,
        match="configuration",
    ):
        load_training_checkpoint(
            path,
            model,
            optimizer,
            expected_config=wrong_config,
        )


def test_rng_restoration():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    state = capture_rng_state()

    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    restore_rng_state(state)

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(
        torch.rand(3),
        expected[2],
    )


def test_history_row_and_csv(tmp_path):
    training = {
        "loss": 0.25,
        "images": 48000,
    }

    validation = {
        "recall_at_1": 0.001,
        "recall_at_5": 0.005,
        "recall_at_10": 0.01,
        "hit_at_1": 0.10,
        "hit_at_5": 0.25,
        "hit_at_10": 0.35,
        "map": 0.04,
    }

    row = make_history_row(
        epoch=1,
        training=training,
        validation=validation,
        epoch_seconds=500.0,
    )

    path = tmp_path / "history.csv"

    save_history(path, [row])

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train_images"] == "48000"
    assert float(rows[0]["val_map"]) == 0.04
    assert float(rows[0]["val_hit_at_1"]) == 0.10


def test_history_row_without_validation():
    row = make_history_row(
        epoch=2,
        training={
            "loss": 0.20,
            "images": 48000,
        },
        validation=None,
        epoch_seconds=480.0,
    )

    assert row["val_map"] is None
    assert row["train_images"] == 48000
