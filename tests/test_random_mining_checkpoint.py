
import csv
import random

import numpy as np
import pytest
import torch

from optimizers import Adam
from scripts.experiment_random_mining import (
    make_experiment_config,
    save_experiment_checkpoint,
    load_experiment_checkpoint,
    save_experiment_metrics,
)


def make_components():
    torch.manual_seed(42)

    model = torch.nn.Linear(3, 2)
    optimizer = Adam(
        model.parameters(),
        lr=0.0001,
    )

    config = make_experiment_config()

    history = [{
        "epoch": 1,
        "train_loss": 0.25,
        "random_ordering": 0.64,
        "hard_ordering": 0.01,
        "positive_distance": 0.45,
        "negative_distance": 0.60,
        "embedding_spread": 0.40,
    }]

    baseline = {
        "random_ordering": 0.59,
        "hard_ordering": 0.0,
        "positive_distance": 0.17,
        "negative_distance": 0.18,
        "embedding_spread": 0.13,
    }

    return (
        model,
        optimizer,
        config,
        history,
        baseline,
    )


def perform_optimizer_update(model, optimizer):
    inputs = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    ])

    optimizer.zero_grad()

    prediction = model(inputs)
    loss = prediction.square().mean()

    loss.backward()
    optimizer.step()


def test_checkpoint_restores_model_weights(tmp_path):
    (
        model,
        optimizer,
        config,
        history,
        baseline,
    ) = make_components()

    path = tmp_path / "model.pt"

    original_weights = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    save_experiment_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=history,
        config=config,
        baseline=baseline,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    restored = load_experiment_checkpoint(
        path,
        model,
        optimizer,
        config,
    )

    assert restored["epoch"] == 1
    assert restored["history"] == history
    assert restored["baseline"] == baseline

    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            original_weights[key],
        )


def test_checkpoint_restores_optimizer_state(tmp_path):
    (
        model,
        optimizer,
        config,
        history,
        baseline,
    ) = make_components()

    perform_optimizer_update(
        model,
        optimizer,
    )

    path = tmp_path / "optimizer.pt"

    save_experiment_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=history,
        config=config,
        baseline=baseline,
    )

    saved_optimizer_state = optimizer.state_dict()

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = Adam(
        restored_model.parameters(),
        lr=0.0001,
    )

    load_experiment_checkpoint(
        path,
        restored_model,
        restored_optimizer,
        config,
    )

    restored_state = restored_optimizer.state_dict()

    assert (
        saved_optimizer_state.keys()
        == restored_state.keys()
    )

    def compare_values(first, second):
        if isinstance(first, torch.Tensor):
            assert torch.equal(first, second)
        elif isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                compare_values(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            assert len(first) == len(second)
            for a, b in zip(first, second):
                compare_values(a, b)
        else:
            assert first == second

    compare_values(
        saved_optimizer_state,
        restored_state,
    )


def test_checkpoint_restores_rng_states(tmp_path):
    (
        model,
        optimizer,
        config,
        history,
        baseline,
    ) = make_components()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    path = tmp_path / "rng.pt"

    save_experiment_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=history,
        config=config,
        baseline=baseline,
    )

    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(3)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    load_experiment_checkpoint(
        path,
        model,
        optimizer,
        config,
    )

    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(
        torch.rand(3),
        expected_torch,
    )


def test_checkpoint_rejects_wrong_configuration(tmp_path):
    (
        model,
        optimizer,
        config,
        history,
        baseline,
    ) = make_components()

    path = tmp_path / "config.pt"

    save_experiment_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=history,
        config=config,
        baseline=baseline,
    )

    wrong_config = make_experiment_config(
        lr=0.001,
    )

    with pytest.raises(
        ValueError,
        match="configuration",
    ):
        load_experiment_checkpoint(
            path,
            model,
            optimizer,
            wrong_config,
        )


def test_checkpoint_missing_file(tmp_path):
    (
        model,
        optimizer,
        config,
        _,
        _,
    ) = make_components()

    with pytest.raises(FileNotFoundError):
        load_experiment_checkpoint(
            tmp_path / "missing.pt",
            model,
            optimizer,
            config,
        )


def test_metrics_csv_round_trip(tmp_path):
    (
        _,
        _,
        _,
        history,
        _,
    ) = make_components()

    path = tmp_path / "metrics.csv"

    save_experiment_metrics(
        path,
        history,
    )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert float(
        rows[0]["random_ordering"]
    ) == 0.64

    assert float(
        rows[0]["embedding_spread"]
    ) == 0.40
