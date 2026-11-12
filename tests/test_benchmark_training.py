
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from optimizers import Adam
from scripts.benchmark_training import benchmark_steps


def make_test_components():
    torch.manual_seed(42)

    images = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.7, 0.3],
    ])

    labels = torch.tensor([
        0, 0, 1, 1,
        0, 0, 1, 1,
    ])

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
        shuffle=False,
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    return model, loader, optimizer


def test_benchmark_returns_timings():
    model, loader, optimizer = make_test_components()

    result = benchmark_steps(
        model,
        loader,
        optimizer,
        steps=2,
        warmup=1,
    )

    assert result["steps"] == 2
    assert result["batch_size"] == 4
    assert result["images_per_second"] > 0

    for name in (
        "data",
        "forward",
        "loss",
        "backward",
        "optimizer",
        "total",
    ):
        assert name in result["timings"]
        assert math.isfinite(result["timings"][name])
        assert result["timings"][name] >= 0

    assert result["timings"]["total"] > 0
    assert math.isfinite(result["mean_loss"])


def test_benchmark_updates_parameters():
    model, loader, optimizer = make_test_components()

    before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    benchmark_steps(
        model,
        loader,
        optimizer,
        steps=2,
        warmup=0,
    )

    changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )

    assert changed


def test_benchmark_restarts_loader():
    model, loader, optimizer = make_test_components()

    result = benchmark_steps(
        model,
        loader,
        optimizer,
        steps=4,
        warmup=1,
    )

    assert result["steps"] == 4
    assert result["batch_size"] == 4


def test_benchmark_invalid_steps():
    model, loader, optimizer = make_test_components()

    with pytest.raises(ValueError):
        benchmark_steps(
            model,
            loader,
            optimizer,
            steps=0,
        )

    with pytest.raises(ValueError):
        benchmark_steps(
            model,
            loader,
            optimizer,
            steps=1,
            warmup=-1,
        )


def test_benchmark_empty_loader():
    model, _, optimizer = make_test_components()

    empty_images = torch.empty((0, 3))
    empty_labels = torch.empty((0,), dtype=torch.long)

    loader = DataLoader(
        TensorDataset(empty_images, empty_labels),
        batch_size=4,
    )

    with pytest.raises(ValueError):
        benchmark_steps(
            model,
            loader,
            optimizer,
        )
