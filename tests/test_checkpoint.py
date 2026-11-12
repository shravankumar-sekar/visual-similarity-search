
import random

import numpy as np
import pytest
import torch

from train import save_checkpoint, load_checkpoint
from optimizers import Adam


def test_checkpoint_restore(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=0.01)

    x = torch.tensor([[1.0, 2.0]])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    saved_weights = [
        p.detach().clone() for p in model.parameters()
    ]

    path = tmp_path / "checkpoint.pt"
    history = [{"epoch": 1, "train_loss": 0.5}]

    save_checkpoint(
        path, model, optimizer, epoch=1, history=history
    )

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    epoch, restored_history = load_checkpoint(
        path, model, optimizer
    )

    assert epoch == 1
    assert restored_history == history

    for p, expected in zip(
        model.parameters(), saved_weights
    ):
        assert torch.allclose(p, expected)

    assert optimizer.steps == [1, 1]


def test_random_states_restored(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters())

    path = tmp_path / "random_state.pt"

    save_checkpoint(
        path, model, optimizer, epoch=1, history=[]
    )

    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)

    random.random()
    np.random.rand()
    torch.rand(1)

    load_checkpoint(path, model, optimizer)

    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.allclose(
        torch.rand(1), expected_torch
    )


def test_checkpoint_configuration_mismatch(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters())

    path = tmp_path / "config_test.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=[],
        config={"embedding_dim": 128},
    )

    with pytest.raises(
        ValueError, match="configuration does not match"
    ):
        load_checkpoint(
            path,
            model,
            optimizer,
            config={"embedding_dim": 256},
        )


def test_checkpoint_matching_configuration(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters())

    path = tmp_path / "matching_config.pt"
    config = {
        "embedding_dim": 128,
        "optimizer": "adam",
        "lr": 0.001,
    }

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=3,
        history=[{"epoch": 3}],
        config=config,
    )

    epoch, history = load_checkpoint(
        path,
        model,
        optimizer,
        config=config,
    )

    assert epoch == 3
    assert history == [{"epoch": 3}]


def test_checkpoint_file_created(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = Adam(model.parameters())

    path = tmp_path / "nested" / "model.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        history=[],
    )

    assert path.exists()
    assert path.stat().st_size > 0
    assert not path.with_suffix(".pt.tmp").exists()
