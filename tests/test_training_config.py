
import pytest
import torch

from train import (
    build_parser,
    make_config,
    triplet_metrics,
    validate_args,
)


def test_default_training_configuration():
    parser = build_parser()
    args = parser.parse_args([])

    validate_args(args)

    assert args.threads == 8
    assert args.classes_per_batch == 16
    assert args.samples_per_class == 4
    assert args.embedding_dim == 128
    assert args.max_batches == 100
    assert args.val_batches == 20


def test_custom_training_configuration():
    parser = build_parser()

    args = parser.parse_args([
        "--epochs", "4",
        "--threads", "4",
        "--classes-per-batch", "8",
        "--max-batches", "50",
        "--val-batches", "10",
        "--lr", "0.0005",
    ])

    validate_args(args)
    config = make_config(args)

    assert args.epochs == 4
    assert args.threads == 4
    assert config["classes_per_batch"] == 8
    assert config["max_batches"] == 50
    assert config["val_batches"] == 10
    assert config["lr"] == 0.0005


def test_invalid_training_configuration():
    parser = build_parser()

    invalid_options = [
        ["--epochs", "0"],
        ["--threads", "0"],
        ["--lr", "-0.1"],
        ["--margin", "-0.3"],
        ["--classes-per-batch", "1"],
        ["--samples-per-class", "7"],
        ["--max-batches", "-1"],
        ["--val-batches", "-1"],
        ["--log-every", "0"],
    ]

    for options in invalid_options:
        args = parser.parse_args(options)

        with pytest.raises(ValueError):
            validate_args(args)


def test_triplet_metrics_values():
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [2.0, 0.0],
        [2.1, 0.0],
    ])

    labels = torch.tensor([0, 0, 1, 1])

    metrics = triplet_metrics(
        embeddings,
        labels,
        margin=0.3,
    )

    assert metrics["ordering_accuracy"] == 1.0
    assert metrics["margin_accuracy"] == 1.0
    assert metrics["embedding_spread"] > 0


def test_triplet_metrics_detects_collapse():
    embeddings = torch.zeros(4, 3)
    labels = torch.tensor([0, 0, 1, 1])

    metrics = triplet_metrics(
        embeddings,
        labels,
        margin=0.3,
    )

    assert metrics["ordering_accuracy"] == 0.0
    assert metrics["margin_accuracy"] == 0.0
    assert metrics["embedding_spread"] == 0.0


def test_triplet_metrics_invalid_batch():
    embeddings = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 0, 0])

    with pytest.raises(ValueError):
        triplet_metrics(
            embeddings,
            labels,
        )
