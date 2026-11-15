
"""Tests for consolidated ablation reporting."""

import csv

import pytest

from scripts.generate_ablation_report import (
    EXPERIMENTS,
    build_report,
    checkpoint_history_row,
    choose_checkpoint,
    create_summary_row,
    read_history,
    validate_retrieval,
    validate_summary,
)


def sample_rows():
    rows = []

    for name, category, variant in EXPERIMENTS:
        collapsed = name == "mining_hard"

        rows.append({
            "experiment": name,
            "category": category,
            "variant": variant,
            "epochs_completed": 1 if collapsed else 3,
            "checkpoint_epoch": 1 if collapsed else 3,
            "train_images_last_epoch": 48000,
            "train_loss": 0.2,
            "embedding_dim": 128,
            "margin": 0.3,
            "optimizer": "adam",
            "mining": "hard" if collapsed else "random",
            "learning_rate": 0.0001,
            "batch_size": 64,
            "recall_at_1": 0.001,
            "recall_at_5": 0.005,
            "recall_at_10": 0.010,
            "hit_at_1": 0.10,
            "hit_at_5": 0.30,
            "hit_at_10": 0.40,
            "map": 0.045,
            "embedding_spread": 0.5,
            "spread_ratio": 4.0,
            "collapsed": collapsed,
            "checkpoint_used": "best.pt",
        })

    return rows


def test_select_best_checkpoint(tmp_path):
    (tmp_path / "best.pt").touch()
    (tmp_path / "last.pt").touch()

    assert choose_checkpoint(tmp_path).name == "best.pt"


def test_fallback_to_last(tmp_path):
    (tmp_path / "last.pt").touch()

    assert choose_checkpoint(tmp_path).name == "last.pt"


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        choose_checkpoint(tmp_path)


def test_read_history(tmp_path):
    path = tmp_path / "metrics.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "val_map"],
        )
        writer.writeheader()
        writer.writerow({
            "epoch": 1,
            "val_map": 0.04,
        })

    rows = read_history(path)

    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"


def test_find_checkpoint_epoch():
    history = [
        {"epoch": "1"},
        {"epoch": "2"},
        {"epoch": "3"},
    ]

    assert checkpoint_history_row(
        history, 2
    )["epoch"] == "2"

    with pytest.raises(ValueError):
        checkpoint_history_row(history, 4)


def test_summary_row_with_actual_recall():
    checkpoint = {
        "epoch": 3,
        "config": {
            "embedding_dim": 128,
            "margin": 0.3,
            "optimizer": "adam",
            "mining": "random",
            "lr": 0.0001,
            "classes_per_batch": 16,
            "samples_per_class": 4,
        },
    }

    history = [{
        "epoch": "3",
        "train_images": "48000",
        "train_loss": "0.2",
        "spread_ratio": "4.0",
        "collapsed": "False",
    }]

    retrieval = {
        "recall_at_1": 0.001,
        "recall_at_5": 0.005,
        "recall_at_10": 0.01,
        "hit_at_1": 0.1,
        "hit_at_5": 0.3,
        "hit_at_10": 0.4,
        "map": 0.045,
        "spread": 0.55,
    }

    row = create_summary_row(
        "dim128",
        "Embedding dimension",
        "128D",
        history,
        checkpoint,
        "best.pt",
        retrieval,
    )

    assert row["batch_size"] == 64
    assert row["epochs_completed"] == 1
    assert row["recall_at_5"] == 0.005
    assert row["map"] == 0.045


def test_validation_detects_metric_mismatch():
    row = {
        "experiment": "dim128",
        "map": 0.04,
        "hit_at_1": 0.1,
        "hit_at_5": 0.3,
        "hit_at_10": 0.4,
    }

    recorded = {
        "val_map": "0.08",
        "val_hit_at_1": "0.1",
        "val_hit_at_5": "0.3",
        "val_hit_at_10": "0.4",
    }

    with pytest.raises(ValueError):
        validate_retrieval(row, recorded)


def test_all_experiments_required():
    rows = sample_rows()

    validate_summary(rows)

    with pytest.raises(ValueError):
        validate_summary(rows[:-1])


def test_recall_must_be_monotonic():
    rows = sample_rows()

    rows[0]["recall_at_5"] = 0.0001

    with pytest.raises(ValueError):
        validate_summary(rows)


def test_report_contains_recall():
    report = build_report(sample_rows())

    assert "Recall@1" in report
    assert "Recall@5" in report
    assert "Recall@10" in report
    assert "Batch-hard" in report
    assert "collapsed" in report
    assert "15.525%" in report
