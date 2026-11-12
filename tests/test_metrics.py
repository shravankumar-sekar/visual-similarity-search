
import csv

from metrics import METRIC_FIELDS, save_metrics


def make_row(epoch):
    return {
        "epoch": epoch,
        "train_loss": 0.4,
        "train_ordering": 0.1,
        "train_margin": 0.0,
        "val_loss": 0.35,
        "val_ordering": 0.2,
        "val_margin": 0.0,
    }


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def test_save_metrics(tmp_path):
    path = tmp_path / "metrics.csv"
    history = [make_row(1)]

    save_metrics(history, path)

    rows = read_rows(path)

    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert float(rows[0]["train_loss"]) == 0.4


def test_resume_without_duplicate_rows(tmp_path):
    path = tmp_path / "metrics.csv"

    save_metrics([make_row(1)], path)
    save_metrics([make_row(1), make_row(2)], path)

    rows = read_rows(path)

    assert len(rows) == 2
    assert [row["epoch"] for row in rows] == ["1", "2"]


def test_metrics_column_names(tmp_path):
    path = tmp_path / "metrics.csv"

    save_metrics([make_row(1)], path)

    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == METRIC_FIELDS


def test_creates_output_directory(tmp_path):
    path = tmp_path / "results" / "experiment" / "metrics.csv"

    save_metrics([make_row(1)], path)

    assert path.exists()
