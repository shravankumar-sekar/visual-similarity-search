
import csv
from pathlib import Path


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_ordering",
    "train_margin",
    "val_loss",
    "val_ordering",
    "val_margin",
]


def save_metrics(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDS,
        )
        writer.writeheader()
        writer.writerows(history)

    temp_path.replace(path)
