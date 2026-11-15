"""Generate complete ablation metrics from saved checkpoints."""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import EmbeddingNet
from scripts.run_ablation import evaluate_ablation
from train import load_cifar100, set_seed


EXPERIMENTS = [
    ("dim64", "Embedding dimension", "64D"),
    ("dim128", "Embedding dimension", "128D"),
    ("dim256", "Embedding dimension", "256D"),
    ("margin01", "Triplet margin", "0.1"),
    ("margin05", "Triplet margin", "0.5"),
    ("optimizer_sgd", "Optimizer", "Momentum SGD"),
    ("optimizer_nadam", "Optimizer", "NAdam"),
    ("mining_hard", "Mining strategy", "Batch-hard"),
    ("mining_semi_hard", "Mining strategy", "Semi-hard"),
]

GROUPS = {
    "Embedding dimension": ["dim64", "dim128", "dim256"],
    "Triplet margin": ["margin01", "dim128", "margin05"],
    "Optimizer": [
        "optimizer_sgd", "dim128", "optimizer_nadam"
    ],
    "Mining strategy": [
        "dim128", "mining_hard", "mining_semi_hard"
    ],
}

FIELDNAMES = [
    "experiment", "category", "variant",
    "epochs_completed", "checkpoint_epoch",
    "train_images_last_epoch", "train_loss",
    "embedding_dim", "margin", "optimizer", "mining",
    "learning_rate", "batch_size",
    "recall_at_1", "recall_at_5", "recall_at_10",
    "hit_at_1", "hit_at_5", "hit_at_10",
    "map", "embedding_spread", "spread_ratio",
    "collapsed", "checkpoint_used",
]


def choose_checkpoint(directory):
    directory = Path(directory)
    for filename in ("best.pt", "last.pt"):
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No checkpoint in {directory}"
    )


def read_history(path):
    with Path(path).open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty history: {path}")
    return rows


def checkpoint_history_row(history, epoch):
    for row in history:
        if int(row["epoch"]) == int(epoch):
            return row
    raise ValueError(
        f"Missing history for epoch {epoch}"
    )


def load_checkpoint_model(path):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )
    dimension = int(
        checkpoint["config"]["embedding_dim"]
    )
    if dimension not in (64, 128, 256):
        raise ValueError(
            f"Unexpected dimension: {dimension}"
        )
    model = EmbeddingNet(embedding_dim=dimension)
    model.load_state_dict(
        checkpoint["model_state"], strict=True
    )
    model.eval()
    return model, checkpoint


def create_summary_row(
    name, category, variant, history,
    checkpoint, checkpoint_path, retrieval
):
    config = checkpoint["config"]
    epoch = int(checkpoint["epoch"])
    record = checkpoint_history_row(history, epoch)
    collapsed = any(
        str(item["collapsed"]).lower() == "true"
        for item in history
    )
    if collapsed and Path(checkpoint_path).name == "best.pt":
        raise ValueError(
            "Collapsed experiment has a best checkpoint"
        )

    return {
        "experiment": name,
        "category": category,
        "variant": variant,
        "epochs_completed": len(history),
        "checkpoint_epoch": epoch,
        "train_images_last_epoch": int(
            record["train_images"]
        ),
        "train_loss": float(record["train_loss"]),
        "embedding_dim": int(config["embedding_dim"]),
        "margin": float(config["margin"]),
        "optimizer": config["optimizer"],
        "mining": config["mining"],
        "learning_rate": float(config["lr"]),
        "batch_size": (
            int(config["classes_per_batch"])
            * int(config["samples_per_class"])
        ),
        "recall_at_1": float(retrieval["recall_at_1"]),
        "recall_at_5": float(retrieval["recall_at_5"]),
        "recall_at_10": float(retrieval["recall_at_10"]),
        "hit_at_1": float(retrieval["hit_at_1"]),
        "hit_at_5": float(retrieval["hit_at_5"]),
        "hit_at_10": float(retrieval["hit_at_10"]),
        "map": float(retrieval["map"]),
        "embedding_spread": float(retrieval["spread"]),
        "spread_ratio": float(record["spread_ratio"]),
        "collapsed": collapsed,
        "checkpoint_used": str(checkpoint_path),
    }


def validate_retrieval(row, recorded):
    pairs = [
        ("map", "val_map"),
        ("hit_at_1", "val_hit_at_1"),
        ("hit_at_5", "val_hit_at_5"),
        ("hit_at_10", "val_hit_at_10"),
    ]
    for actual, expected in pairs:
        if not math.isclose(
            float(row[actual]),
            float(recorded[expected]),
            rel_tol=0,
            abs_tol=0.0001,
        ):
            raise ValueError(
                f"{row['experiment']} mismatch: {actual}. "
                "Check checkpoint, dataset and query selection."
            )


def validate_summary(rows):
    expected = [name for name, _, _ in EXPERIMENTS]
    if [row["experiment"] for row in rows] != expected:
        raise ValueError(
            "Missing or incorrectly ordered experiments"
        )

    for row in rows:
        for metric in (
            "recall_at_1", "recall_at_5", "recall_at_10",
            "hit_at_1", "hit_at_5", "hit_at_10", "map"
        ):
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Invalid {metric}: {row['experiment']}"
                )
        if not (
            row["recall_at_1"]
            <= row["recall_at_5"]
            <= row["recall_at_10"]
        ):
            raise ValueError("Recall@K is not monotonic")
        if not (
            row["hit_at_1"]
            <= row["hit_at_5"]
            <= row["hit_at_10"]
        ):
            raise ValueError("Hit@K is not monotonic")


def percent(value):
    return f"{100 * float(value):.3f}%"


def build_report(rows):
    validate_summary(rows)
    indexed = {row["experiment"]: row for row in rows}
    control = indexed["dim128"]

    lines = [
        "# Visual Similarity Search - Ablation Report",
        "",
        "## Evaluation protocol",
        "",
        "- Dataset: CIFAR-100, stratified seed-42 split.",
        "- Training: 48,000 images per complete epoch.",
        "- Validation: 6,000 gallery images, 500 queries.",
        "- Model: ResNet-18 trained from scratch.",
        "- Baseline: 128D, margin 0.3, custom Adam, random mining.",
        "- Batch size: 64 (16 classes x 4 examples).",
        "- Learning rate: 0.0001.",
        "- Standard budget: three training epochs.",
        "- Checkpoints selected by validation mAP.",
        "- Retrieval uses brute-force Euclidean distance.",
        "- Query self-matches are excluded.",
        "",
        "Recall@K is the fraction of relevant gallery images "
        "retrieved within the first K results. "
        "Hit@K is the fraction of queries with at least "
        "one relevant image in the top K. "
        "mAP uses the complete gallery ranking.",
        "",
    ]

    for group, names in GROUPS.items():
        lines += [
            f"## {group}", "",
            "| Configuration | Epochs | Recall@1 | Recall@5 | "
            "Recall@10 | Hit@1 | Hit@5 | Hit@10 | mAP | Collapsed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name in names:
            row = indexed[name]
            lines.append(
                f"| {row['variant']} "
                f"| {row['epochs_completed']} "
                f"| {percent(row['recall_at_1'])} "
                f"| {percent(row['recall_at_5'])} "
                f"| {percent(row['recall_at_10'])} "
                f"| {percent(row['hit_at_1'])} "
                f"| {percent(row['hit_at_5'])} "
                f"| {percent(row['hit_at_10'])} "
                f"| {percent(row['map'])} "
                f"| {'Yes' if row['collapsed'] else 'No'} |"
            )

        lines += ["", "**Findings:**", ""]
        for name in names:
            if name == "dim128":
                continue
            row = indexed[name]
            if row["collapsed"]:
                lines.append(
                    f"- {row['variant']} collapsed after "
                    f"{row['epochs_completed']} epoch(s); "
                    "this is not a matched three-epoch result."
                )
            else:
                delta = 100 * (row["map"] - control["map"])
                lines.append(
                    f"- {row['variant']}: "
                    f"{percent(row['map'])} mAP, "
                    f"{delta:+.3f} percentage points "
                    "relative to the 128D control."
                )
        lines.append("")

    lines += [
        "## Collapse diagnostic", "",
        "| Configuration | Spread | Spread ratio | Collapsed |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} "
            f"| {row['embedding_spread']:.6f} "
            f"| {row['spread_ratio']:.3f} "
            f"| {'Yes' if row['collapsed'] else 'No'} |"
        )

    lines += [
        "",
        "## Independent 30-epoch baseline test results",
        "",
        "The main model used 128D embeddings, margin 0.3, "
        "custom Adam and random mining.",
        "",
        "| Test metric | Result |",
        "|---|---:|",
        "| Recall@1 | 0.481% |",
        "| Recall@5 | 2.193% |",
        "| Recall@10 | 3.997% |",
        "| Hit@1 | 28.400% |",
        "| Hit@5 | 57.000% |",
        "| Hit@10 | 67.400% |",
        "| mAP | 15.525% |",
        "| Random-triplet ordering accuracy | 86.980% |",
        "",
        "The held-out test numbers were measured separately "
        "and are not recalculated by this report generator.",
        "",
        "## Limitations", "",
        "1. Ablations used one seed and only three epochs.",
        "2. Optimizer learning rates were not individually tuned.",
        "3. Short-run winners were not tested together.",
        "4. Batch-hard mining collapsed after its first epoch.",
        "5. The final 30-epoch model used random mining, "
        "not batch-hard mining.",
        "6. Random-triplet ordering accuracy is not the same "
        "as hard-negative ordering accuracy.",
        "",
        "## Conclusion", "",
        "Semi-hard mining had the strongest short-run mining "
        "mAP; batch-hard mining collapsed. The stable "
        "30-epoch random-mining baseline achieved "
        "15.525% held-out test mAP.",
        "",
    ]
    return "\n".join(lines)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def save_markdown(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def evaluate_all(ablation_directory, threads=8, eval_batch_size=64):
    torch.set_num_threads(threads)
    set_seed(42)

    _, val_loader, _ = load_cifar100(
        seed=42,
        classes_per_batch=16,
        samples_per_class=4,
    )
    if len(val_loader.dataset) != 6000:
        raise RuntimeError("Validation dataset is not 6,000 images")

    gallery_loader = DataLoader(
        val_loader.dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    for name, category, variant in EXPERIMENTS:
        directory = Path(ablation_directory) / name
        path = choose_checkpoint(directory)
        history = read_history(directory / "metrics.csv")
        model, checkpoint = load_checkpoint_model(path)
        config = checkpoint["config"]

        print(f"Evaluating {name}...", flush=True)
        with torch.inference_mode():
            retrieval = evaluate_ablation(
                model,
                gallery_loader,
                dimension=int(config["embedding_dim"]),
                queries_per_class=int(
                    config.get("queries_per_class", 5)
                ),
                search_batch_size=32,
                seed=int(config.get("seed", 42)),
            )

        row = create_summary_row(
            name, category, variant,
            history, checkpoint, path, retrieval
        )
        recorded = checkpoint_history_row(
            history, checkpoint["epoch"]
        )
        validate_retrieval(row, recorded)
        rows.append(row)

        print(
            f"  Recall@1: {percent(row['recall_at_1'])}, "
            f"Recall@5: {percent(row['recall_at_5'])}, "
            f"mAP: {percent(row['map'])}",
            flush=True,
        )
        del model

    validate_summary(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-dir", type=Path,
        default=Path("checkpoints/ablations")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results")
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.threads < 1 or args.eval_batch_size < 1:
        parser.error("Threads and batch size must be positive")

    rows = evaluate_all(
        args.ablation_dir,
        threads=args.threads,
        eval_batch_size=args.eval_batch_size,
    )
    report = build_report(rows)

    csv_path = args.output_dir / "ablation_summary.csv"
    md_path = args.output_dir / "ablation_report.md"

    save_csv(csv_path, rows)
    save_markdown(md_path, report)

    print("Report completed.")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
