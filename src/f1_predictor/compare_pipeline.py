"""End-to-end pipeline: sync deps, pull data, build features, run every
prediction model, and write a model-comparison table into README.md.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from f1_predictor import data_loader, features
from f1_predictor.last3_quali_binary_classification import (
    run_last3_quali_binary_classification,
)
from f1_predictor.simple_top10_binary_classification import run_binary_classification

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"

# Same season range as data_loader's __main__ -- widen there and here together.
SEASONS = [2022, 2023, 2024, 2025, 2026]

COMPARISON_START = "<!-- MODEL_COMPARISON_START -->"
COMPARISON_END = "<!-- MODEL_COMPARISON_END -->"


def run_build_steps() -> None:
    print("== Build: uv sync ==")
    subprocess.run(["uv", "sync"], cwd=REPO_ROOT, check=True)


def run_data_pull() -> None:
    print("== Data: pulling race + qualifying results ==")
    data_loader.build_raw_dataset(seasons=SEASONS)


def run_feature_build() -> None:
    print("== Features: building feature table ==")
    features.build_feature_table()


def run_predictions() -> dict:
    print("== Predictions: simple_top10 (rolling form) ==")
    _, simple_scores = run_binary_classification()

    print("== Predictions: last3_quali (last-3-race lag) ==")
    _, last3_scores = run_last3_quali_binary_classification()

    return {
        "Rolling form (simple_top10)": simple_scores,
        "Last-3 races + quali (last3_quali)": last3_scores,
    }


def _format_metric(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def build_comparison_table(results: dict) -> str:
    rows = [
        "| Feature set | Model | Accuracy | ROC AUC | Log Loss | Folds |",
        "|---|---|---|---|---|---|",
    ]
    for feature_set, model_scores in results.items():
        for model_name, summary in model_scores.items():
            rows.append(
                f"| {feature_set} | {model_name} | "
                f"{_format_metric(summary['accuracy_mean'], summary['accuracy_std'])} | "
                f"{_format_metric(summary['roc_auc_mean'], summary['roc_auc_std'])} | "
                f"{_format_metric(summary['log_loss_mean'], summary['log_loss_std'])} | "
                f"{summary['n_folds']} |"
            )
    return "\n".join(rows)


def update_readme(results: dict) -> None:
    """Replace the content between the comparison markers in README.md.

    The markers must already exist in README.md -- this only swaps what's
    between them, it never touches the rest of the file.
    """
    table = build_comparison_table(results)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    section = (
        f"{COMPARISON_START}\n"
        f"*Last generated {generated} by `make pipeline`.*\n\n"
        f"{table}\n"
        f"{COMPARISON_END}"
    )

    text = README_PATH.read_text()
    if COMPARISON_START not in text or COMPARISON_END not in text:
        raise RuntimeError(
            f"{README_PATH} is missing the {COMPARISON_START}/{COMPARISON_END} markers"
        )

    before = text.split(COMPARISON_START)[0]
    after = text.split(COMPARISON_END)[1]
    README_PATH.write_text(before + section + after)
    print(f"Updated model comparison table in {README_PATH}")


def main() -> None:
    run_build_steps()
    run_data_pull()
    run_feature_build()
    results = run_predictions()
    update_readme(results)


if __name__ == "__main__":
    main()
