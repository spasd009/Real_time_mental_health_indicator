from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["stress_pct", "anxiety_pct", "depression_pct"]
PRETTY_LABELS = {
    "stress_pct": "Stress",
    "anxiety_pct": "Anxiety",
    "depression_pct": "Depression",
}


def load_government_values(government_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(government_csv)
    cols = set(df.columns)

    # Format A: wide, columns include stress_pct, anxiety_pct, depression_pct
    if all(col in cols for col in TARGET_COLUMNS):
        row = df.iloc[0]
        return {col: float(row[col]) for col in TARGET_COLUMNS}

    # Format B: long, with indicator + government_pct
    if {"indicator", "government_pct"}.issubset(cols):
        mapping = dict(zip(df["indicator"].astype(str), df["government_pct"]))
        missing = [col for col in TARGET_COLUMNS if col not in mapping]
        if missing:
            raise ValueError(
                f"Government CSV is missing indicators in long format: {missing}. "
                "Expected indicators: stress_pct, anxiety_pct, depression_pct."
            )
        return {col: float(mapping[col]) for col in TARGET_COLUMNS}

    raise ValueError(
        "Unsupported government CSV schema. Use either: "
        "(1) wide columns stress_pct/anxiety_pct/depression_pct, or "
        "(2) long columns indicator/government_pct."
    )


def load_processed_means(processed_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(processed_csv)
    missing = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Processed CSV is missing required columns: {missing}")
    means = df[TARGET_COLUMNS].mean(numeric_only=True)
    return {col: float(means[col]) for col in TARGET_COLUMNS}


def build_summary(government: Dict[str, float], processed: Dict[str, float]) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for metric in TARGET_COLUMNS:
        gov_value = government[metric]
        proc_value = processed[metric]
        rows.append(
            {
                "metric": metric,
                "label": PRETTY_LABELS[metric],
                "government_pct": round(gov_value, 4),
                "processed_pct": round(proc_value, 4),
                "difference_pct_points": round(proc_value - gov_value, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, chart_path: Path) -> None:
    labels = summary["label"].tolist()
    government_values = summary["government_pct"].astype(float).to_numpy()
    processed_values = summary["processed_pct"].astype(float).to_numpy()

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.2))
    gov_bars = ax.bar(x - width / 2, government_values, width, label="Government Data")
    proc_bars = ax.bar(x + width / 2, processed_values, width, label="Processed Data")

    ax.set_title("Government vs Processed Data Comparison")
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in (gov_bars, proc_bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare government mental-health percentages against processed model output and generate a bar chart."
    )
    parser.add_argument(
        "--government-csv",
        type=Path,
        default=Path("dataset/government_mental_health_reference.csv"),
        help="Path to government reference CSV.",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("reports/realtime_predictions_log.csv"),
        help="Path to processed prediction CSV.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=Path("reports/government_vs_processed_summary.csv"),
        help="Path to save summary comparison CSV.",
    )
    parser.add_argument(
        "--output-chart",
        type=Path,
        default=Path("reports/government_vs_processed_bar.png"),
        help="Path to save output grouped-bar chart.",
    )
    args = parser.parse_args()

    government = load_government_values(args.government_csv)
    processed = load_processed_means(args.processed_csv)
    summary = build_summary(government, processed)

    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    plot_comparison(summary, args.output_chart)

    print("Comparison complete.")
    print(f"Government source: {args.government_csv}")
    print(f"Processed source: {args.processed_csv}")
    print(f"Summary CSV: {args.output_summary_csv}")
    print(f"Bar chart: {args.output_chart}")
    print("\nSummary table:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

