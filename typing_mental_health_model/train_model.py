from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = [
    "avg_dwell_time_ms",
    "avg_flight_time_ms",
    "pause_ratio",
    "backspace_rate",
    "error_rate",
    "correction_latency_ms",
    "words_per_minute",
    "burst_length",
    "session_duration_sec",
    "late_night_ratio",
    "sentiment_polarity",
    "negation_ratio",
    "uncertainty_ratio",
    "first_person_ratio",
    "lexical_diversity",
]

TEXT_COLUMN = "text"
TARGET_COLUMNS = ["stress_pct", "anxiety_pct", "depression_pct"]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bounded_noise(rng: np.random.Generator, scale: float) -> float:
    return float(rng.normal(0.0, scale))


def generate_text(
    rng: np.random.Generator,
    stress: float,
    anxiety: float,
    depression: float,
) -> str:
    neutral_openers = [
        "I completed the main tasks for today and reviewed the next steps.",
        "The workday was manageable and I moved through the checklist steadily.",
        "I sent updates, answered messages, and planned tomorrow's priorities.",
        "I finished the review, documented the results, and cleaned up the notes.",
    ]
    work_closers = [
        "I will revisit the remaining items tomorrow.",
        "The schedule is clear enough for the next session.",
        "I can continue after a short break and another review pass.",
        "The current draft is usable, but it still needs refinement.",
    ]
    stress_phrases = [
        "I am rushing through deadlines and keep switching between tasks.",
        "There is too much to finish at once and the pace feels intense.",
        "I keep correcting myself because I do not want to miss anything important.",
        "Everything feels urgent and I am trying to catch up quickly.",
    ]
    anxiety_phrases = [
        "I keep worrying that something might go wrong even after I check it.",
        "I am uncertain about the outcome and keep second-guessing the details.",
        "My thoughts keep looping around possible mistakes and consequences.",
        "I am uneasy and keep revisiting the same point for reassurance.",
    ]
    depression_phrases = [
        "It is hard to stay motivated and my energy feels low.",
        "I feel mentally drained and focusing takes more effort than usual.",
        "The work feels heavier than it should and I am moving slowly.",
        "I do not feel engaged and it is difficult to sustain momentum.",
    ]

    phrases: List[str] = [rng.choice(neutral_openers)]

    if stress >= 55:
        phrases.append(rng.choice(stress_phrases))
    if anxiety >= 55:
        phrases.append(rng.choice(anxiety_phrases))
    if depression >= 55:
        phrases.append(rng.choice(depression_phrases))

    if len(phrases) == 1:
        phrases.append(rng.choice(work_closers))
    else:
        if rng.random() < 0.65:
            phrases.append(rng.choice(work_closers))

    return " ".join(phrases)


def generate_synthetic_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | str]] = []

    for _ in range(n_samples):
        stress_latent = rng.beta(2.1, 2.0)
        anxiety_latent = clamp(0.55 * stress_latent + 0.45 * rng.beta(2.2, 2.1) + rng.normal(0, 0.05), 0, 1)
        depression_latent = clamp(0.35 * stress_latent + 0.65 * rng.beta(1.9, 2.4) + rng.normal(0, 0.06), 0, 1)

        stress = clamp(100 * stress_latent + bounded_noise(rng, 4), 0, 100)
        anxiety = clamp(100 * anxiety_latent + bounded_noise(rng, 4), 0, 100)
        depression = clamp(100 * depression_latent + bounded_noise(rng, 4), 0, 100)

        avg_dwell_time_ms = clamp(85 + 0.38 * stress + 0.20 * anxiety + bounded_noise(rng, 10), 45, 220)
        avg_flight_time_ms = clamp(95 + 0.32 * stress + 0.25 * depression + bounded_noise(rng, 12), 50, 260)
        pause_ratio = clamp(0.05 + 0.0027 * anxiety + 0.0016 * depression + bounded_noise(rng, 0.03), 0.01, 0.75)
        backspace_rate = clamp(0.015 + 0.0016 * stress + 0.0009 * anxiety + bounded_noise(rng, 0.015), 0.0, 0.35)
        error_rate = clamp(0.012 + 0.0014 * stress + 0.0011 * anxiety + bounded_noise(rng, 0.012), 0.0, 0.30)
        correction_latency_ms = clamp(130 + 0.95 * anxiety + 0.75 * stress + bounded_noise(rng, 18), 60, 420)
        words_per_minute = clamp(78 - 0.16 * stress - 0.22 * depression + bounded_noise(rng, 6), 10, 120)
        burst_length = clamp(11 - 0.038 * depression - 0.018 * anxiety + bounded_noise(rng, 1.5), 1, 20)
        session_duration_sec = clamp(420 + 6.5 * stress + 4.0 * depression + bounded_noise(rng, 95), 60, 2400)
        late_night_ratio = clamp(0.04 + 0.0021 * depression + 0.0014 * stress + bounded_noise(rng, 0.03), 0.0, 0.90)
        sentiment_polarity = clamp(0.72 - 0.009 * stress - 0.013 * depression + bounded_noise(rng, 0.18), -1.0, 1.0)
        negation_ratio = clamp(0.01 + 0.0011 * depression + 0.0009 * anxiety + bounded_noise(rng, 0.01), 0.0, 0.20)
        uncertainty_ratio = clamp(0.01 + 0.0015 * anxiety + bounded_noise(rng, 0.012), 0.0, 0.20)
        first_person_ratio = clamp(0.04 + 0.0012 * depression + bounded_noise(rng, 0.012), 0.0, 0.20)
        lexical_diversity = clamp(0.78 - 0.0017 * depression - 0.0010 * stress + bounded_noise(rng, 0.05), 0.20, 0.95)

        text = generate_text(rng, stress, anxiety, depression)

        rows.append(
            {
                TEXT_COLUMN: text,
                "avg_dwell_time_ms": round(avg_dwell_time_ms, 3),
                "avg_flight_time_ms": round(avg_flight_time_ms, 3),
                "pause_ratio": round(pause_ratio, 5),
                "backspace_rate": round(backspace_rate, 5),
                "error_rate": round(error_rate, 5),
                "correction_latency_ms": round(correction_latency_ms, 3),
                "words_per_minute": round(words_per_minute, 3),
                "burst_length": round(burst_length, 3),
                "session_duration_sec": round(session_duration_sec, 3),
                "late_night_ratio": round(late_night_ratio, 5),
                "sentiment_polarity": round(sentiment_polarity, 5),
                "negation_ratio": round(negation_ratio, 5),
                "uncertainty_ratio": round(uncertainty_ratio, 5),
                "first_person_ratio": round(first_person_ratio, 5),
                "lexical_diversity": round(lexical_diversity, 5),
                "stress_pct": round(stress, 3),
                "anxiety_pct": round(anxiety, 3),
                "depression_pct": round(depression, 3),
            }
        )

    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=1200, ngram_range=(1, 2)), TEXT_COLUMN),
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    model = MultiOutputRegressor(Ridge(alpha=1.5))

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        rmse = float(np.sqrt(mean_squared_error(y_true[target], y_pred[:, idx])))
        r2 = float(r2_score(y_true[target], y_pred[:, idx]))
        metrics[target] = {
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a research prototype for typing-based mental health indicator prediction.")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of synthetic samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Project root directory containing dataset/, models/, and reports/.",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of the synthetic dataset.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    dataset_dir = output_dir / "dataset"
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_dir / "synthetic_typing_mental_health.csv"
    model_path = models_dir / "typing_mental_health_pipeline.joblib"
    metrics_path = reports_dir / "metrics.json"
    preview_path = reports_dir / "test_predictions_preview.csv"

    if args.regenerate_data or not dataset_path.exists():
        df = generate_synthetic_dataset(args.n_samples, args.seed)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    X = df[[TEXT_COLUMN] + NUMERIC_COLUMNS]
    y = df[TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = compute_metrics(y_test.reset_index(drop=True), preds)

    preview = X_test.reset_index(drop=True).copy()
    for idx, target in enumerate(TARGET_COLUMNS):
        preview[f"true_{target}"] = y_test.reset_index(drop=True)[target]
        preview[f"pred_{target}"] = preds[:, idx]
    preview.head(25).to_csv(preview_path, index=False)

    joblib.dump(
        {
            "pipeline": pipeline,
            "numeric_columns": NUMERIC_COLUMNS,
            "text_column": TEXT_COLUMN,
            "target_columns": TARGET_COLUMNS,
        },
        model_path,
    )

    report = {
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "n_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
    }

    metrics_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
