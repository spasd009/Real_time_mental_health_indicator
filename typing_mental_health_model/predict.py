from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]


DEFAULT_RECORD: Dict[str, Any] = {
    "text": "I have been switching between tasks all day and I keep checking the same details because I am worried I missed something.",
    "avg_dwell_time_ms": 118.0,
    "avg_flight_time_ms": 131.0,
    "pause_ratio": 0.23,
    "backspace_rate": 0.12,
    "error_rate": 0.10,
    "correction_latency_ms": 228.0,
    "words_per_minute": 46.0,
    "burst_length": 6.0,
    "session_duration_sec": 980.0,
    "late_night_ratio": 0.28,
    "sentiment_polarity": -0.18,
    "negation_ratio": 0.06,
    "uncertainty_ratio": 0.09,
    "first_person_ratio": 0.07,
    "lexical_diversity": 0.59,
}

TARGET_DISPLAY_NAMES = {
    "stress_pct": "Stress",
    "anxiety_pct": "Anxiety",
    "depression_pct": "Depression",
}

DEFAULT_COMPARISON_RECORDS: list[Dict[str, Any]] = [
    {
        "label": "baseline",
        **DEFAULT_RECORD,
    },
    {
        "label": "high_stress_night",
        "text": "I am anxious and overwhelmed, I keep fixing errors and I cannot focus.",
        "avg_dwell_time_ms": 146.0,
        "avg_flight_time_ms": 170.0,
        "pause_ratio": 0.36,
        "backspace_rate": 0.19,
        "error_rate": 0.16,
        "correction_latency_ms": 312.0,
        "words_per_minute": 33.0,
        "burst_length": 3.6,
        "session_duration_sec": 1220.0,
        "late_night_ratio": 0.88,
        "sentiment_polarity": -0.42,
        "negation_ratio": 0.12,
        "uncertainty_ratio": 0.15,
        "first_person_ratio": 0.10,
        "lexical_diversity": 0.54,
    },
    {
        "label": "moderate_fatigue",
        "text": "I feel tired today and maybe slower, but I can still complete the work.",
        "avg_dwell_time_ms": 132.0,
        "avg_flight_time_ms": 148.0,
        "pause_ratio": 0.27,
        "backspace_rate": 0.14,
        "error_rate": 0.12,
        "correction_latency_ms": 260.0,
        "words_per_minute": 40.0,
        "burst_length": 4.8,
        "session_duration_sec": 1080.0,
        "late_night_ratio": 0.38,
        "sentiment_polarity": -0.19,
        "negation_ratio": 0.08,
        "uncertainty_ratio": 0.10,
        "first_person_ratio": 0.09,
        "lexical_diversity": 0.58,
    },
    {
        "label": "calm_productive",
        "text": "I feel clear and focused, progress is good and tasks are moving well.",
        "avg_dwell_time_ms": 98.0,
        "avg_flight_time_ms": 110.0,
        "pause_ratio": 0.12,
        "backspace_rate": 0.06,
        "error_rate": 0.05,
        "correction_latency_ms": 170.0,
        "words_per_minute": 58.0,
        "burst_length": 8.4,
        "session_duration_sec": 860.0,
        "late_night_ratio": 0.07,
        "sentiment_polarity": 0.31,
        "negation_ratio": 0.02,
        "uncertainty_ratio": 0.03,
        "first_person_ratio": 0.06,
        "lexical_diversity": 0.64,
    },
    {
        "label": "recovering",
        "text": "I was worried earlier but now I am better and getting back on track.",
        "avg_dwell_time_ms": 110.0,
        "avg_flight_time_ms": 124.0,
        "pause_ratio": 0.18,
        "backspace_rate": 0.09,
        "error_rate": 0.08,
        "correction_latency_ms": 205.0,
        "words_per_minute": 50.0,
        "burst_length": 6.9,
        "session_duration_sec": 920.0,
        "late_night_ratio": 0.16,
        "sentiment_polarity": 0.07,
        "negation_ratio": 0.05,
        "uncertainty_ratio": 0.06,
        "first_person_ratio": 0.07,
        "lexical_diversity": 0.62,
    },
]

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "nothing",
    "neither",
    "nor",
    "cannot",
    "can't",
    "dont",
    "don't",
    "won't",
    "isn't",
    "wasn't",
    "didn't",
    "shouldn't",
    "couldn't",
    "wouldn't",
}

UNCERTAINTY_WORDS = {
    "maybe",
    "perhaps",
    "might",
    "uncertain",
    "unsure",
    "probably",
    "possibly",
    "guess",
    "seems",
    "seem",
    "appears",
    "appear",
    "likely",
}

FIRST_PERSON_WORDS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "calm",
    "okay",
    "fine",
    "stable",
    "confident",
    "productive",
    "clear",
    "improving",
    "hopeful",
    "better",
    "focused",
    "successful",
}

NEGATIVE_WORDS = {
    "bad",
    "worse",
    "worried",
    "anxious",
    "stressed",
    "drained",
    "tired",
    "overwhelmed",
    "sad",
    "stuck",
    "afraid",
    "panic",
    "frustrated",
    "exhausted",
}


@dataclass
class TypingCapture:
    text: str
    start_ts: float
    end_ts: float
    event_times: list[float]
    event_is_character: list[bool]
    backspace_count: int
    correction_latencies_sec: list[float]


def load_record_from_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("JSON input must be a single object representing one typing session.")
    return data


def load_records_from_json(path: Path) -> list[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        if not data:
            raise ValueError("JSON input list is empty. Provide at least one typing-session object.")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON input list must contain only objects.")
        return data
    raise ValueError("JSON input must be a single object or a list of objects.")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def compute_text_features(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return {
            "sentiment_polarity": 0.0,
            "negation_ratio": 0.0,
            "uncertainty_ratio": 0.0,
            "first_person_ratio": 0.0,
            "lexical_diversity": 0.0,
        }

    pos_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
    neg_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    negation_count = sum(1 for token in tokens if token in NEGATION_WORDS)
    uncertainty_count = sum(1 for token in tokens if token in UNCERTAINTY_WORDS)
    first_person_count = sum(1 for token in tokens if token in FIRST_PERSON_WORDS)

    sentiment_polarity = clamp((pos_count - neg_count) / n_tokens, -1.0, 1.0)
    negation_ratio = clamp(negation_count / n_tokens, 0.0, 1.0)
    uncertainty_ratio = clamp(uncertainty_count / n_tokens, 0.0, 1.0)
    first_person_ratio = clamp(first_person_count / n_tokens, 0.0, 1.0)
    lexical_diversity = clamp(len(set(tokens)) / n_tokens, 0.0, 1.0)

    return {
        "sentiment_polarity": sentiment_polarity,
        "negation_ratio": negation_ratio,
        "uncertainty_ratio": uncertainty_ratio,
        "first_person_ratio": first_person_ratio,
        "lexical_diversity": lexical_diversity,
    }


def build_record_from_capture(
    capture: TypingCapture,
    pause_threshold_sec: float,
    base_record: Dict[str, Any],
) -> Dict[str, Any]:
    record = dict(base_record)
    text = capture.text
    record["text"] = text

    duration_sec = max(capture.end_ts - capture.start_ts, 0.1)
    intervals = [
        capture.event_times[idx] - capture.event_times[idx - 1]
        for idx in range(1, len(capture.event_times))
    ]

    avg_flight_ms = (
        mean(intervals) * 1000.0
        if intervals
        else float(record.get("avg_flight_time_ms", DEFAULT_RECORD["avg_flight_time_ms"]))
    )

    # We only capture key press timing, so dwell is estimated from flight.
    avg_dwell_ms = clamp(0.82 * avg_flight_ms + 18.0, 45.0, 220.0)
    pause_time_sec = sum(max(0.0, interval - pause_threshold_sec) for interval in intervals if interval > pause_threshold_sec)
    pause_ratio = clamp(pause_time_sec / duration_sec, 0.0, 0.95)

    total_events = max(len(capture.event_times), 1)
    backspace_rate = clamp(capture.backspace_count / total_events, 0.0, 1.0)
    error_rate = clamp(backspace_rate * 0.85, 0.0, 0.3)

    if capture.correction_latencies_sec:
        correction_latency_ms = mean(capture.correction_latencies_sec) * 1000.0
    else:
        correction_latency_ms = clamp(avg_flight_ms * 1.8, 60.0, 420.0)

    words = re.findall(r"\S+", text.strip())
    word_count = len(words)
    words_per_minute = clamp(word_count / (duration_sec / 60.0), 0.0, 180.0)

    bursts: list[int] = []
    current_burst = 0
    for idx, is_char in enumerate(capture.event_is_character):
        if not is_char:
            continue
        current_burst += 1
        if idx < len(capture.event_times) - 1:
            gap = capture.event_times[idx + 1] - capture.event_times[idx]
            if gap > pause_threshold_sec:
                bursts.append(current_burst)
                current_burst = 0
    if current_burst > 0:
        bursts.append(current_burst)

    avg_char_burst = mean(bursts) if bursts else max(word_count * 5.0, 1.0)
    burst_length = clamp(avg_char_burst / 5.0, 1.0, 20.0)

    local_hour = datetime.now().hour
    late_night_ratio = 0.85 if local_hour >= 23 or local_hour <= 5 else 0.05

    text_features = compute_text_features(text)

    record.update(
        {
            "avg_dwell_time_ms": round(avg_dwell_ms, 5),
            "avg_flight_time_ms": round(clamp(avg_flight_ms, 50.0, 260.0), 5),
            "pause_ratio": round(pause_ratio, 5),
            "backspace_rate": round(backspace_rate, 5),
            "error_rate": round(error_rate, 5),
            "correction_latency_ms": round(clamp(correction_latency_ms, 60.0, 420.0), 5),
            "words_per_minute": round(words_per_minute, 5),
            "burst_length": round(burst_length, 5),
            "session_duration_sec": round(duration_sec, 5),
            "late_night_ratio": round(late_night_ratio, 5),
            **{key: round(value, 5) for key, value in text_features.items()},
        }
    )

    return record


def capture_typing_session() -> TypingCapture:
    if msvcrt is None:
        raise RuntimeError("Live typing mode is only supported on Windows terminals.")
    if not sys.stdin.isatty():
        raise RuntimeError("Live typing mode requires an interactive terminal (TTY).")

    buffer: list[str] = []
    event_times: list[float] = []
    event_is_character: list[bool] = []
    backspace_count = 0
    correction_latencies_sec: list[float] = []
    pending_correction_since: float | None = None
    start_ts = time.time()

    while True:
        ch = msvcrt.getwch()
        ts = time.time()

        if ch in {"\x00", "\xe0"}:
            _ = msvcrt.getwch()
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch in {"\r", "\n"}:
            print()
            end_ts = ts
            break

        event_times.append(ts)

        if ch in {"\b", "\x7f"}:
            event_is_character.append(False)
            backspace_count += 1
            pending_correction_since = ts
            if buffer:
                buffer.pop()
                print("\b \b", end="", flush=True)
            continue

        if ch.isprintable():
            event_is_character.append(True)
            buffer.append(ch)
            if pending_correction_since is not None:
                correction_latencies_sec.append(ts - pending_correction_since)
                pending_correction_since = None
            print(ch, end="", flush=True)
        else:
            event_is_character.append(False)

    if not event_times:
        event_times = [start_ts, end_ts]
        event_is_character = [False, False]

    return TypingCapture(
        text="".join(buffer).strip(),
        start_ts=start_ts,
        end_ts=end_ts,
        event_times=event_times,
        event_is_character=event_is_character,
        backspace_count=backspace_count,
        correction_latencies_sec=correction_latencies_sec,
    )


def predict_from_record(
    pipeline: Any,
    text_column: str,
    numeric_columns: list[str],
    target_columns: list[str],
    record: Dict[str, Any],
) -> Dict[str, float]:
    required = [text_column] + numeric_columns
    missing = [column for column in required if column not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    features = pd.DataFrame([record])[required]
    prediction = pipeline.predict(features)[0]
    return {target: round(float(value), 3) for target, value in zip(target_columns, prediction)}


def predict_records(
    pipeline: Any,
    text_column: str,
    numeric_columns: list[str],
    target_columns: list[str],
    records: list[Dict[str, Any]],
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        label_value = record.get("label")
        if isinstance(label_value, str) and label_value.strip():
            label = label_value.strip()
        else:
            label = f"record_{index}"

        prediction = predict_from_record(
            pipeline=pipeline,
            text_column=text_column,
            numeric_columns=numeric_columns,
            target_columns=target_columns,
            record=record,
        )
        rows.append({"record_label": label, **prediction})

    return pd.DataFrame(rows)


def plot_prediction_comparison(
    comparison_df: pd.DataFrame,
    target_columns: list[str],
    chart_path: Path,
) -> None:
    labels = comparison_df["record_label"].astype(str).tolist()
    x = np.arange(len(labels))
    bar_width = 0.24

    figure_width = max(10.0, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(figure_width, 5.4))

    for idx, target in enumerate(target_columns):
        values = comparison_df[target].astype(float).to_numpy()
        offset = (idx - (len(target_columns) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=TARGET_DISPLAY_NAMES.get(target, target))

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Typing Session Mental-Health Indicator Comparison")
    ax.set_ylabel("Predicted Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)


def append_prediction_log(
    log_csv_path: Path,
    mode: str,
    record: Dict[str, Any],
    result: Dict[str, float],
) -> None:
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        **record,
        **result,
    }
    should_write_header = not log_csv_path.exists() or log_csv_path.stat().st_size == 0
    pd.DataFrame([payload]).to_csv(log_csv_path, mode="a", header=should_write_header, index=False)


def run_realtime_loop(
    pipeline: Any,
    text_column: str,
    numeric_columns: list[str],
    target_columns: list[str],
    base_record: Dict[str, Any],
    log_csv_path: Path | None = None,
) -> None:
    print("Realtime mode enabled. Enter one JSON object per line. Type 'exit' to stop.")
    print("Each JSON line can include partial fields if --input-json was supplied as defaults.")

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            break

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"Invalid JSON: {exc.msg}"}))
            continue

        if not isinstance(parsed, dict):
            print(json.dumps({"error": "Input must be a JSON object."}))
            continue

        merged_record = {**base_record, **parsed}
        try:
            result = predict_from_record(
                pipeline=pipeline,
                text_column=text_column,
                numeric_columns=numeric_columns,
                target_columns=target_columns,
                record=merged_record,
            )
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}))
            continue

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        print(json.dumps(payload))
        if log_csv_path is not None:
            append_prediction_log(
                log_csv_path=log_csv_path,
                mode="realtime_json",
                record=merged_record,
                result=result,
            )


def run_live_typing_loop(
    pipeline: Any,
    text_column: str,
    numeric_columns: list[str],
    target_columns: list[str],
    base_record: Dict[str, Any],
    pause_threshold_sec: float,
    log_csv_path: Path | None = None,
) -> None:
    print("Live typing mode enabled. Type a line and press Enter for instant analysis.")
    print("Type /exit and press Enter to stop.")

    defaults = {**DEFAULT_RECORD, **base_record}

    while True:
        print("> ", end="", flush=True)
        capture = capture_typing_session()

        if not capture.text:
            continue
        if capture.text.strip().lower() in {"exit", "quit", "/exit"}:
            break

        record = build_record_from_capture(
            capture=capture,
            pause_threshold_sec=pause_threshold_sec,
            base_record=defaults,
        )

        result = predict_from_record(
            pipeline=pipeline,
            text_column=text_column,
            numeric_columns=numeric_columns,
            target_columns=target_columns,
            record=record,
        )

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_duration_sec": record["session_duration_sec"],
            "words_per_minute": record["words_per_minute"],
            **result,
        }
        print(json.dumps(payload))
        if log_csv_path is not None:
            append_prediction_log(
                log_csv_path=log_csv_path,
                mode="live_typing",
                record=record,
                result=result,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained typing mental health model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/typing_mental_health_pipeline.joblib"),
        help="Path to the trained joblib model bundle.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Path to JSON input. Supports one object or a list of objects for comparison mode.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run interactive real-time inference using JSON lines from stdin.",
    )
    parser.add_argument(
        "--live-typing",
        action="store_true",
        help="Capture keyboard input in terminal and infer from auto-derived features per typed line.",
    )
    parser.add_argument(
        "--pause-threshold-sec",
        type=float,
        default=0.8,
        help="Gap threshold used to estimate pauses and burst length in --live-typing mode.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV path to append prediction history.",
    )
    parser.add_argument(
        "--compare-all-data",
        action="store_true",
        help="Run prediction for multiple records and generate comparison CSV + grouped-bar chart.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("reports/prediction_comparison.csv"),
        help="Output CSV path for multi-record comparison predictions.",
    )
    parser.add_argument(
        "--comparison-chart",
        type=Path,
        default=Path("reports/prediction_comparison_bar.png"),
        help="Output chart path for multi-record comparison graph.",
    )
    args = parser.parse_args()

    if args.realtime and args.live_typing:
        raise ValueError("Use either --realtime or --live-typing, not both.")
    if args.compare_all_data and (args.realtime or args.live_typing):
        parser.error("--compare-all-data cannot be used with --realtime or --live-typing.")
    if args.live_typing and msvcrt is None:
        parser.error("--live-typing is only supported on Windows terminals.")
    if args.live_typing and not sys.stdin.isatty():
        parser.error("--live-typing requires running from an interactive terminal.")

    bundle = joblib.load(args.model_path)
    pipeline = bundle["pipeline"]
    numeric_columns = bundle["numeric_columns"]
    text_column = bundle["text_column"]
    target_columns = bundle["target_columns"]

    input_records: list[Dict[str, Any]] | None = None
    if args.input_json is not None:
        input_records = load_records_from_json(args.input_json)

    base_record: Dict[str, Any] = {}
    if args.realtime or args.live_typing:
        if input_records is not None:
            if len(input_records) != 1:
                parser.error("--input-json must contain one object in --realtime or --live-typing mode.")
            base_record = input_records[0]

    if args.realtime:
        run_realtime_loop(
            pipeline=pipeline,
            text_column=text_column,
            numeric_columns=numeric_columns,
            target_columns=target_columns,
            base_record=base_record,
            log_csv_path=args.log_csv,
        )
        return
    if args.live_typing:
        run_live_typing_loop(
            pipeline=pipeline,
            text_column=text_column,
            numeric_columns=numeric_columns,
            target_columns=target_columns,
            base_record=base_record,
            pause_threshold_sec=max(0.1, args.pause_threshold_sec),
            log_csv_path=args.log_csv,
        )
        return

    if args.compare_all_data:
        comparison_records = input_records if input_records is not None else DEFAULT_COMPARISON_RECORDS
        comparison_df = predict_records(
            pipeline=pipeline,
            text_column=text_column,
            numeric_columns=numeric_columns,
            target_columns=target_columns,
            records=comparison_records,
        )
        args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(args.comparison_csv, index=False)
        plot_prediction_comparison(
            comparison_df=comparison_df,
            target_columns=target_columns,
            chart_path=args.comparison_chart,
        )

        print("Comparison complete.")
        print(f"Records compared: {len(comparison_records)}")
        print(f"Comparison CSV: {args.comparison_csv}")
        print(f"Comparison chart: {args.comparison_chart}")
        print("\nComparison table:")
        print(comparison_df.to_string(index=False))
        return

    if input_records is not None:
        if len(input_records) != 1:
            parser.error("When not using --compare-all-data, --input-json must contain one object.")
        record = input_records[0]
    else:
        record = DEFAULT_RECORD

    result = predict_from_record(
        pipeline=pipeline,
        text_column=text_column,
        numeric_columns=numeric_columns,
        target_columns=target_columns,
        record=record,
    )
    print(json.dumps(result, indent=2))
    if args.log_csv is not None:
        append_prediction_log(
            log_csv_path=args.log_csv,
            mode="single",
            record=record,
            result=result,
        )


if __name__ == "__main__":
    main()
