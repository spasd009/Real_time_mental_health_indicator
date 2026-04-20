# Real-Time Typing Behavior Analysis for Mental Health Indicators Using NLP
**Author:** Shubham

## 1. What this project is
This package is a **working research prototype** for the project shown in your image:

- **Project title:** Real-Time Typing Behavior Analysis for Mental Health Indicators Using NLP
- **Outcome target:** Research paper / research prototype
- **Core skills covered:** Python, Machine Learning, NLP, Data Analysis, Statistics

The model estimates three **non-clinical research indicators** from typing-session data:

- `stress_pct`
- `anxiety_pct`
- `depression_pct`

These values are produced as **0 to 100 percentage-style scores**.

## 2. Important limitation
This repository uses a **synthetic dataset** generated from explicit rules. That makes it useful for:

- architecture design
- feature engineering demonstrations
- baseline model training
- paper prototyping
- pipeline validation

It is **not** suitable for clinical claims, diagnosis, screening deployment, or publication-grade performance claims without a real, consented dataset and formal validation.

## 3. Project structure

```text
typing_mental_health_model/
|-- .venv/
|-- train_model.py                      # launcher (run from outer folder)
|-- predict.py                          # launcher (run from outer folder)
|-- compare_government_processed.py     # launcher (run from outer folder)
`-- typing_mental_health_model/
    |-- README.md
    |-- requirements.txt
    |-- train_model.py
    |-- predict.py
    |-- sample_input.json
    |-- sample_compare_input.json
    |-- dataset/
    |   |-- synthetic_typing_mental_health.csv
    |   `-- government_mental_health_reference.csv
    |-- models/
    |   `-- typing_mental_health_pipeline.joblib
    `-- reports/
        |-- metrics.json
        |-- test_predictions_preview.csv
        |-- prediction_comparison.csv
        |-- prediction_comparison_bar.png
        |-- realtime_predictions_log.csv
        |-- government_vs_processed_summary.csv
        `-- government_vs_processed_bar.png
```

## 4. Problem framing
The research goal is to infer **behavioral indicators** from typing sessions without interrupting the user.

### Input signal groups
1. **Keystroke dynamics**
   - dwell time
   - flight time
   - pause ratio
   - correction latency
   - backspace rate
   - error rate
   - typing speed
   - burst length

2. **Contextual usage signals**
   - session duration
   - late-night typing ratio

3. **Lightweight NLP features**
   - session text
   - sentiment polarity
   - negation ratio
   - uncertainty ratio
   - first-person ratio
   - lexical diversity

### Output targets
The model predicts three continuous outputs:

- stress percentage
- anxiety percentage
- depression percentage

## 5. Dataset schema
The dataset row unit is **one typing session**.

| Column | Type | Meaning |
|---|---|---|
| `text` | string | Typed text sample from the session |
| `avg_dwell_time_ms` | float | Average key hold time |
| `avg_flight_time_ms` | float | Average inter-key interval |
| `pause_ratio` | float | Fraction of paused time |
| `backspace_rate` | float | Backspace events per tokenized unit |
| `error_rate` | float | Estimated typo/error proportion |
| `correction_latency_ms` | float | Time from error to correction |
| `words_per_minute` | float | Typing speed |
| `burst_length` | float | Mean uninterrupted token burst length |
| `session_duration_sec` | float | Total session duration |
| `late_night_ratio` | float | Fraction of activity occurring late night |
| `sentiment_polarity` | float | Normalized sentiment score |
| `negation_ratio` | float | Rate of negation language |
| `uncertainty_ratio` | float | Rate of uncertainty words |
| `first_person_ratio` | float | First-person pronoun usage |
| `lexical_diversity` | float | Unique-token diversity proxy |
| `stress_pct` | float | Target stress indicator |
| `anxiety_pct` | float | Target anxiety indicator |
| `depression_pct` | float | Target depression indicator |

## 6. Model design

### Pipeline
The training code builds a multi-output regression system:

1. **Text branch**
   - `TfidfVectorizer`
   - unigrams and bigrams
   - max 1200 features

2. **Numeric branch**
   - median imputation
   - standardization

3. **Fusion layer**
   - `ColumnTransformer` combines text and numeric features

4. **Prediction layer**
   - `MultiOutputRegressor(Ridge)`
   - one regressor per target

### Why this baseline is reasonable
- sparse text features are handled cleanly
- numeric behavioral signals stay interpretable
- training is fast
- baseline is stable and easy to extend
- output is continuous, which matches percentage-style indicators

## 7. How the synthetic labels are produced
The data generator creates correlated latent variables for:

- stress
- anxiety
- depression

Then it maps those latent states into:

- typing speed changes
- pause and correction patterns
- night-time activity patterns
- sentiment and language markers
- generated session text

This yields a dataset with structure strong enough to validate the end-to-end pipeline.

## 8. How to run

### Install dependencies
```bash
pip install -r .\typing_mental_health_model\requirements.txt
```

### PowerShell / VS Code terminal setup
Run from outer folder:

```powershell
Set-Location "C:\Users\spasd\Downloads\typing_mental_health_model"
& ".\.venv\Scripts\Activate.ps1"
```

### Train the model
```powershell
python train_model.py --output-dir .
```

### Run inference on the demo record
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json
```

Save prediction history to CSV:
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json --log-csv reports/realtime_predictions_log.csv
```

### Compare multiple records and generate a graph
Use built-in comparison samples:
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --compare-all-data
```

Use custom multi-record input:
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_compare_input.json --compare-all-data --comparison-csv reports/prediction_comparison.csv --comparison-chart reports/prediction_comparison_bar.png
```

Outputs:
- `reports/prediction_comparison.csv`
- `reports/prediction_comparison_bar.png`

### Run real-time analysis (interactive JSON lines)
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json --realtime
```

Then enter one JSON object per line (you can send only changed fields when `--input-json` is provided):
```json
{"text":"I keep revisiting tasks and feel uncertain about mistakes."}
```

With logging:
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json --realtime --log-csv reports/realtime_predictions_log.csv
```

### Run live typing capture (Windows terminal)
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json --live-typing
```

Behavior:
- Type naturally in the terminal and press Enter.
- The script estimates typing features from key timing and immediately prints prediction JSON.
- Type `/exit` and press Enter to stop.

Optional:
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json --live-typing --pause-threshold-sec 0.8
```

With logging:
```powershell
python predict.py --model-path models/typing_mental_health_pipeline.joblib --input-json sample_input.json --live-typing --log-csv reports/realtime_predictions_log.csv
```

### Compare government data vs processed data (bar graph)
```powershell
python compare_government_processed.py --government-csv dataset/government_mental_health_reference.csv --processed-csv reports/realtime_predictions_log.csv --output-summary-csv reports/government_vs_processed_summary.csv --output-chart reports/government_vs_processed_bar.png
```

Outputs:
- `reports/government_vs_processed_summary.csv`
- `reports/government_vs_processed_bar.png`

### Example outputs (verified)
Single prediction output:
```json
{
  "stress_pct": 58.038,
  "anxiety_pct": 43.217,
  "depression_pct": 42.627
}
```

Comparison output summary:
```text
record_label        stress_pct  anxiety_pct  depression_pct
baseline               52.208       40.353          37.604
high_stress_night      93.487       61.717          82.612
moderate_fatigue       59.233       44.678          53.192
calm_productive        23.233       13.484          21.058
recovering             36.720       29.007          30.564
```

## 9. Expected outputs after training
Training creates:

- `dataset/synthetic_typing_mental_health.csv`
- `models/typing_mental_health_pipeline.joblib`
- `reports/metrics.json`
- `reports/test_predictions_preview.csv`

## 10. Suggested research-paper methodology section
Use this structure in your paper:

### Title
Real-Time Typing Behavior Analysis for Mental Health Indicators Using NLP

### Objective
Develop a privacy-aware, non-intrusive machine-learning framework that estimates behavioral mental-health indicators from keystroke dynamics and lightweight NLP features.

### Method
- collect session-level typing logs with informed consent
- extract keystroke timing features
- derive NLP features from typed text
- normalize and fuse features
- train multi-output regression or ordinal classification models
- evaluate with cross-validation and calibration checks

### Evaluation metrics
- RMSE per target
- MAE per target
- R2 per target
- calibration error
- subgroup robustness checks

### Ethics section
- no diagnosis claims
- explicit consent required
- data minimization
- on-device processing preferred where possible
- human review for any high-risk workflow

## 11. How to replace the synthetic dataset with a real one
A real dataset should keep the same input columns. Minimum practical steps:

1. Collect one row per session.
2. Keep the numeric feature names identical when possible.
3. Replace the synthetic target columns with validated study labels or questionnaire-linked scores.
4. Retrain using the same training script.
5. Re-run evaluation.

## 12. Strong next-step upgrades
If you want publication-quality work, replace the baseline with one of these:

### Option A: Better tabular model
- LightGBM or XGBoost for numeric features
- separate text encoder
- late fusion ensemble

### Option B: Sequence model
- temporal windows over keystroke streams
- LSTM, TCN, or Transformer encoder

### Option C: Personalized modeling
- subject-specific normalization
- mixed-effects modeling
- domain adaptation by user group

## 13. Deployment architecture for a real system
- local keystroke collector
- feature extraction buffer
- privacy filter for text
- model inference service
- risk-indicator dashboard
- threshold-free longitudinal trend view

## 14. Final warning
This package is a **research scaffold**, not a health product. Use it to demonstrate methodology, code structure, and a reproducible baseline.
