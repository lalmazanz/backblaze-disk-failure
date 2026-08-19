# Backblaze Disk Failure Prediction

Predictive maintenance pipeline for identifying hard drives at elevated risk of failure using historical SMART telemetry from the **Backblaze Drive Stats** dataset.

The project is designed as an end-to-end machine learning workflow rather than a standalone notebook: raw drive telemetry is filtered and transformed into time-aware features, models are trained using leakage-safe temporal splits, and predictions are evaluated under a realistic **daily inspection budget**.

---

## Overview

Hard drive failures are rare events, making this a highly imbalanced classification problem.

In a real infrastructure environment, however, the goal is not simply to maximize classification accuracy. An operations team has limited capacity and cannot inspect every drive that receives a non-zero failure probability.

The more useful question is:

> **If only a small fraction of the fleet can be inspected each day, how many drives that are about to fail can the model identify in advance?**

This project therefore focuses on **risk ranking**, early-warning detection and operational evaluation.

The final selected model is a **Random Forest classifier** that predicts whether a healthy drive will fail within the next **7 days**.

Predictions are converted into alerts using a **daily top-1% policy**, where only the highest-risk 1% of drives evaluated on each day are flagged for inspection.

---

## Dataset

The project uses the public **Backblaze Drive Stats** dataset.

Backblaze publishes daily SMART telemetry for hundreds of thousands of hard drives deployed in its storage infrastructure.

Each drive-day observation contains information such as:

* date
* serial number
* hard drive model
* failure indicator
* SMART attributes reported by the drive

The full historical dataset contains hundreds of millions of observations, so this project works with a controlled subset of high-volume drive models and a fixed temporal window to keep local experimentation computationally practical.

### Selected drive models

The current experiment focuses on:

* `ST12000NM0008`
* `TOSHIBA MG08ACA16TA`
* `HGST HUH721212ALN604`

This provides observations from multiple hardware families while maintaining enough drive-days and failure events for meaningful evaluation.

---

## Methodology

### Prediction target

The task is formulated as a **7-day failure prediction problem**.

For every healthy drive observation on date `t`, the target `failure_next_7d` is defined as:

* `1` if the drive fails within the following 7 days;
* `0` otherwise.

The failure day itself is excluded from the modeling dataset.

Conceptually:

```text
Observation date                          Failure
      │                                     │
      ▼                                     ▼
──────●─────●─────●─────●─────●─────●─────●──────► time
      └──────────── 7-day horizon ──────────┘

      failure_next_7d = 1
```

The model must therefore detect elevated risk **before the drive actually fails**.

---

### SMART signals

Eight raw SMART attributes are used as the core telemetry signals:

| SMART attribute | General signal                     |
| --------------- | ---------------------------------- |
| SMART 1         | Raw read error rate                |
| SMART 5         | Reallocated sector count           |
| SMART 7         | Seek error rate                    |
| SMART 9         | Power-on hours                     |
| SMART 194       | Temperature                        |
| SMART 197       | Current pending sector count       |
| SMART 198       | Offline uncorrectable sector count |
| SMART 199       | UDMA CRC error count               |

The pipeline does not rely only on the current value of each attribute. It also captures how the signal evolves over time.

---

### Feature engineering

For every SMART attribute, five representations are generated:

```text
current raw value
1-day change
7-day change
7-day rolling mean
7-day rolling maximum
```

Two additional features indicate whether enough historical information is available:

```text
has_1d_history
has_7d_history
```

This produces:

```text
8 SMART attributes × 5 representations = 40
2 history indicators                    =  2
                                             ──
Total model features                    = 42
```

The objective is to capture both the **current state of the drive** and recent changes that may indicate progressive degradation.

---

## Temporal validation strategy

Random row-level splitting is intentionally avoided.

Daily observations from the same hard drive are strongly correlated. Randomly distributing those observations between training and evaluation could expose the model to temporally adjacent states of the same drive and result in overly optimistic performance estimates.

The project instead uses chronological train, validation and test periods.

### Model selection

```text
TRAIN
2026-01-01 ─────────────────── 2026-02-21

                 7-day purge

VALIDATION
2026-03-01 ───────── 2026-03-10
```

The validation period is used for model comparison and hyperparameter selection.

### Final evaluation

After model selection, the training window is expanded:

```text
FINAL TRAIN
2026-01-01 ───────────────────────── 2026-03-03

                            7-day purge

TEST
2026-03-11 ─────────────── 2026-03-24
```

The final test period remains unseen during model selection.

---

## Preventing temporal leakage

The prediction label itself uses information from the following seven days.

Without an additional temporal gap, an observation close to the end of training could receive its label from a failure occurring during the validation or test period.

The project therefore introduces a **7-day temporal purge** between training and evaluation.

For the final evaluation:

```text
Final training ends
2026-03-03

Purged period
2026-03-04 → 2026-03-10

Test begins
2026-03-11
```

The purge duration matches the prediction horizon.

This ensures that the training labels are fully resolved before the evaluation period begins.

---

## Handling class imbalance

Drive failures are extremely rare compared with normal drive-day observations.

Training directly on every healthy observation would create a highly imbalanced dataset and substantially increase local training cost.

The training pipeline therefore:

* keeps **all positive observations**;
* deterministically samples negative observations;
* uses a ratio of **1 positive : 50 negatives**.

The validation and test datasets are not undersampled.

This means reported evaluation results still reflect the natural failure prevalence of the original data.

A fixed random seed is used for reproducibility:

```python
random_state = 42
```

---

## Model selection

Several classification approaches were explored during development:

* Logistic Regression
* LightGBM
* Random Forest

The final selected model is a **Random Forest classifier**.

Its configuration is:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
```

The model outputs a continuous probability-like `risk_score` for every drive-day observation.

The system uses that score primarily for **ranking drives by risk**, rather than converting predictions using a fixed threshold such as `0.5`.

---

## Daily alert policy

The model is evaluated under an explicit operational constraint:

> **Only the highest-risk 1% of drives evaluated each day can generate an alert.**

For each evaluation date:

1. every available drive receives a `risk_score`;
2. drives are ranked from highest to lowest risk;
3. the daily alert budget is calculated as 1% of that day's evaluated fleet;
4. only drives inside that budget are flagged.

At least one drive can be alerted on each evaluation day.

Conceptually:

```text
Daily fleet
────────────────────────────────────────────

Model risk ranking

#1   ████████████████████   ALERT
#2   ███████████████████    ALERT
#3   ██████████████████     ALERT
...          top 1%

────────────────────────── alert cutoff

...
#N   ██                     no alert
```

This converts the model from a pure classifier into a **prioritization system**.

Instead of asking whether every individual drive is predicted correctly, the system asks which drives should receive scarce maintenance attention first.

---

## Evaluation metrics

Because drive failures are highly imbalanced, traditional accuracy is not used as the main success metric.

The project reports several complementary measures.

### ROC-AUC

Measures the overall ranking ability of the model across positive and negative observations.

### PR-AUC

Precision-Recall AUC focuses more directly on the rare positive class and is therefore particularly relevant for failure prediction.

### Alert volume

Measures the number of drive-day observations flagged under the daily 1% operational policy.

### Drive-level recall

Measures whether a drive that eventually fails generated at least one successful alert during its pre-failure prediction window.

The central operational question becomes:

> **Within a daily inspection budget of only 1% of the fleet, how many failing drives can be identified in advance?**

---

## Results

The final Random Forest model was evaluated on the held-out test period:

```text
2026-03-11 → 2026-03-24
```

The test set contains **947,283 drive-day observations**, including **199 positive rows** corresponding to drives within seven days of failure.

### Overall performance

| Metric                  |     Result |
| ----------------------- | ---------: |
| ROC-AUC                 | **0.8533** |
| PR-AUC                  | **0.0397** |
| Row-level precision     | **0.0623** |
| Row-level recall        | **0.4070** |
| Failing drives          |     **54** |
| Detected failing drives |     **31** |
| Drive-level recall      |  **57.4%** |
| Daily inspection budget |     **1%** |

The operational result is the most important one:

> **The model identified 31 of 54 failing drives while restricting daily inspections to approximately 1% of the evaluated fleet.**

Across the test period, the policy generated 9,465 alert observations involving 1,437 unique drives.

---

### Failure detection by drive model

![Failure detection by drive model](reports/figures/failure_detection_by_model.png)

Performance varies noticeably across hardware families:

| Drive model          | Failing drives | Detected | Drive recall |
| -------------------- | -------------: | -------: | -----------: |
| ST12000NM0008        |             19 |       13 |    **68.4%** |
| TOSHIBA MG08ACA16TA  |             21 |       10 |    **47.6%** |
| HGST HUH721212ALN604 |             14 |        8 |    **57.1%** |

The Seagate model achieves the highest operational recall, detecting more than two thirds of the drives that failed.

The HGST model, however, achieves the strongest ROC-AUC despite lower drive-level recall. This illustrates why conventional ranking metrics alone are not sufficient for evaluating a predictive-maintenance system under a constrained alert budget.

---

### Failure warning lead time

For each successfully detected failing drive, lead time is measured as the number of days between its earliest useful alert and the observed failure date.

![Failure warning lead time](reports/figures/failure_warning_lead_time.png)

The model achieved:

* **mean warning time:** 4.52 days;
* **median warning time:** 6 days;
* **maximum warning time:** 7 days;
* **14 of 31 detected failures** were identified with the full 7-day warning horizon.

The distribution was:

| Days before failure | Detected drives |
| ------------------: | --------------: |
|                   1 |               8 |
|                   2 |               3 |
|                   3 |               1 |
|                   4 |               2 |
|                   5 |               1 |
|                   6 |               2 |
|                   7 |              14 |

This means the model is not merely detecting drives immediately before failure. A substantial proportion of successful alerts occur several days in advance, which is significantly more useful from a maintenance perspective.

---

## Model explainability

Predicting elevated failure risk is useful, but understanding **why** the model assigns that risk is also important.

SHAP analysis is used at both global and individual prediction levels.

### Global feature behaviour

The SHAP beeswarm plot shows how feature values influence predictions across a sample of test observations.

![SHAP feature impact](reports/figures/final_model_shap_beeswarm.png)

This visualization helps identify which SMART measurements and temporal changes contribute most strongly to elevated or reduced failure risk.

Unlike standard Random Forest feature importance, SHAP also provides information about the **direction and magnitude** of each feature's effect on individual predictions.

---

### Explaining a detected failure

The project also generates an explanation for an individual drive that was successfully detected before failure.

![Detected failure SHAP waterfall](reports/figures/final_model_detected_drive_waterfall.png)

The waterfall plot decomposes the drive's predicted risk score into the contribution of individual SMART features.

This provides a concrete example of how the system moves from raw telemetry to an operational maintenance alert:

```text
SMART telemetry
      ↓
Engineered temporal features
      ↓
Random Forest risk score
      ↓
Daily risk ranking
      ↓
Top-1% alert
      ↓
Drive fails within 7 days
```

Together, the global and local explanations provide two complementary views:

* **global SHAP:** which signals generally drive model behaviour;
* **local SHAP:** why a particular drive was considered high risk.

---

## Key takeaway

The final system should be interpreted as a **risk-prioritization tool**, not as a binary failure detector.

Under a daily inspection capacity of approximately **1% of the fleet**, it detects **57.4% of failing drives**, with a median warning time of **6 days**.

The result demonstrates the trade-off at the core of predictive maintenance:

> **maximize useful early warnings while keeping the number of operational interventions small.**


### Performance by drive model

| Drive model          |    Rows | Positive rows | Alert rows | Failing drives | Detected drives |   ROC-AUC | PR-AUC | Drive recall |
| -------------------- | ------: | ------------: | ---------: | -------------: | --------------: | --------: | -----: | -----------: |
| ST12000NM0008        | 258,282 |            69 |      3,678 |             19 |              13 |     0.781 |  0.055 |    **68.4%** |
| TOSHIBA MG08ACA16TA  | 554,096 |            87 |      3,053 |             21 |              10 |     0.831 |  0.044 |    **47.6%** |
| HGST HUH721212ALN604 | 134,905 |            43 |      2,734 |             14 |               8 | **0.959** |  0.026 |    **57.1%** |

These results show an important property of rare-event predictive maintenance.

A model can achieve strong ranking metrics while still producing different operational outcomes across hardware families.

For example, the HGST model achieves the highest ROC-AUC, while the Seagate model captures a larger proportion of failing drives under the same daily alert constraint.

This is why conventional metrics such as ROC-AUC are evaluated alongside operational drive-level recall.

---

## Pipeline

The complete workflow is organized as a reproducible pipeline:

```text
Backblaze Drive Stats
          │
          ▼
      Raw ingestion
          │
          ▼
   Drive-model filtering
          │
          ▼
 Build 7-day failure target
          │
          ▼
   Feature engineering
      42 features
          │
          ▼
    Temporal splitting
          │
    ┌─────┴─────┐
    │           │
 Training     Evaluation
    │           │
1:50 negative   │
undersampling   │
    │           │
    └─────┬─────┘
          ▼
    Model training
          │
          ▼
      Risk scoring
          │
          ▼
 Rank drives each day
          │
          ▼
 Daily top-1% policy
          │
          ▼
 ROC-AUC / PR-AUC
 Drive-level recall
```

---

## Project structure

```text
backblaze-disk-failure/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── modeling/
│   └── evaluation/
│
├── tests/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

### Main modules

`src/ingestion/`

Handles raw Backblaze data exploration and creation of the selected-model dataset.

`src/features/`

Builds the 7-day failure target, engineered SMART features and temporal split diagnostics.

`src/modeling/`

Contains dataset loading utilities, model training, tuning, inference and model export.

`src/evaluation/`

Contains final evaluation, breakdown analysis, alert-policy logic, feature importance and explainability utilities.

---

## Reproducibility

The project is designed so that the same source data and configuration reproduce the same modeling workflow.

Important reproducibility decisions include:

* fixed temporal boundaries;
* explicit 7-day leakage purge;
* deterministic negative undersampling;
* fixed random seed;
* centralized feature configuration;
* centralized model parameters;
* reusable alert-policy logic;
* automated tests;
* modular training and evaluation scripts.

Configuration is centralized in:

```text
src/config.py
```

This includes:

* selected drive models;
* SMART attributes;
* feature definitions;
* prediction horizon;
* train/validation/test periods;
* undersampling ratio;
* alert budget;
* model hyperparameters.

---

## Running the project

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Run the test suite

```bash
pytest -q
```

### 4. Run the pipeline

```bash
python -m src.pipeline
```

### 5. Run the final evaluation

```bash
python -m src.evaluation.evaluate_random_forest_final
```

### 6. Evaluate performance breakdowns

```bash
python -m src.evaluation.evaluate_breakdowns
```

---

## Tech stack

* **Python**
* **pandas**
* **DuckDB**
* **scikit-learn**
* **LightGBM**
* **Parquet**
* **pytest**
* **Ruff**
* **Git**
* **GitHub**

---

## Engineering decisions

### Why DuckDB?

The original Backblaze dataset contains hundreds of millions of drive-day observations.

DuckDB allows large CSV and Parquet datasets to be filtered, aggregated and transformed efficiently without requiring a separate database server.

This makes it well suited for an analytical project that runs locally.

---

### Why Parquet?

Intermediate and processed datasets are stored as Parquet files.

Compared with repeatedly processing the original CSV data, Parquet provides:

* columnar storage;
* reduced disk usage;
* faster analytical queries;
* efficient integration with DuckDB and pandas.

---

### Why not use a random train/test split?

SMART observations from the same drive evolve over time and are highly correlated.

A random split could put nearby observations from the same physical drive into both training and evaluation.

The project therefore treats this as a **temporal prediction problem** and evaluates only on future periods.

---

### Why use a 7-day horizon?

The purpose of the system is early warning rather than same-day failure detection.

A multi-day horizon creates time for a hypothetical operations team to inspect, migrate or replace an elevated-risk drive before failure occurs.

---

### Why use a purge period?

Because the target itself looks seven days into the future, training rows close to an evaluation boundary could otherwise derive their labels from failures occurring during the evaluation period.

A seven-day purge prevents this leakage.

---

### Why undersample negatives?

The overwhelming majority of drive-day observations correspond to healthy drives.

Keeping all of those rows provides limited additional information while dramatically increasing computational cost.

Negative undersampling keeps all observed failure-related examples while providing a large and reproducible healthy comparison sample.

---

### Why use PR-AUC?

Failure observations represent only a tiny fraction of the dataset.

ROC-AUC remains useful for ranking evaluation but can look strong even in extremely imbalanced problems.

PR-AUC provides additional insight into how effectively the model isolates the positive class.

---

### Why use a daily top-1% policy?

A predictive-maintenance system has an operational cost.

If the model flags a very large portion of the fleet, even high recall may not translate into a useful maintenance strategy.

Limiting intervention to approximately 1% of drives per day creates an explicit resource constraint and allows the model to be evaluated as a prioritization system.

---

### Why evaluate each drive model separately?

SMART behaviour is not perfectly standardized across manufacturers and drive families.

The same raw SMART attribute can behave differently across hardware models.

A single aggregate score could hide these differences, so performance is also broken down by individual drive model.

---

## Limitations

The current project has several important limitations.

### Rare failure events

Even very large telemetry datasets contain relatively few failures compared with healthy observations.

This limits the statistical certainty of fine-grained model comparisons.

### Hardware heterogeneity

Different drive manufacturers and models may expose different degradation patterns through SMART telemetry.

A single global model may therefore perform unevenly across drive families.

### Limited temporal window

The current experiment focuses on a restricted period rather than the full Backblaze historical dataset.

A longer evaluation horizon would provide stronger evidence about temporal generalization.

### Fixed alert budget

The daily 1% threshold is intentionally simple.

A real production system would likely optimize intervention thresholds according to maintenance capacity and the economic cost of false positives and missed failures.

### Failure definition

The model predicts the failure events recorded by Backblaze.

It does not distinguish between different physical causes of failure.

---

## Possible extensions

Future work could include:

* additional drive models;
* longer historical training windows;
* rolling backtesting;
* probability calibration;
* cost-sensitive learning;
* model-specific alert thresholds;
* alternative prediction horizons;
* SHAP-based model interpretation;
* survival analysis;
* fleet-level drift monitoring;
* automated experiment tracking;
* comparison with additional gradient-boosting approaches;
* deployment of the scoring pipeline as a batch inference service.

---

## What this project demonstrates

This project is intended to demonstrate more than model training alone.

It covers:

* large-scale data processing;
* SQL analytics with DuckDB;
* Parquet-based analytical workflows;
* temporal feature engineering;
* rare-event classification;
* class imbalance handling;
* leakage-aware validation;
* model comparison;
* hyperparameter tuning;
* predictive-maintenance modeling;
* operational alert-policy design;
* model evaluation by hardware segment;
* reproducible Python pipelines;
* automated testing;
* modular project architecture.

The objective is not simply to obtain a high predictive score, but to build a machine learning workflow that answers a realistic operational question:

> **Which drives should be inspected today if maintenance capacity is limited?**
