# Backblaze Disk Failure Prediction

Predictive maintenance project using Backblaze hard drive SMART telemetry.

## Objective

Predict whether a hard drive will fail within the next 7 days using its current
SMART values and recent historical trends.

## Initial scope

- Backblaze Drive Stats
- Q1 2026 as initial exploration period
- 2-3 HDD models with sufficient drive-days and failure events
- DuckDB + Python for data processing
- Logistic Regression baseline
- LightGBM model
- SHAP interpretability
- Streamlit demo

## Project structure

```text
data/
├── raw/
├── interim/
└── processed/

src/
├── ingestion/
├── features/
├── modeling/
└── evaluation/

notebooks/
app/
tests/