from pathlib import Path

# Project paths
RAW_DATA_GLOB = Path("data/raw/*.csv")
SUBSET_PATH = Path("data/interim/q1_2026_selected_models.parquet")
TARGET_PATH = Path("data/processed/q1_2026_failure_target.parquet")
FEATURES_PATH = Path("data/processed/q1_2026_features.parquet")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
MODEL_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

# Selected drive models
SELECTED_MODELS = [
    "ST12000NM0008",
    "TOSHIBA MG08ACA16TA",
    "HGST HUH721212ALN604",
]


# SMART attributes used as core signals
SMART_COLUMNS = [
    "smart_1_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_199_raw",
]


# Engineered model features
FEATURE_COLUMNS = [
    "has_1d_history",
    "has_7d_history",
    "smart_1_raw",
    "smart_1_raw_delta_1d",
    "smart_1_raw_delta_7d",
    "smart_1_raw_mean_7d",
    "smart_1_raw_max_7d",
    "smart_5_raw",
    "smart_5_raw_delta_1d",
    "smart_5_raw_delta_7d",
    "smart_5_raw_mean_7d",
    "smart_5_raw_max_7d",
    "smart_7_raw",
    "smart_7_raw_delta_1d",
    "smart_7_raw_delta_7d",
    "smart_7_raw_mean_7d",
    "smart_7_raw_max_7d",
    "smart_9_raw",
    "smart_9_raw_delta_1d",
    "smart_9_raw_delta_7d",
    "smart_9_raw_mean_7d",
    "smart_9_raw_max_7d",
    "smart_194_raw",
    "smart_194_raw_delta_1d",
    "smart_194_raw_delta_7d",
    "smart_194_raw_mean_7d",
    "smart_194_raw_max_7d",
    "smart_197_raw",
    "smart_197_raw_delta_1d",
    "smart_197_raw_delta_7d",
    "smart_197_raw_mean_7d",
    "smart_197_raw_max_7d",
    "smart_198_raw",
    "smart_198_raw_delta_1d",
    "smart_198_raw_delta_7d",
    "smart_198_raw_mean_7d",
    "smart_198_raw_max_7d",
    "smart_199_raw",
    "smart_199_raw_delta_1d",
    "smart_199_raw_delta_7d",
    "smart_199_raw_mean_7d",
    "smart_199_raw_max_7d",
]


# Prediction setup
PREDICTION_HORIZON_DAYS = 7
TEMPORAL_PURGE_DAYS = PREDICTION_HORIZON_DAYS
LAST_OBSERVABLE_DATE = "2026-03-24"


# Temporal split
#
# A 7-day purge is applied between the final labelled
# training observation and each evaluation period.
#
# Validation starts on 2026-03-01:
# 2026-02-21 + 7 days = 2026-02-28
#
# Test starts on 2026-03-11:
# 2026-03-03 + 7 days = 2026-03-10

TRAIN_START = "2026-01-01"
TRAIN_END = "2026-02-21"

VALIDATION_START = "2026-03-01"
VALIDATION_END = "2026-03-10"

FINAL_TRAIN_START = "2026-01-01"
FINAL_TRAIN_END = "2026-03-03"

TEST_START = "2026-03-11"
TEST_END = "2026-03-24"


# Modeling
NEGATIVE_RATIO = 50
TOP_PCT = 0.01
RANDOM_STATE = 42


# LightGBM
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
