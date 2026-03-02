# '''
# TODO:
# Reconfigure XGBoost script to be importable and callable from this main.py file.
# Call and test functionality in main.py
# Repeat for RF and SARIMA

# FUTURE:
# Look at refining classification algorithm and how we assign best parameters
# '''
# #test 
# # IMPORT LIBRARIES
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# from pathlib import Path
# from models.xgb import xgb_model, y_pred as xgb_y_pred

# BASE_DIR = Path(__file__).resolve().parent.parent


# # LOAD TRAIN & TEST DATA
# train_df = pd.read_excel(BASE_DIR / "Parameter Optimisation" / "data" / "batchedtrain_dataset.xlsx")
# test_df  = pd.read_excel(BASE_DIR / "Parameter Optimisation" / "data" / "batchedtest_dataset.xlsx")

# # CALL XGBOOST FUNCTIONS
# # CALL RF FUNCTIONS
# # CALL SARIMA FUNCTIONS

 # main.py
# from pathlib import Path
# import pandas as pd
# from Utils.data import load_classifier_data
# from Utils.metrics import evaluate_classifier
# from models.rf import train_rf, predict_rf
# from models.xgb import train_xgb, predict_xgb
# from models.sarima import train_sarima, predict_sarima, evaluate_forecast
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# # main.py (top)
# # from models.Data import load_classifier_data
# # from models.Metrics import evaluate_classifier

# # from models.rf import train_rf, predict_rf
# # from models.xgb import train_xgb, predict_xgb
# # from models.sarima import train_sarima, predict_sarima, evaluate_forecast
# # ---------- Paths ----------
# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / "Parameter Optimisation" / "data"

# TRAIN_CLS_PATH = DATA_DIR / "batchedtrain_dataset.xlsx"
# TEST_CLS_PATH  = DATA_DIR / "batchedtest_dataset.xlsx"

# # Optional time-series paths (provide your actual files)
# # Must contain columns: ['unique_id', 'ds' (datetime), 'y']
# TS_TRAIN_PATH = BASE_DIR / "ts_data" / "train_timeseries.xlsx"
# TS_TEST_PATH  = BASE_DIR / "ts_data" / "test_timeseries.xlsx"

# # ---------- Features ----------
# FEATURES = [
#     "offpeakconsumption",
#     "standardconsumption",
#     "peakconsumption",
#     "totalconsumption",
# ]
# TARGET = "sic_batch"

# def run_classification():
#     print("\n=== Classification Pipeline (RF & XGBoost) ===")

#     X_train, X_test, y_train, y_test, label_encoder, used_features = load_classifier_data(
#         train_path=TRAIN_CLS_PATH,
#         test_path=TEST_CLS_PATH,
#         target_col=TARGET,
#         feature_cols=FEATURES  # or None to auto-select numeric features
#     )

#     label_names = label_encoder.classes_

#     # -------- Random Forest --------
#     print("\n--- Random Forest ---")
#     rf_model = train_rf(X_train, y_train)
#     rf_pred = predict_rf(rf_model, X_test)
#     rf_results = evaluate_classifier(
#         y_true=y_test,
#         y_pred=rf_pred,
#         label_names=label_names,
#         title="Random Forest – SIC Batch Confusion Matrix (13 Batches)",
#         show_plot=True
#     )

#     # -------- XGBoost --------
#     print("\n--- XGBoost ---")
#     xgb_model = train_xgb(X_train, y_train)
#     xgb_pred = predict_xgb(xgb_model, X_test)
#     xgb_results = evaluate_classifier(
#         y_true=y_test,
#         y_pred=xgb_pred,
#         label_names=label_names,
#         title="XGBoost – SIC Batch Confusion Matrix",
#         show_plot=True
#     )

#     print("\n[RF] Accuracy:", rf_results["accuracy"])
#     print("[XGB] Accuracy:", xgb_results["accuracy"])

# def load_ts_df(path: Path) -> pd.DataFrame:
#     """
#     Helper to load a time series Excel with columns ['unique_id','ds','y'].
#     Ensures 'ds' is datetime and monthly-aligned.
#     """
#     df = pd.read_excel(path)
#     # Basic validation
#     required = {"unique_id", "ds", "y"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Time series file missing columns: {missing}")

#     df = df.copy()
#     df["ds"] = pd.to_datetime(df["ds"])
#     # Align to month start if monthly
#     df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
#     return df

# def run_sarima():
#     print("\n=== SARIMA Pipeline ===")
#     if not TS_TRAIN_PATH.exists() or not TS_TEST_PATH.exists():
#         print("Time series files not found; skipping SARIMA. "
#               f"Expected:\n  {TS_TRAIN_PATH}\n  {TS_TEST_PATH}")
#         return

#     ts_train = load_ts_df(TS_TRAIN_PATH)
#     ts_test  = load_ts_df(TS_TEST_PATH)

#     # Train SARIMA (AutoARIMA seasonal)
#     sf = train_sarima(ts_train, season_length=12, alias="SARIMA", freq="MS")

#     # Forecast horizon = how many periods in test set per series
#     # If multiple series, use the minimum count per series
#     h = (
#         ts_test.groupby("unique_id")["ds"]
#         .count()
#         .min()
#     )
#     if pd.isna(h) or h <= 0:
#         raise ValueError("Invalid forecast horizon computed from test data.")

#     y_pred = predict_sarima(sf, h=int(h))
#     # y_pred columns typically: ['unique_id','ds','SARIMA']
#     # Evaluate
#     eval_df = evaluate_forecast(ts_test, y_pred)
#     print("\nSARIMA Evaluation (by metric):")
#     print(eval_df)

# if __name__ == "__main__":
#     # 1) Classification: Random Forest & XGBoost
#     run_classification()

#     # 2) Forecasting: SARIMA (if files present)
#     run_sarima()

# main.py
from pathlib import Path
print(">>> main.py started")

# ---- Imports (must succeed without reading files) ----
from Utils.data import load_classifier_data
from Utils.metrics import evaluate_classifier
from models.rf import train_rf, predict_rf
from models.xgb import train_xgb, predict_xgb
# from models.sarima import train_sarima, predict_sarima, evaluate_forecast  # optional for later

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent

# 🔧 CHANGE THIS to the folder that ACTUALLY has your Excel files:
DATA_DIR = BASE_DIR / "data"     # e.g., .../Parameter-Optimisation/data

TRAIN_CLS_PATH = DATA_DIR / "batchedtrain_dataset.xlsx"
TEST_CLS_PATH  = DATA_DIR / "batchedtest_dataset.xlsx"

# ---- Features & Target ----
FEATURES = ["offpeakconsumption","standardconsumption","peakconsumption","totalconsumption"]
TARGET = "sic_batch"

def run_classification():
    # Debug: verify paths
    print("CWD:", Path.cwd())
    print("BASE_DIR:", BASE_DIR)
    print("TRAIN:", TRAIN_CLS_PATH, "exists:", TRAIN_CLS_PATH.exists())
    print("TEST :", TEST_CLS_PATH,  "exists:", TEST_CLS_PATH.exists())

    if not TRAIN_CLS_PATH.exists() or not TEST_CLS_PATH.exists():
        raise FileNotFoundError(
            "Check DATA_DIR and filenames. The paths above must exist.\n"
            f"DATA_DIR currently: {DATA_DIR}"
        )

    # Load data
    X_train, X_test, y_train, y_test, le, used_features = load_classifier_data(
        train_path=TRAIN_CLS_PATH,
        test_path=TEST_CLS_PATH,
        target_col=TARGET,
        feature_cols=FEATURES
    )
    print(">>> Data loaded:", X_train.shape, X_test.shape)

    # RF
    print(">>> Training RF...")
    rf_model = train_rf(X_train, y_train)
    rf_pred = predict_rf(rf_model, X_test)
    evaluate_classifier(
        y_true=y_test,
        y_pred=rf_pred,
        label_names=le.classes_,
        title="Random Forest – SIC Batch",
        show_plot=True
    )

    # XGB
    print(">>> Training XGB...")
    xgb_model = train_xgb(X_train, y_train)
    xgb_pred = predict_xgb(xgb_model, X_test)
    evaluate_classifier(
        y_true=y_test,
        y_pred=xgb_pred,
        label_names=le.classes_,
        title="XGBoost – SIC Batch",
        show_plot=True
    )

if __name__ == "__main__":
    run_classification()
    print(">>> finished")