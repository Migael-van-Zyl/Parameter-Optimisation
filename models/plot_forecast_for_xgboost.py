import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor

# =====================================================
# SETTINGS
# =====================================================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"

output_folder = Path(
    r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\results\forecast_plots_xgb"
)
output_folder.mkdir(parents=True, exist_ok=True)

category = "Agriculture"
subsic = "Cereal"

consumption_col = "totalconsumption"
max_customers = 10
min_history = 24
lags = 3

n_estimators = 100
max_depth = 3
learning_rate = 0.05

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(file_path)
df["reportingmonth"] = pd.to_datetime(df["reportingmonth"], errors="coerce")

df = df[
    (df["category"] == category) &
    (df["subsic"] == subsic)
].copy()

print("Filtered rows:", len(df))
print("Unique customers:", df["customerid"].nunique())

customers = df["customerid"].dropna().unique()[:max_customers]

# =====================================================
# CREATE LAGS
# =====================================================
def create_lags(ts, lags=3):
    lag_df = pd.DataFrame({"target": ts})

    for i in range(1, lags + 1):
        lag_df[f"lag_{i}"] = lag_df["target"].shift(i)

    return lag_df.dropna()

# =====================================================
# LOOP THROUGH CUSTOMERS
# =====================================================
for customer in customers:

    print(f"\nProcessing customer: {customer}")

    customer_df = df[df["customerid"] == customer].copy()

    ts = (
        customer_df
        .groupby("reportingmonth")[consumption_col]
        .sum()
        .sort_index()
    )

    if len(ts) < min_history:
        print(f"Skipped customer {customer}: only {len(ts)} months")
        continue

    lag_df = create_lags(ts, lags=lags)

    if len(lag_df) < 10:
        print(f"Skipped customer {customer}: not enough lagged data")
        continue

    # =================================================
    # TRAIN / TEST SPLIT
    # =================================================
    split = int(len(lag_df) * 0.8)

    train = lag_df.iloc[:split]
    test = lag_df.iloc[split:]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]

    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # =================================================
    # XGBOOST MODEL
    # =================================================
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    forecast = model.predict(X_test)

    # =================================================
    # PLOT
    # =================================================
    plt.figure(figsize=(12, 6))

    plt.plot(train.index, y_train.values, label="Train / Historical Data")
    plt.plot(test.index, y_test.values, label="Actual Test Data")
    plt.plot(test.index, forecast, label="XGBoost Forecast")

    plt.title(
        f"XGBoost Forecast | Customer {customer}\n"
        f"{category} - {subsic} | "
        f"trees={n_estimators}, depth={max_depth}, learning rate={learning_rate}"
    )

    plt.xlabel("Reporting Month")
    plt.ylabel("Total Consumption")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = output_folder / f"{subsic}_customer_{customer}_xgboost_forecast.png"
    plt.savefig(save_path)
    plt.show()

    print("Saved plot:", save_path)