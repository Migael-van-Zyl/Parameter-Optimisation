import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pmdarima import auto_arima

# =========================
# SETTINGS
# =========================

# =========================
# FILE PATHS
# =========================
from pathlib import Path

file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"

output_folder = Path(
    r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\results\forecast_plots"
)

output_folder.mkdir(parents=True, exist_ok=True)
category = "Agriculture"
subsic = "Berries"
consumption_col = "totalconsumption"

max_customers = 10
min_history = 24
seasonal_period = 12

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(file_path)

df["reportingmonth"] = pd.to_datetime(df["reportingmonth"], errors="coerce")

df = df[
    (df["category"] == category) &
    (df["subsic"] == subsic)
].copy()

print("Filtered rows:", len(df))
print("Unique customers:", df["customerid"].nunique())

customers = df["customerid"].dropna().unique()[:max_customers]

# =========================
# LOOP THROUGH CUSTOMERS
# =========================
for customer in customers:

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

    split = int(len(ts) * 0.8)

    train = ts.iloc[:split]
    test = ts.iloc[split:]

    print(f"\nRunning Auto SARIMA for customer: {customer}")

    try:
        model = auto_arima(
            train,
            seasonal=True,
            m=seasonal_period,
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            start_P=0,
            start_Q=0,
            max_P=5,
            max_Q=5,
            d=None,
            D=None,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )

        forecast = model.predict(n_periods=len(test))

        plt.figure(figsize=(12, 6))

        plt.plot(train.index, train.values, label="Train / Historical Data")
        plt.plot(test.index, test.values, label="Actual Test Data")
        plt.plot(test.index, forecast, label="Auto SARIMA Forecast")

        plt.title(
            f"Customer {customer} | {category} - {subsic}\n"
            f"Order: {model.order} | Seasonal Order: {model.seasonal_order}"
        )

        plt.xlabel("Reporting Month")
        plt.ylabel("Total Consumption")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = output_folder / f"customer_{customer}_forecast.png"
        plt.savefig(save_path)
        plt.show()

        print("Saved plot:", save_path)
        print("Best order:", model.order)
        print("Best seasonal order:", model.seasonal_order)

    except Exception as e:
        print(f"Failed for customer {customer}: {e}")