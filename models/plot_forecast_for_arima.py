import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pmdarima import auto_arima

# =====================================================
# SETTINGS
# =====================================================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"

output_folder = Path(
    r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\results\forecast_plots_arima"
)

output_folder.mkdir(parents=True, exist_ok=True)

category = "Agriculture"
subsic = "Berries"      

consumption_col = "totalconsumption"

max_customers = 10
min_history = 24


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_excel(file_path)

df["reportingmonth"] = pd.to_datetime(
    df["reportingmonth"],
    errors="coerce"
)

df = df[
    (df["category"] == category) &
    (df["subsic"] == subsic)
].copy()

print("\nFiltered rows:", len(df))
print("Unique customers:", df["customerid"].nunique())

customers = df["customerid"].dropna().unique()[:max_customers]

# =====================================================
# LOOP THROUGH CUSTOMERS
# =====================================================
for customer in customers:

    print(f"\nProcessing customer: {customer}")

    customer_df = df[
        df["customerid"] == customer
    ].copy()

    ts = (
        customer_df
        .groupby("reportingmonth")[consumption_col]
        .sum()
        .sort_index()
    )

    # =================================================
    # CHECK HISTORY LENGTH
    # =================================================
    if len(ts) < min_history:
        print(
            f"Skipped customer {customer}: "
            f"only {len(ts)} months"
        )
        continue

    # =================================================
    # TRAIN / TEST SPLIT
    # =================================================
    split = int(len(ts) * 0.8)

    train = ts.iloc[:split]
    test = ts.iloc[split:]

    print(
        f"Train size: {len(train)} | "
        f"Test size: {len(test)}"
    )

    # =================================================
    # AUTO ARIMA (NOT SARIMA)
    # =================================================
    try:

        model = auto_arima(
            train,

            seasonal=False,

            start_p=0,
            start_q=0,

            max_p=5,
            max_q=5,

            d=None,
            max_d=5,

            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True
        )

        # =============================================
        # FORECAST
        # =============================================
        forecast = model.predict(
            n_periods=len(test)
        )

        # =============================================
        # PLOT
        # =============================================
        plt.figure(figsize=(12, 6))

        # TRAIN
        plt.plot(
            train.index,
            train.values,
            label="Train / Historical Data"
        )

        # ACTUAL TEST
        plt.plot(
            test.index,
            test.values,
            label="Actual Test Data"
        )

        # FORECAST
        plt.plot(
            test.index,
            forecast,
            label="Auto ARIMA Forecast"
        )

        plt.title(
            f"Customer {customer} | "
            f"{category} - {subsic}\n"
            f"ARIMA Order: {model.order}"
        )

        plt.xlabel("Reporting Month")
        plt.ylabel("Consumption")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # =============================================
        # SAVE PLOT
        # =============================================
        save_path = (
            output_folder /
            f"{subsic}_customer_{customer}_arima_forecast.png"
        )

        plt.savefig(save_path)

        plt.show()

        # =============================================
        # PRINT PARAMETERS
        # =============================================
        print("\nBest ARIMA Order:")
        print(model.order)

        print("Saved plot:", save_path)

    except Exception as e:

        print(
            f"ARIMA failed for customer "
            f"{customer}: {e}"
        )