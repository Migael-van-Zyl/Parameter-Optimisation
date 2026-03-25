import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

file_path = r"C:\Users\MauduH\Documents\Migael\cleanedFortrack_dataset.xlsx"
df = pd.read_excel(file_path)

# Filter Agriculture + Cereal
df = df[(df["category"] == "Agriculture") & (df["subsic"] == "Cereal")]

df["reportingmonth"] = pd.to_datetime(df["reportingmonth"])

# PARAMETERS TO TEST
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

results = []

# LOOP THROUGH CUSTOMERS
customers = df["customerid"].unique()

for customer in customers:
    customer_df = df[df["customerid"] == customer]

    # Create time series
    ts = customer_df.groupby("reportingmonth")["totalconsumption"].sum()
    ts = ts.sort_index()

    # Skip if too little data
    if len(ts) < 12:
        continue

    # Train-test split
    split = int(len(ts) * 0.8)
    train = ts.iloc[:split]
    test = ts.iloc[split:]

    best_mape = np.inf
    best_order = None

    # GRID SEARCH ARIMA
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train, order=(p, d, q)).fit()
                    preds = model.forecast(steps=len(test))

                    mape = mean_absolute_percentage_error(test, preds) * 100

                    if mape < best_mape:
                        best_mape = mape
                        best_order = (p, d, q)

                except:
                    continue

    # Store results
    results.append({
        "customerid": customer,
        "best_p": best_order[0] if best_order else None,
        "best_d": best_order[1] if best_order else None,
        "best_q": best_order[2] if best_order else None,
        "MAPE": best_mape
    })

    print(f"Done customer {customer} | Best ARIMA: {best_order} | MAPE: {best_mape:.2f}%")

# RESULTS DATAFRAME
results_df = pd.DataFrame(results)

# FIND MOST COMMON PARAMETERS
param_counts = results_df.groupby(["best_p", "best_d", "best_q"]).size().reset_index(name="count")
param_counts = param_counts.sort_values(by="count", ascending=False)

print("\n=== MOST COMMON PARAMETERS ===")
print(param_counts.head())

# EXPORT TO EXCEL
results_df.to_excel(r"C:\Users\MauduH\Documents\Migael\customer_results2.xlsx", index=False)
param_counts.to_excel(r"C:\Users\MauduH\Documents\Migael\parameter_frequency_arima2.xlsx", index=False)

print("\n=== TESTING COMMON MODEL ARIMA(0,1,0) ===")

common_results = []

order = (0,1,0)

for customer in customers:

    customer_df = df[df["customerid"] == customer]

    ts = customer_df.groupby("reportingmonth")["totalconsumption"].sum()
    ts = ts.sort_index()

    if len(ts) < 12:
        continue

    split = int(len(ts) * 0.8)

    train = ts.iloc[:split]
    test = ts.iloc[split:]

    try:

        model = ARIMA(train, order=order).fit()

        preds = model.forecast(steps=len(test))

        mape = mean_absolute_percentage_error(test, preds) * 100

        common_results.append({
            "customerid": customer,
            "ARIMA_model": "ARIMA(0,1,0)",
            "MAPE": mape
        })

        print(f"Customer {customer} | MAPE: {mape:.2f}%")

    except:
        continue

common_df = pd.DataFrame(common_results)

common_df.to_excel(
    r"C:\Users\MauduH\Documents\Migael\arima_common_model_results.xlsx",
    index=False
)

print("\nARIMA(0,1,0) results exported.")