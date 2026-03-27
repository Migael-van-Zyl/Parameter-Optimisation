import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# 1. LOAD DATA
file_path = r"C:\Users\MauduH\Documents\Migael\cleanedFortrack_dataset.xlsx"
output_folder = r"C:\Users\MauduH\Documents\Migael\ARIMA_Outputs"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_excel(file_path)

# 2. BASIC CLEANING
df["reportingmonth"] = pd.to_datetime(df["reportingmonth"], errors="coerce")
df = df.dropna(subset=["reportingmonth", "category", "subsic", "customerid"])

# 3. CONSUMPTION COLUMNS
consumption_columns = {
    "total": "totalconsumption",
    "peak": "peakconsumption",
    "standard": "standardconsumption",
    "offpeak": "offpeakconsumption"
}

consumption_columns = {
    period_name: col_name
    for period_name, col_name in consumption_columns.items()
    if col_name in df.columns
}

if not consumption_columns:
    raise ValueError(
        "None of the consumption columns were found in the dataset. "
        "Please update the consumption_columns dictionary."
    )

# 4. ARIMA PARAMETERS TO TEST
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

# 5. HELPER FUNCTION
def calculate_smape(actual, forecast):
    """
    SMAPE works better than MAPE when data contains zeros or negatives.
    """
    actual = np.array(actual)
    forecast = np.array(forecast)

    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    smape = np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask]) * 100
    return smape


# 6. MASTER RESULT LISTS
all_best_results = []
all_param_counts = []
all_common_model_results = []

# 7. LOOP THROUGH ALL CATEGORIES
for category in sorted(df["category"].dropna().unique()):
    category_df = df[df["category"] == category].copy()

    subsics = sorted(category_df["subsic"].dropna().unique())

    print(f"\n========== CATEGORY: {category} ==========")

    # 8. LOOP THROUGH SUBSICS
    for subsic in subsics:
        subsic_df = category_df[category_df["subsic"] == subsic].copy()

        print(f"\n----- SUBSIC: {subsic} -----")

        # 9. LOOP THROUGH CONSUMPTION TYPES
        for period_name, consumption_col in consumption_columns.items():
            print(f"\n### Running period: {period_name} | column: {consumption_col}")

            # Remove rows where consumption is missing
            working_df = subsic_df.dropna(subset=[consumption_col]).copy()

            if working_df.empty:
                print(f"Skipped {category} | {subsic} | {period_name}: no data")
                continue

            customers = working_df["customerid"].dropna().unique()

            best_results = []

            # 10. LOOP THROUGH CUSTOMERS
            for customer in customers:
                customer_df = working_df[working_df["customerid"] == customer].copy()

                # Create monthly time series
                ts = customer_df.groupby("reportingmonth")[consumption_col].sum().sort_index()

                # Skip if too little data
                if len(ts) < 12:
                    continue

                split = int(len(ts) * 0.8)

                # Ensure enough test data
                if split <= 0 or split >= len(ts):
                    continue

                train = ts.iloc[:split]
                test = ts.iloc[split:]

                # Statistics for diagnosis
                record_count = len(ts)
                mean_val = ts.mean()
                std_val = ts.std()
                min_val = ts.min()
                max_val = ts.max()
                zero_count = (ts == 0).sum()
                negative_count = (ts < 0).sum()

                best_score = np.inf
                best_order = None

                # 11. GRID SEARCH FOR BEST ARIMA
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            try:
                                model = ARIMA(train, order=(p, d, q))
                                model_fit = model.fit()

                                preds = model_fit.forecast(steps=len(test))

                                # Use MAE because your data has negatives and zeros
                                mae = mean_absolute_error(test, preds)

                                if np.isfinite(mae) and mae < best_score:
                                    best_score = mae
                                    best_order = (p, d, q)

                            except Exception:
                                continue

                if best_order is None:
                    continue

                best_results.append({
                    "category": category,
                    "subsic": subsic,
                    "period": period_name,
                    "consumption_column": consumption_col,
                    "customerid": customer,
                    "best_p": best_order[0],
                    "best_d": best_order[1],
                    "best_q": best_order[2],
                    "best_MAE": best_score,
                    "records": record_count,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "zero_count": zero_count,
                    "negative_count": negative_count
                })

                print(
                    f"Done customer {customer} | "
                    f"{category} | {subsic} | {period_name} | "
                    f"Best ARIMA: {best_order} | MAE: {best_score:.4f}"
                )

            # If no successful customers for this group, skip
            if not best_results:
                print(f"No valid customer results for {category} | {subsic} | {period_name}")
                continue

            best_results_df = pd.DataFrame(best_results)

            # Store in master list
            all_best_results.append(best_results_df)

            # 12. FIND MOST COMMON PARAMETERS
            param_counts = (
                best_results_df
                .groupby(["category", "subsic", "period", "consumption_column", "best_p", "best_d", "best_q"])
                .size()
                .reset_index(name="count")
                .sort_values(by="count", ascending=False)
            )

            all_param_counts.append(param_counts)

            most_common_row = param_counts.iloc[0]
            common_order = (
                int(most_common_row["best_p"]),
                int(most_common_row["best_d"]),
                int(most_common_row["best_q"])
            )

            print(
                f"Most common ARIMA for {category} | {subsic} | {period_name}: "
                f"{common_order}"
            )

            # 13. TEST COMMON PARAMETERS ON ALL CUSTOMERS
            common_results = []

            for customer in customers:
                customer_df = working_df[working_df["customerid"] == customer].copy()

                ts = customer_df.groupby("reportingmonth")[consumption_col].sum().sort_index()

                if len(ts) < 12:
                    continue

                split = int(len(ts) * 0.8)

                if split <= 0 or split >= len(ts):
                    continue

                train = ts.iloc[:split]
                test = ts.iloc[split:]

                record_count = len(ts)
                mean_val = ts.mean()
                std_val = ts.std()
                min_val = ts.min()
                max_val = ts.max()
                zero_count = (ts == 0).sum()
                negative_count = (ts < 0).sum()

                try:
                    model = ARIMA(train, order=common_order)
                    model_fit = model.fit()
                    preds = model_fit.forecast(steps=len(test))

                    mae = mean_absolute_error(test, preds)
                    smape = calculate_smape(test, preds)

                    common_results.append({
                        "category": category,
                        "subsic": subsic,
                        "period": period_name,
                        "consumption_column": consumption_col,
                        "customerid": customer,
                        "common_p": common_order[0],
                        "common_d": common_order[1],
                        "common_q": common_order[2],
                        "MAE": mae,
                        "SMAPE": smape,
                        "records": record_count,
                        "mean": mean_val,
                        "std": std_val,
                        "min": min_val,
                        "max": max_val,
                        "zero_count": zero_count,
                        "negative_count": negative_count
                    })

                    print(
                        f"Common model tested | customer {customer} | "
                        f"{category} | {subsic} | {period_name} | "
                        f"ARIMA{common_order} | MAE: {mae:.4f} | SMAPE: {smape:.2f}"
                    )

                except Exception:
                    continue

            if common_results:
                common_results_df = pd.DataFrame(common_results)
                all_common_model_results.append(common_results_df)

# 14. COMBINE AND EXPORT EVERYTHING
if all_best_results:
    final_best_results = pd.concat(all_best_results, ignore_index=True)
else:
    final_best_results = pd.DataFrame()

if all_param_counts:
    final_param_counts = pd.concat(all_param_counts, ignore_index=True)
else:
    final_param_counts = pd.DataFrame()

if all_common_model_results:
    final_common_results = pd.concat(all_common_model_results, ignore_index=True)
else:
    final_common_results = pd.DataFrame()

best_results_file = os.path.join(output_folder, "all_categories_best_arima_results.xlsx")
param_counts_file = os.path.join(output_folder, "all_categories_arima_parameter_frequency.xlsx")
common_results_file = os.path.join(output_folder, "all_categories_common_arima_results.xlsx")

final_best_results.to_excel(best_results_file, index=False)
final_param_counts.to_excel(param_counts_file, index=False)
final_common_results.to_excel(common_results_file, index=False)

print("\n====================================")
print("ARIMA processing complete.")
print(f"Best parameter results saved to: {best_results_file}")
print(f"Parameter frequency saved to: {param_counts_file}")
print(f"Common model results saved to: {common_results_file}")
print("====================================")