
# from itertools import product
# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# from Utils.data import make_customer_monthly_series, split_series_time
# from Utils.metrics import mape


# def run_arima_parameter_search(
#     customer_df: pd.DataFrame,
#     customer_id,
#     param_grid: dict | None = None,
# ):
#     if param_grid is None:
#         param_grid = {
#             "p": [0, 1, 2],
#             "d": [0, 1],
#             "q": [0, 1, 2],
#         }

#     ts = make_customer_monthly_series(customer_df)

#     if len(ts) < 12:
#         return None

#     train_ts, test_ts = split_series_time(ts, test_size=0.2)

#     best_mape = float("inf")
#     best_params = None

#     for p, d, q in product(param_grid["p"], param_grid["d"], param_grid["q"]):
#         try:
#             model = ARIMA(train_ts, order=(p, d, q)).fit()
#             preds = model.forecast(steps=len(test_ts))
#             score = mape(test_ts, preds)

#             if score < best_mape:
#                 best_mape = score
#                 best_params = {"p": p, "d": d, "q": q}
#         except Exception:
#             continue

#     return {
#         "customerid": customer_id,
#         "best_params": best_params,
#         "mape": best_mape,
#     }


# def run_sarima_parameter_search(
#     customer_df: pd.DataFrame,
#     customer_id,
#     param_grid: dict | None = None,
# ):
#     if param_grid is None:
#         param_grid = {
#             "p": [0, 1],
#             "d": [0, 1],
#             "q": [0, 1],
#             "P": [0, 1],
#             "D": [0, 1],
#             "Q": [0, 1],
#             "s": [12],
#         }

#     ts = make_customer_monthly_series(customer_df)

#     if len(ts) < 24:
#         return None

#     train_ts, test_ts = split_series_time(ts, test_size=0.2)

#     best_mape = float("inf")
#     best_params = None

#     for p, d, q, P, D, Q, s in product(
#         param_grid["p"],
#         param_grid["d"],
#         param_grid["q"],
#         param_grid["P"],
#         param_grid["D"],
#         param_grid["Q"],
#         param_grid["s"],
#     ):
#         try:
#             model = SARIMAX(
#                 train_ts,
#                 order=(p, d, q),
#                 seasonal_order=(P, D, Q, s),
#                 enforce_stationarity=False,
#                 enforce_invertibility=False,
#             ).fit(disp=False)

#             preds = model.forecast(steps=len(test_ts))
#             score = mape(test_ts, preds)

#             if score < best_mape:
#                 best_mape = score
#                 best_params = {
#                     "p": p, "d": d, "q": q,
#                     "P": P, "D": D, "Q": Q, "s": s,
#                 }
#         except Exception:
#             continue

#     return {
#         "customerid": customer_id,
#         "best_params": best_params,
#         "mape": best_mape,
#     }

import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

from Utils.metrics import mape, mae

warnings.filterwarnings("ignore")

# 1. LOAD DATA
file_path = r"C:\Users\MauduH\Documents\Migael\cleanedFortrack_dataset.xlsx"
output_folder = r"C:\Users\MauduH\Documents\Migael\SARIMA_Outputs"

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

# 4. SARIMA PARAMETERS TO TEST
p_values = [0, 1]
d_values = [0, 1]
q_values = [0, 1]

P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]

seasonal_period = 12

# 5. MASTER RESULT LISTS
all_best_results = []
all_param_counts = []
all_common_model_results = []

# 6. LOOP THROUGH ALL CATEGORIES
for category in sorted(df["category"].dropna().unique()):
    category_df = df[df["category"] == category].copy()

    subsics = sorted(category_df["subsic"].dropna().unique())

    print(f"\n========== CATEGORY: {category} ==========")

    # 7. LOOP THROUGH SUBSICS
    for subsic in subsics:
        subsic_df = category_df[category_df["subsic"] == subsic].copy()

        print(f"\n----- SUBSIC: {subsic} -----")

        # 8. LOOP THROUGH CONSUMPTION TYPES
        for period_name, consumption_col in consumption_columns.items():
            print(f"\n### Running period: {period_name} | column: {consumption_col}")

            # Remove rows where consumption is missing
            working_df = subsic_df.dropna(subset=[consumption_col]).copy()

            if working_df.empty:
                print(f"Skipped {category} | {subsic} | {period_name}: no data")
                continue

            customers = working_df["customerid"].dropna().unique()

            best_results = []

            # 9. LOOP THROUGH CUSTOMERS
            for customer in customers:
                customer_df = working_df[working_df["customerid"] == customer].copy()

                # Create monthly time series
                ts = customer_df.groupby("reportingmonth")[consumption_col].sum().sort_index()

                # Skip if too little data
                if len(ts) < 24:
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

                best_mape = np.inf
                best_mae = np.inf
                best_order = None
                best_seasonal_order = None

                # 10. GRID SEARCH FOR BEST SARIMA
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            for P in P_values:
                                for D in D_values:
                                    for Q in Q_values:
                                        try:
                                            model = SARIMAX(
                                                train,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, seasonal_period),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False
                                            )
                                            model_fit = model.fit(disp=False)

                                            preds = model_fit.forecast(steps=len(test))

                                            score_mape = mape(test, preds)
                                            score_mae = mae(test, preds)

                                            if np.isfinite(score_mape) and score_mape < best_mape:
                                                best_mape = score_mape
                                                best_mae = score_mae
                                                best_order = (p, d, q)
                                                best_seasonal_order = (P, D, Q, seasonal_period)

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
                    "best_P": best_seasonal_order[0],
                    "best_D": best_seasonal_order[1],
                    "best_Q": best_seasonal_order[2],
                    "seasonal_period": best_seasonal_order[3],
                    "best_MAPE": best_mape,
                    "best_MAE": best_mae,
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
                    f"Best SARIMA: {best_order} x {best_seasonal_order} | "
                    f"MAPE: {best_mape:.4f} | MAE: {best_mae:.4f}"
                )

            # If no successful customers for this group, skip
            if not best_results:
                print(f"No valid customer results for {category} | {subsic} | {period_name}")
                continue

            best_results_df = pd.DataFrame(best_results)

            # Store in master list
            all_best_results.append(best_results_df)

            # 11. FIND MOST COMMON PARAMETERS
            param_counts = (
                best_results_df
                .groupby([
                    "category", "subsic", "period", "consumption_column",
                    "best_p", "best_d", "best_q",
                    "best_P", "best_D", "best_Q", "seasonal_period"
                ])
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
            common_seasonal_order = (
                int(most_common_row["best_P"]),
                int(most_common_row["best_D"]),
                int(most_common_row["best_Q"]),
                int(most_common_row["seasonal_period"])
            )

            print(
                f"Most common SARIMA for {category} | {subsic} | {period_name}: "
                f"{common_order} x {common_seasonal_order}"
            )

            # 12. TEST COMMON PARAMETERS ON ALL CUSTOMERS
            common_results = []

            for customer in customers:
                customer_df = working_df[working_df["customerid"] == customer].copy()

                ts = customer_df.groupby("reportingmonth")[consumption_col].sum().sort_index()

                if len(ts) < 24:
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
                    model = SARIMAX(
                        train,
                        order=common_order,
                        seasonal_order=common_seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_fit = model.fit(disp=False)
                    preds = model_fit.forecast(steps=len(test))

                    score_mape = mape(test, preds)
                    score_mae = mae(test, preds)

                    common_results.append({
                        "category": category,
                        "subsic": subsic,
                        "period": period_name,
                        "consumption_column": consumption_col,
                        "customerid": customer,
                        "common_p": common_order[0],
                        "common_d": common_order[1],
                        "common_q": common_order[2],
                        "common_P": common_seasonal_order[0],
                        "common_D": common_seasonal_order[1],
                        "common_Q": common_seasonal_order[2],
                        "seasonal_period": common_seasonal_order[3],
                        "MAPE": score_mape,
                        "MAE": score_mae,
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
                        f"SARIMA{common_order}x{common_seasonal_order} | "
                        f"MAPE: {score_mape:.4f} | MAE: {score_mae:.4f}"
                    )

                except Exception:
                    continue

            if common_results:
                common_results_df = pd.DataFrame(common_results)
                all_common_model_results.append(common_results_df)

# 13. COMBINE AND EXPORT EVERYTHING
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

best_results_file = os.path.join(output_folder, "all_categories_best_sarima_results.xlsx")
param_counts_file = os.path.join(output_folder, "all_categories_sarima_parameter_frequency.xlsx")
common_results_file = os.path.join(output_folder, "all_categories_common_sarima_results.xlsx")

final_best_results.to_excel(best_results_file, index=False)
final_param_counts.to_excel(param_counts_file, index=False)
final_common_results.to_excel(common_results_file, index=False)

print("\n====================================")
print("SARIMA processing complete.")
print(f"Best parameter results saved to: {best_results_file}")
print(f"Parameter frequency saved to: {param_counts_file}")
print(f"Common model results saved to: {common_results_file}")
print("====================================")