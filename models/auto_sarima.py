import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

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

# 3. FILTER ONLY AGRICULTURE + CEREAL
df = df[
    (df["category"] == "Agriculture") &
    (df["subsic"] == "Cereal")
].copy()

if df.empty:
    raise ValueError("No data found for category='Agriculture' and subsic='Cereal'.")

# 4. CONSUMPTION COLUMNS
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

# 5. MANUAL SARIMA PARAMETERS TO TEST: 0 TO 5
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1, 2, 3, 4, 5]
q_values = [0, 1, 2, 3, 4, 5]

P_values = [0, 1, 2, 3, 4, 5]
D_values = [0, 1, 2, 3, 4, 5]
Q_values = [0, 1, 2, 3, 4, 5]

seasonal_period = 12

# 6. MASTER RESULT LISTS - MANUAL SARIMA
all_best_results = []
all_param_counts = []
all_common_model_results = []

# 7. MASTER RESULT LISTS - AUTO SARIMA
all_auto_best_results = []
all_auto_param_counts = []
all_auto_common_model_results = []

category = "Agriculture"
subsic = "Cereal"

print(f"\n========== CATEGORY: {category} ==========")
print(f"----- SUBSIC: {subsic} -----")

# 8. LOOP THROUGH CONSUMPTION TYPES
for period_name, consumption_col in consumption_columns.items():
    print(f"\n### Running period: {period_name} | column: {consumption_col}")

    working_df = df.dropna(subset=[consumption_col]).copy()

    if working_df.empty:
        print(f"Skipped {category} | {subsic} | {period_name}: no data")
        continue

    customers = working_df["customerid"].dropna().unique()[:10]

    manual_best_results = []
    auto_best_results = []

    # 9. LOOP THROUGH CUSTOMERS
    for customer in customers:
        customer_df = working_df[working_df["customerid"] == customer].copy()

        ts = customer_df.groupby("reportingmonth")[consumption_col].sum().sort_index()

        # Need more history for SARIMA
        if len(ts) < 24:
            continue

        split = int(len(ts) * 0.8)

        if split <= 0 or split >= len(ts):
            continue

        train = ts.iloc[:split]
        test = ts.iloc[split:]

        # Statistics
        record_count = len(ts)
        mean_val = ts.mean()
        std_val = ts.std()
        min_val = ts.min()
        max_val = ts.max()
        zero_count = (ts == 0).sum()
        negative_count = (ts < 0).sum()

        # =========================================================
        # 10. MANUAL GRID SEARCH SARIMA
        # =========================================================
        best_mape = np.inf
        best_mae = np.inf
        best_order = None
        best_seasonal_order = None

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

        if best_order is not None:
            manual_best_results.append({
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
                f"Manual SARIMA | customer {customer} | "
                f"{period_name} | Best: {best_order} x {best_seasonal_order} | "
                f"MAPE: {best_mape:.4f} | MAE: {best_mae:.4f}"
            )

        # =========================================================
        # 11. AUTO-SARIMA
        # =========================================================
        auto_best_mape = np.inf
        auto_best_mae = np.inf
        auto_order = None
        auto_seasonal_order = None

        try:
            auto_model = auto_arima(
                train,
                seasonal=True,
                m=seasonal_period,
                start_p=0,
                start_q=0,
                max_p=5,
                max_d=5,
                max_q=5,
                start_P=0,
                start_Q=0,
                max_P=5,
                max_D=5,
                max_Q=5,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                trace=False
            )

            auto_order = auto_model.order
            auto_seasonal_order = auto_model.seasonal_order

            preds = auto_model.predict(n_periods=len(test))

            auto_best_mape = mape(test, preds)
            auto_best_mae = mae(test, preds)

            auto_best_results.append({
                "category": category,
                "subsic": subsic,
                "period": period_name,
                "consumption_column": consumption_col,
                "customerid": customer,
                "best_p": auto_order[0],
                "best_d": auto_order[1],
                "best_q": auto_order[2],
                "best_P": auto_seasonal_order[0],
                "best_D": auto_seasonal_order[1],
                "best_Q": auto_seasonal_order[2],
                "seasonal_period": auto_seasonal_order[3],
                "best_MAPE": auto_best_mape,
                "best_MAE": auto_best_mae,
                "records": record_count,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "zero_count": zero_count,
                "negative_count": negative_count
            })

            print(
                f"Auto SARIMA | customer {customer} | "
                f"{period_name} | Best: {auto_order} x {auto_seasonal_order} | "
                f"MAPE: {auto_best_mape:.4f} | MAE: {auto_best_mae:.4f}"
            )

        except Exception:
            continue

    # =========================================================
    # 12. MANUAL SARIMA PARAM COUNTS + COMMON MODEL
    # =========================================================
    if manual_best_results:
        manual_best_df = pd.DataFrame(manual_best_results)
        all_best_results.append(manual_best_df)

        manual_param_counts = (
            manual_best_df
            .groupby([
                "category", "subsic", "period", "consumption_column",
                "best_p", "best_d", "best_q",
                "best_P", "best_D", "best_Q", "seasonal_period"
            ])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )

        all_param_counts.append(manual_param_counts)

        top_row = manual_param_counts.iloc[0]
        common_order = (
            int(top_row["best_p"]),
            int(top_row["best_d"]),
            int(top_row["best_q"])
        )
        common_seasonal_order = (
            int(top_row["best_P"]),
            int(top_row["best_D"]),
            int(top_row["best_Q"]),
            int(top_row["seasonal_period"])
        )

        print(
            f"Most common MANUAL SARIMA for {category} | {subsic} | {period_name}: "
            f"{common_order} x {common_seasonal_order}"
        )

        manual_common_results = []

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

                manual_common_results.append({
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

            except Exception:
                continue

        if manual_common_results:
            all_common_model_results.append(pd.DataFrame(manual_common_results))

    # =========================================================
    # 13. AUTO-SARIMA PARAM COUNTS + COMMON MODEL
    # =========================================================
    if auto_best_results:
        auto_best_df = pd.DataFrame(auto_best_results)
        all_auto_best_results.append(auto_best_df)

        auto_param_counts = (
            auto_best_df
            .groupby([
                "category", "subsic", "period", "consumption_column",
                "best_p", "best_d", "best_q",
                "best_P", "best_D", "best_Q", "seasonal_period"
            ])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )

        all_auto_param_counts.append(auto_param_counts)

        top_row = auto_param_counts.iloc[0]
        auto_common_order = (
            int(top_row["best_p"]),
            int(top_row["best_d"]),
            int(top_row["best_q"])
        )
        auto_common_seasonal_order = (
            int(top_row["best_P"]),
            int(top_row["best_D"]),
            int(top_row["best_Q"]),
            int(top_row["seasonal_period"])
        )

        print(
            f"Most common AUTO SARIMA for {category} | {subsic} | {period_name}: "
            f"{auto_common_order} x {auto_common_seasonal_order}"
        )

        auto_common_results = []

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
                    order=auto_common_order,
                    seasonal_order=auto_common_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
                preds = model_fit.forecast(steps=len(test))

                score_mape = mape(test, preds)
                score_mae = mae(test, preds)

                auto_common_results.append({
                    "category": category,
                    "subsic": subsic,
                    "period": period_name,
                    "consumption_column": consumption_col,
                    "customerid": customer,
                    "common_p": auto_common_order[0],
                    "common_d": auto_common_order[1],
                    "common_q": auto_common_order[2],
                    "common_P": auto_common_seasonal_order[0],
                    "common_D": auto_common_seasonal_order[1],
                    "common_Q": auto_common_seasonal_order[2],
                    "seasonal_period": auto_common_seasonal_order[3],
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

            except Exception:
                continue

        if auto_common_results:
            all_auto_common_model_results.append(pd.DataFrame(auto_common_results))

# 14. COMBINE AND EXPORT EVERYTHING - MANUAL SARIMA
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

manual_best_file = os.path.join(output_folder, "agriculture_cereal_best_sarima_results.xlsx")
manual_param_file = os.path.join(output_folder, "agriculture_cereal_sarima_parameter_frequency.xlsx")
manual_common_file = os.path.join(output_folder, "agriculture_cereal_common_sarima_results.xlsx")

final_best_results.to_excel(manual_best_file, index=False)
final_param_counts.to_excel(manual_param_file, index=False)
final_common_results.to_excel(manual_common_file, index=False)

# 15. COMBINE AND EXPORT EVERYTHING - AUTO SARIMA
if all_auto_best_results:
    final_auto_best_results = pd.concat(all_auto_best_results, ignore_index=True)
else:
    final_auto_best_results = pd.DataFrame()

if all_auto_param_counts:
    final_auto_param_counts = pd.concat(all_auto_param_counts, ignore_index=True)
else:
    final_auto_param_counts = pd.DataFrame()

if all_auto_common_model_results:
    final_auto_common_results = pd.concat(all_auto_common_model_results, ignore_index=True)
else:
    final_auto_common_results = pd.DataFrame()

auto_best_file = os.path.join(output_folder, "agriculture_cereal_best_auto_sarima_results.xlsx")
auto_param_file = os.path.join(output_folder, "agriculture_cereal_auto_sarima_parameter_frequency.xlsx")
auto_common_file = os.path.join(output_folder, "agriculture_cereal_common_auto_sarima_results.xlsx")

final_auto_best_results.to_excel(auto_best_file, index=False)
final_auto_param_counts.to_excel(auto_param_file, index=False)
final_auto_common_results.to_excel(auto_common_file, index=False)

print("\n====================================")
print("SARIMA processing complete.")
print(f"Manual best results saved to: {manual_best_file}")
print(f"Manual parameter frequency saved to: {manual_param_file}")
print(f"Manual common results saved to: {manual_common_file}")
print(f"Auto best results saved to: {auto_best_file}")
print(f"Auto parameter frequency saved to: {auto_param_file}")
print(f"Auto common results saved to: {auto_common_file}")
print("====================================")