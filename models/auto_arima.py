import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from Utils.metrics import mape, mae

warnings.filterwarnings("ignore")

# =========================================================
# 1. SETTINGS
# =========================================================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"
output_folder = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\results\forecast_plots"


os.makedirs(output_folder, exist_ok=True)

category = "Agriculture"
subsic = "Berries"

max_customers = 10
min_history = 24

consumption_columns = {
    "total": "totalconsumption",
    "peak": "peakconsumption",
    "standard": "standardconsumption",
    "offpeak": "offpeakconsumption"
}

# =========================================================
# ARIMA PARAMETERS ONLY: (p, d, q)
# =========================================================
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1, 2, 3, 4, 5]
q_values = [0, 1, 2, 3, 4, 5]

# =========================================================
# 2. LOAD DATA
# =========================================================
df = pd.read_excel(file_path)

df["reportingmonth"] = pd.to_datetime(df["reportingmonth"], errors="coerce")
df = df.dropna(subset=["reportingmonth", "category", "subsic", "customerid"])

df = df[
    (df["category"] == category) &
    (df["subsic"] == subsic)
].copy()

if df.empty:
    raise ValueError(f"No data found for category='{category}' and subsic='{subsic}'.")

consumption_columns = {
    period_name: col_name
    for period_name, col_name in consumption_columns.items()
    if col_name in df.columns
}

if not consumption_columns:
    raise ValueError("No valid consumption columns found.")

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def prepare_customer_series(dataframe, customer_id, consumption_col):
    customer_df = dataframe[dataframe["customerid"] == customer_id].copy()

    ts = (
        customer_df
        .groupby("reportingmonth")[consumption_col]
        .sum()
        .sort_index()
    )

    return ts


def train_test_split_series(ts, train_ratio=0.8):
    split = int(len(ts) * train_ratio)

    if split <= 0 or split >= len(ts):
        return None, None

    return ts.iloc[:split], ts.iloc[split:]


def get_series_stats(ts):
    return {
        "records": len(ts),
        "mean": ts.mean(),
        "std": ts.std(),
        "min": ts.min(),
        "max": ts.max(),
        "zero_count": (ts == 0).sum(),
        "negative_count": (ts < 0).sum()
    }


def run_manual_arima(train, test):
    best_mape = np.inf
    best_mae = np.inf
    best_order = None

    total_combinations = len(p_values) * len(d_values) * len(q_values)
    checked = 0

    print(f"Manual ARIMA combinations to test: {total_combinations}")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                checked += 1

                if checked % 50 == 0:
                    print(f"Checked {checked}/{total_combinations} ARIMA combinations...")

                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()

                    preds = model_fit.forecast(steps=len(test))

                    score_mape = mape(test, preds)
                    score_mae = mae(test, preds)

                    if np.isfinite(score_mape) and score_mape < best_mape:
                        best_mape = score_mape
                        best_mae = score_mae
                        best_order = (p, d, q)

                except Exception:
                    continue

    return best_order, best_mape, best_mae


def run_auto_arima(train, test):
    auto_model = auto_arima(
        train,
        seasonal=False,   # IMPORTANT: ARIMA only
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=None,
        max_d=5,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        trace=True
    )

    preds = auto_model.predict(n_periods=len(test))

    auto_mape = mape(test, preds)
    auto_mae = mae(test, preds)

    return auto_model.order, auto_mape, auto_mae


def evaluate_fixed_arima(train, test, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    preds = model_fit.forecast(steps=len(test))

    score_mape = mape(test, preds)
    score_mae = mae(test, preds)

    return score_mape, score_mae


# =========================================================
# 4. MASTER RESULT LISTS
# =========================================================
all_manual_best_results = []
all_manual_param_counts = []
all_manual_common_results = []

all_auto_best_results = []
all_auto_param_counts = []
all_auto_common_results = []

print(f"\n========== CATEGORY: {category} ==========")
print(f"----- SUBSIC: {subsic} -----")

# =========================================================
# 5. LOOP THROUGH CONSUMPTION TYPES
# =========================================================
for period_name, consumption_col in consumption_columns.items():

    print(f"\n### Running period: {period_name} | column: {consumption_col}")

    working_df = df.dropna(subset=[consumption_col]).copy()

    if working_df.empty:
        print(f"Skipped {period_name}: no data")
        continue

    customers = working_df["customerid"].dropna().unique()[:max_customers]

    manual_best_results = []
    auto_best_results = []
    valid_customer_series = {}

    # =====================================================
    # A. BEST ARIMA PER CUSTOMER
    # =====================================================
    for i, customer in enumerate(customers, start=1):

        print(f"\nProcessing customer {i}/{len(customers)}: {customer}")

        ts = prepare_customer_series(working_df, customer, consumption_col)

        if len(ts) < min_history:
            print(f"Skipped customer {customer}: less than {min_history} records")
            continue

        train, test = train_test_split_series(ts)

        if train is None or test is None:
            print(f"Skipped customer {customer}: invalid split")
            continue

        stats = get_series_stats(ts)
        valid_customer_series[customer] = (train, test, stats)

        # ---------------- MANUAL ARIMA ----------------
        print(f"Running MANUAL ARIMA for customer {customer}...")

        best_order, best_mape, best_mae = run_manual_arima(train, test)

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
                "best_MAPE": best_mape,
                "best_MAE": best_mae,
                **stats
            })

            print(
                f"Manual ARIMA | customer {customer} | "
                f"Best: {best_order} | "
                f"MAPE: {best_mape:.4f} | MAE: {best_mae:.4f}"
            )
        else:
            print(f"Manual ARIMA failed for customer {customer}")

        # ---------------- AUTO ARIMA ----------------
        print(f"Running AUTO ARIMA for customer {customer}...")

        try:
            auto_order, auto_mape, auto_mae = run_auto_arima(train, test)

            auto_best_results.append({
                "category": category,
                "subsic": subsic,
                "period": period_name,
                "consumption_column": consumption_col,
                "customerid": customer,
                "best_p": auto_order[0],
                "best_d": auto_order[1],
                "best_q": auto_order[2],
                "best_MAPE": auto_mape,
                "best_MAE": auto_mae,
                **stats
            })

            print(
                f"Auto ARIMA | customer {customer} | "
                f"Best: {auto_order} | "
                f"MAPE: {auto_mape:.4f} | MAE: {auto_mae:.4f}"
            )

        except Exception as e:
            print(f"Auto ARIMA failed for customer {customer}: {e}")

    # =====================================================
    # B. MANUAL PARAMETER FREQUENCY + COMMON MODEL
    # =====================================================
    if manual_best_results:
        manual_best_df = pd.DataFrame(manual_best_results)
        all_manual_best_results.append(manual_best_df)

        manual_param_counts = (
            manual_best_df
            .groupby([
                "category", "subsic", "period", "consumption_column",
                "best_p", "best_d", "best_q"
            ])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )

        all_manual_param_counts.append(manual_param_counts)

        top_row = manual_param_counts.iloc[0]

        common_order = (
            int(top_row["best_p"]),
            int(top_row["best_d"]),
            int(top_row["best_q"])
        )

        print(
            f"\nMost common MANUAL ARIMA for {category} | {subsic} | {period_name}: "
            f"{common_order}"
        )

        common_results = []

        for customer, (train, test, stats) in valid_customer_series.items():
            try:
                score_mape, score_mae = evaluate_fixed_arima(
                    train,
                    test,
                    common_order
                )

                common_results.append({
                    "category": category,
                    "subsic": subsic,
                    "period": period_name,
                    "consumption_column": consumption_col,
                    "customerid": customer,
                    "common_p": common_order[0],
                    "common_d": common_order[1],
                    "common_q": common_order[2],
                    "MAPE": score_mape,
                    "MAE": score_mae,
                    **stats
                })

            except Exception:
                continue

        if common_results:
            all_manual_common_results.append(pd.DataFrame(common_results))

    # =====================================================
    # C. AUTO PARAMETER FREQUENCY + COMMON MODEL
    # =====================================================
    if auto_best_results:
        auto_best_df = pd.DataFrame(auto_best_results)
        all_auto_best_results.append(auto_best_df)

        auto_param_counts = (
            auto_best_df
            .groupby([
                "category", "subsic", "period", "consumption_column",
                "best_p", "best_d", "best_q"
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

        print(
            f"\nMost common AUTO ARIMA for {category} | {subsic} | {period_name}: "
            f"{auto_common_order}"
        )

        auto_common_results = []

        for customer, (train, test, stats) in valid_customer_series.items():
            try:
                score_mape, score_mae = evaluate_fixed_arima(
                    train,
                    test,
                    auto_common_order
                )

                auto_common_results.append({
                    "category": category,
                    "subsic": subsic,
                    "period": period_name,
                    "consumption_column": consumption_col,
                    "customerid": customer,
                    "common_p": auto_common_order[0],
                    "common_d": auto_common_order[1],
                    "common_q": auto_common_order[2],
                    "MAPE": score_mape,
                    "MAE": score_mae,
                    **stats
                })

            except Exception:
                continue

        if auto_common_results:
            all_auto_common_results.append(pd.DataFrame(auto_common_results))

# =========================================================
# 6. EXPORT RESULTS
# =========================================================
safe_category = category.lower().replace(" ", "_")
safe_subsic = subsic.lower().replace(" ", "_")

final_manual_best = (
    pd.concat(all_manual_best_results, ignore_index=True)
    if all_manual_best_results else pd.DataFrame()
)

final_manual_param_counts = (
    pd.concat(all_manual_param_counts, ignore_index=True)
    if all_manual_param_counts else pd.DataFrame()
)

final_manual_common = (
    pd.concat(all_manual_common_results, ignore_index=True)
    if all_manual_common_results else pd.DataFrame()
)

final_auto_best = (
    pd.concat(all_auto_best_results, ignore_index=True)
    if all_auto_best_results else pd.DataFrame()
)

final_auto_param_counts = (
    pd.concat(all_auto_param_counts, ignore_index=True)
    if all_auto_param_counts else pd.DataFrame()
)

final_auto_common = (
    pd.concat(all_auto_common_results, ignore_index=True)
    if all_auto_common_results else pd.DataFrame()
)

manual_best_file = os.path.join(output_folder, f"{safe_category}_{safe_subsic}_best_arima_results.xlsx")
manual_param_file = os.path.join(output_folder, f"{safe_category}_{safe_subsic}_arima_parameter_frequency.xlsx")
manual_common_file = os.path.join(output_folder, f"{safe_category}_{safe_subsic}_common_arima_results.xlsx")

auto_best_file = os.path.join(output_folder, f"{safe_category}_{safe_subsic}_best_auto_arima_results.xlsx")
auto_param_file = os.path.join(output_folder, f"{safe_category}_{safe_subsic}_auto_arima_parameter_frequency.xlsx")
auto_common_file = os.path.join(output_folder, f"{safe_category}_{safe_subsic}_common_auto_arima_results.xlsx")

final_manual_best.to_excel(manual_best_file, index=False)
final_manual_param_counts.to_excel(manual_param_file, index=False)
final_manual_common.to_excel(manual_common_file, index=False)

final_auto_best.to_excel(auto_best_file, index=False)
final_auto_param_counts.to_excel(auto_param_file, index=False)
final_auto_common.to_excel(auto_common_file, index=False)

print("\n====================================")
print("ARIMA processing complete.")
print(f"Manual best results saved to: {manual_best_file}")
print(f"Manual parameter frequency saved to: {manual_param_file}")
print(f"Manual common results saved to: {manual_common_file}")
print(f"Auto best results saved to: {auto_best_file}")
print(f"Auto parameter frequency saved to: {auto_param_file}")
print(f"Auto common results saved to: {auto_common_file}")
print("====================================")