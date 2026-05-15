import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

from Utils.metrics import mape, mae

warnings.filterwarnings("ignore")


# =========================================================
# 1. SETTINGS
# =========================================================
file_path = r"C:\Users\MauduH\OneDrive - Eskom Holdings SOC Ltd\Python Projects\Parameter-Optimisation\data\Input Dataset - cleanedFortrack_dataset.xlsx"
output_folder = r"C:\Users\MauduH\Documents\Migael\SARIMA_Outputs"

os.makedirs(output_folder, exist_ok=True)

category = "Agriculture"
subsic = "Berries"
seasonal_period = 12
max_customers = 10
min_history = 24

consumption_columns = {
    "total": "totalconsumption",
    "peak": "peakconsumption",
    "standard": "standardconsumption",
    "offpeak": "offpeakconsumption"
}

# =========================================================
# SARIMA PARAMETERS

# =========================================================
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1, 2, 3, 4, 5]
q_values = [0, 1, 2, 3, 4, 5]

P_values = [0, 1, 2, 3, 4, 5]
D_values = [0, 1, 2, 3, 4, 5]
Q_values = [0, 1, 2, 3, 4, 5]


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
    raise ValueError(
        "None of the consumption columns were found in the dataset. "
        "Please update the consumption_columns dictionary."
    )


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


def run_manual_sarima(train, test, seasonal_period):
    best_mape = np.inf
    best_mae = np.inf
    best_order = None
    best_seasonal_order = None
    total_combinations = (
        len(p_values) * len(d_values) * len(q_values) *
        len(P_values) * len(D_values) * len(Q_values)
    )
    checked = 0

    print(f"Manual SARIMA combinations to test: {total_combinations}")

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            checked += 1

                            if checked % 500 == 0:
                                print(
                                    f"Checked {checked}/{total_combinations} "
                                    f"manual combinations..."
                                )

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

    return best_order, best_seasonal_order, best_mape, best_mae


def run_auto_sarima(train, test, seasonal_period):
    auto_model = auto_arima(
        train,
        seasonal=True,
        m=seasonal_period,
        start_p=0,
        start_d=0,
        start_q=0,
        max_p=5,
        max_d=5,
        max_q=5,
        start_P=0,
        start_D=0,
        start_Q=0,
        max_P=5,
        max_D=5,
        max_Q=5,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        trace=True
    )

    preds = auto_model.predict(n_periods=len(test))
    auto_best_mape = mape(test, preds)
    auto_best_mae = mae(test, preds)

    return auto_model.order, auto_model.seasonal_order, auto_best_mape, auto_best_mae


def evaluate_fixed_model(train, test, order, seasonal_order):
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)
    preds = model_fit.forecast(steps=len(test))

    score_mape = mape(test, preds)
    score_mae = mae(test, preds)

    return score_mape, score_mae


# =========================================================
# 4. MASTER RESULT LISTS
# =========================================================
all_best_results = []
all_param_counts = []
all_common_model_results = []

all_auto_best_results = []
all_auto_param_counts = []
all_auto_common_model_results = []


print(f"\n========== CATEGORY: {category} ==========")
print(f"----- SUBSIC: {subsic} -----")


# =========================================================
# 5. LOOP THROUGH CONSUMPTION TYPES
# =========================================================
for period_name, consumption_col in consumption_columns.items():
    print(f"\n### Running period: {period_name} | column: {consumption_col}")

    working_df = df.dropna(subset=[consumption_col]).copy()

    if working_df.empty:
        print(f"Skipped {category} | {subsic} | {period_name}: no data")
        continue

    customers = working_df["customerid"].dropna().unique()[:max_customers]

    manual_best_results = []
    auto_best_results = []
    valid_customer_series = {}

    # -----------------------------------------------------
    # A. RUN BEST MODEL SEARCH PER CUSTOMER
    # -----------------------------------------------------
    for i, customer in enumerate(customers, start=1):
        print(f"\nProcessing customer {i}/{len(customers)}: {customer}")

        ts = prepare_customer_series(working_df, customer, consumption_col)

        if len(ts) < min_history:
            print(f"Skipped customer {customer}: less than {min_history} records")
            continue

        train, test = train_test_split_series(ts)

        if train is None or test is None:
            print(f"Skipped customer {customer}: invalid train/test split")
            continue

        stats = get_series_stats(ts)
        valid_customer_series[customer] = (train, test, stats)

        # ---------------- MANUAL SARIMA ----------------
        print(f"Running MANUAL SARIMA for customer {customer}...")
        best_order, best_seasonal_order, best_mape, best_mae = run_manual_sarima(
            train, test, seasonal_period
        )

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
                **stats
            })

            print(
                f"Manual SARIMA | customer {customer} | "
                f"Best: {best_order} x {best_seasonal_order} | "
                f"MAPE: {best_mape:.4f} | MAE: {best_mae:.4f}"
            )
        else:
            print(f"Manual SARIMA failed for customer {customer}")

        # ---------------- AUTO SARIMA ----------------
        print(f"Running AUTO SARIMA for customer {customer}...")
        try:
            auto_order, auto_seasonal_order, auto_best_mape, auto_best_mae = run_auto_sarima(
                train, test, seasonal_period
            )

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
                **stats
            })

            print(
                f"Auto SARIMA | customer {customer} | "
                f"Best: {auto_order} x {auto_seasonal_order} | "
                f"MAPE: {auto_best_mape:.4f} | MAE: {auto_best_mae:.4f}"
            )

        except Exception as e:
            print(f"Auto SARIMA failed for customer {customer}: {e}")

    # -----------------------------------------------------
    # B. MANUAL PARAM COUNTS + COMMON MODEL
    # -----------------------------------------------------
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
            f"\nMost common MANUAL SARIMA for {category} | {subsic} | {period_name}: "
            f"{common_order} x {common_seasonal_order}"
        )

        manual_common_results = []

        for customer, (train, test, stats) in valid_customer_series.items():
            try:
                score_mape, score_mae = evaluate_fixed_model(
                    train, test, common_order, common_seasonal_order
                )

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
                    **stats
                })

            except Exception:
                continue

        if manual_common_results:
            all_common_model_results.append(pd.DataFrame(manual_common_results))

    # -----------------------------------------------------
    # C. AUTO PARAM COUNTS + COMMON MODEL
    # -----------------------------------------------------
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
            f"\nMost common AUTO SARIMA for {category} | {subsic} | {period_name}: "
            f"{auto_common_order} x {auto_common_seasonal_order}"
        )

        auto_common_results = []

        for customer, (train, test, stats) in valid_customer_series.items():
            try:
                score_mape, score_mae = evaluate_fixed_model(
                    train, test, auto_common_order, auto_common_seasonal_order
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
                    "common_P": auto_common_seasonal_order[0],
                    "common_D": auto_common_seasonal_order[1],
                    "common_Q": auto_common_seasonal_order[2],
                    "seasonal_period": auto_common_seasonal_order[3],
                    "MAPE": score_mape,
                    "MAE": score_mae,
                    **stats
                })

            except Exception:
                continue

        if auto_common_results:
            all_auto_common_model_results.append(pd.DataFrame(auto_common_results))


# =========================================================
# 6. COMBINE AND EXPORT EVERYTHING - MANUAL SARIMA
# =========================================================
final_best_results = pd.concat(all_best_results, ignore_index=True) if all_best_results else pd.DataFrame()
final_param_counts = pd.concat(all_param_counts, ignore_index=True) if all_param_counts else pd.DataFrame()
final_common_results = pd.concat(all_common_model_results, ignore_index=True) if all_common_model_results else pd.DataFrame()

manual_best_file = os.path.join(output_folder, "agriculture_cereal_best_sarima_results.xlsx")
manual_param_file = os.path.join(output_folder, "agriculture_cereal_sarima_parameter_frequency.xlsx")
manual_common_file = os.path.join(output_folder, "agriculture_cereal_common_sarima_results.xlsx")

final_best_results.to_excel(manual_best_file, index=False)
final_param_counts.to_excel(manual_param_file, index=False)
final_common_results.to_excel(manual_common_file, index=False)


# =========================================================
# 7. COMBINE AND EXPORT EVERYTHING - AUTO SARIMA
# =========================================================
final_auto_best_results = pd.concat(all_auto_best_results, ignore_index=True) if all_auto_best_results else pd.DataFrame()
final_auto_param_counts = pd.concat(all_auto_param_counts, ignore_index=True) if all_auto_param_counts else pd.DataFrame()
final_auto_common_results = pd.concat(all_auto_common_model_results, ignore_index=True) if all_auto_common_model_results else pd.DataFrame()

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