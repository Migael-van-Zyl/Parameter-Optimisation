import models.search_file

# from pathlib import Path
# import pandas as pd

# from Utils.data import load_dataset
# from Utils.data import filter_data
# from models.rf import run_random_forest_parameter_search
# from models.xgb import run_xgboost_parameter_search
# from models.arima_sarima import (
#     run_arima_parameter_search,
#     run_sarima_parameter_search,
# )

# BASE_DIR = Path(__file__).resolve().parent
# RESULTS_DIR = BASE_DIR / "results"
# RESULTS_DIR.mkdir(exist_ok=True)

# # ---------------------------
# # SETTINGS
# # ---------------------------
# DATA_FILE = "data/cleanedFortrack_dataset.xlsx"

# CATEGORY_FILTER = "Agriculture"      # Example: "Agriculture"
# SUBSIC_FILTER = "Cereal"        # Example: "Wine"
# MIN_POINTS_PER_CUSTOMER = 12

# RUN_RF = True
# RUN_XGB = True
# RUN_ARIMA = True
# RUN_SARIMA = True

# # ---------------------------
# # LOAD DATA
# # ---------------------------
# df = load_dataset(BASE_DIR, DATA_FILE)

# if CATEGORY_FILTER is not None:
#     df = df[df["category"] == CATEGORY_FILTER]

# if SUBSIC_FILTER is not None:
#     df = df[df["subsic"] == SUBSIC_FILTER]

# # ---------------------------
# # LOOP THROUGH SUBSICS
# # ---------------------------
# all_results = []

# for subsic in sorted(df["subsic"].dropna().unique()):
#     subsic_df = filter_data(df, subsic=subsic)

#     customers = subsic_df["customerid"].dropna().unique()

#     rf_customer_results = []
#     xgb_customer_results = []
#     arima_customer_results = []
#     sarima_customer_results = []

#     print(f"\nProcessing subSIC: {subsic} | Customers: {len(customers)}")

#     for customer_id in customers:
#         customer_df = filter_data(subsic_df, customer_id=customer_id)

#         if len(customer_df) < MIN_POINTS_PER_CUSTOMER:
#             continue

#         if RUN_RF:
#             rf_result = run_random_forest_parameter_search(customer_df, customer_id)
#             if rf_result is not None:
#                 rf_customer_results.append(rf_result)

#         if RUN_XGB:
#             xgb_result = run_xgboost_parameter_search(customer_df, customer_id)
#             if xgb_result is not None:
#                 xgb_customer_results.append(xgb_result)

#         if RUN_ARIMA:
#             arima_result = run_arima_parameter_search(customer_df, customer_id)
#             if arima_result is not None:
#                 arima_customer_results.append(arima_result)

#         if RUN_SARIMA:
#             sarima_result = run_sarima_parameter_search(customer_df, customer_id)
#             if sarima_result is not None:
#                 sarima_customer_results.append(sarima_result)

#     # ---------------------------
#     # HELPER TO FIND MOST COMMON PARAMS
#     # ---------------------------
#     def summarize_model(results_list, model_name):
#         if not results_list:
#             return None

#         temp = pd.DataFrame(results_list)
#         temp["best_params_str"] = temp["best_params"].astype(str)

#         freq = (
#             temp.groupby("best_params_str")
#             .size()
#             .reset_index(name="count")
#             .sort_values("count", ascending=False)
#         )

#         most_common = freq.iloc[0]["best_params_str"]

#         avg_mape = temp["mape"].mean()

#         return {
#             "SIC": CATEGORY_FILTER if CATEGORY_FILTER else "All",
#             "subSIC": subsic,
#             "Model": model_name,
#             "Best Parameters": most_common,
#             "Avg MAPE": avg_mape,
#             "Customer Count": len(temp),
#         }

#     for summary in [
#         summarize_model(rf_customer_results, "Random Forest"),
#         summarize_model(xgb_customer_results, "XGBoost"),
#         summarize_model(arima_customer_results, "ARIMA"),
#         summarize_model(sarima_customer_results, "SARIMA"),
#     ]:
#         if summary is not None:
#             all_results.append(summary)

# # ---------------------------
# # SAVE PARAMETER LIBRARY
# # ---------------------------
# library_df = pd.DataFrame(all_results)
# library_path = RESULTS_DIR / r"C:\Users\MauduH\Documents\Migael\parameter_library.xlsx"
# library_df.to_excel(library_path, index=False)

# print(f"\nParameter library saved to: {library_path}")
# print(library_df)

