import pandas as pd
import numpy as np
import os

# FILE PATHS - CHANGE THESE TO YOUR ACTUAL FILE LOCATIONS
best_file = r"C:\Users\MauduH\Documents\Migael\ARIMA_Outputs\all_categories_best_arima_results.xlsx"
common_file = r"C:\Users\MauduH\Documents\Migael\ARIMA_Outputs\all_categories_common_arima_results.xlsx"

# Output files
best_output_file = r"C:\Users\MauduH\Documents\Migael\ARIMA_Outputs\all_categories_best_arima_results_with_estimated_mape.xlsx"
common_output_file = r"C:\Users\MauduH\Documents\Migael\ARIMA_Outputs\all_categories_common_arima_results_with_estimated_mape.xlsx"


# HELPER FUNCTION
def add_estimated_mape(df: pd.DataFrame, mae_column: str, mean_column: str = "mean") -> pd.DataFrame:
    """
    Adds an estimated MAPE column using:
        estimated_MAPE = (MAE / abs(mean)) * 100

    This is only an approximation.
    """
    df = df.copy()

    # Ensure numeric
    df[mae_column] = pd.to_numeric(df[mae_column], errors="coerce")
    df[mean_column] = pd.to_numeric(df[mean_column], errors="coerce")

    # Avoid division by zero and invalid numbers
    valid_mask = (
        df[mae_column].notna()
        & df[mean_column].notna()
        & np.isfinite(df[mae_column])
        & np.isfinite(df[mean_column])
        & (df[mean_column] != 0)
    )

    df["estimated_MAPE"] = np.nan
    df.loc[valid_mask, "estimated_MAPE"] = (
        df.loc[valid_mask, mae_column] / df.loc[valid_mask, mean_column].abs()
    ) * 100

    return df


# PROCESS BEST ARIMA RESULTS
if os.path.exists(best_file):
    best_df = pd.read_excel(best_file)

    if "best_MAE" not in best_df.columns:
        raise ValueError(
            f"'best_MAE' column not found in best results file: {best_file}"
        )

    if "mean" not in best_df.columns:
        raise ValueError(
            f"'mean' column not found in best results file: {best_file}"
        )

    best_df = add_estimated_mape(best_df, mae_column="best_MAE", mean_column="mean")
    best_df.to_excel(best_output_file, index=False)

    print("Best ARIMA results processed successfully.")
    print(f"Saved to: {best_output_file}")
else:
    print(f"Best ARIMA results file not found: {best_file}")


# PROCESS COMMON ARIMA RESULTS
if os.path.exists(common_file):
    common_df = pd.read_excel(common_file)

    if "MAE" not in common_df.columns:
        raise ValueError(
            f"'MAE' column not found in common results file: {common_file}"
        )

    if "mean" not in common_df.columns:
        raise ValueError(
            f"'mean' column not found in common results file: {common_file}"
        )

    common_df = add_estimated_mape(common_df, mae_column="MAE", mean_column="mean")
    common_df.to_excel(common_output_file, index=False)

    print("Common ARIMA results processed successfully.")
    print(f"Saved to: {common_output_file}")
else:
    print(f"Common ARIMA results file not found: {common_file}")


print("\nDone.")
print("Note: estimated_MAPE is an approximation, not exact MAPE.")