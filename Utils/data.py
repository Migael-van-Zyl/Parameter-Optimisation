# utils/data.py
# from pathlib import Path
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from typing import List, Optional, Tuple

# def load_classifier_data(
#     train_path: Path,
#     test_path: Path,
#     target_col: str = "sic_batch",
#     feature_cols: Optional[List[str]] = None
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder, List[str]]:
#     """
#     Loads train/test Excel files, selects features, and label-encodes the target.
#     Returns: X_train, X_test, y_train_enc, y_test_enc, label_encoder, feature_names
#     """
#     train_df = pd.read_excel(train_path)
#     test_df  = pd.read_excel(test_path)

#     # Choose features
#     if feature_cols is None:
#         # Default: all numeric columns except target
#         feature_cols = [
#             c for c in train_df.select_dtypes(include=["number"]).columns
#             if c != target_col
#         ]
#     else:
#         # Validate features exist
#         missing = [c for c in feature_cols if c not in train_df.columns]
#         if missing:
#             raise ValueError(f"Missing feature(s) in train file: {missing}")

#     X_train = train_df[feature_cols].copy()
#     X_test  = test_df[feature_cols].copy()

#     # Encode target using a single encoder fitted on train and applied to test
#     if target_col not in train_df.columns or target_col not in test_df.columns:
#         raise ValueError(f"Target column '{target_col}' must exist in both files.")

#     le = LabelEncoder()
#     y_train_enc = pd.Series(le.fit_transform(train_df[target_col]), name=target_col)
#     y_test_enc  = pd.Series(le.transform(test_df[target_col]), name=target_col)

#     return X_train, X_test, y_train_enc, y_test_enc, le, feature_cols

# utils/data.py
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Tuple

def load_classifier_data(
    train_path: Path,
    test_path: Path,
    target_col: str = "sic_batch",
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder, List[str]]:
    train_df = pd.read_excel(train_path)
    test_df  = pd.read_excel(test_path)

    if feature_cols is None:
        feature_cols = [c for c in train_df.select_dtypes(include=["number"]).columns if c != target_col]

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' missing in train/test files.")

    le = LabelEncoder()
    y_train_enc = pd.Series(le.fit_transform(train_df[target_col]), name=target_col)
    y_test_enc  = pd.Series(le.transform(test_df[target_col]), name=target_col)

    return X_train, X_test, y_train_enc, y_test_enc, le, feature_cols