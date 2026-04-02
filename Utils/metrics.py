# import numpy as np
# from sklearn.metrics import mean_squared_error


# def rmse(y_true, y_pred) -> float:
#     """
#     Root Mean Squared Error
#     """
#     return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# def mape(y_true, y_pred) -> float:
#     """
#     Mean Absolute Percentage Error (handles zeros safely)
#     """
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     mask = y_true != 0
#     if mask.sum() == 0:
#         return float("nan")

#     return float(
#         np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#     )

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)

    if mask.sum() == 0:
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return float("nan")

    return float(mean_absolute_error(y_true[mask], y_pred[mask]))

    