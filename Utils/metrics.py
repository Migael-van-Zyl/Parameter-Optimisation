# import numpy as np
# from sklearn.metrics import mean_squared_error

# # def rmse(y_true, y_pred):
# #     return np.sqrt(mean_squared_error(y_true, y_pred))


# def mape(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     mask = y_true != 0
#     return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# import numpy as np
# from sklearn.metrics import mean_squared_error


# def rmse(y_true, y_pred) -> float:
#     return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# def mape(y_true, y_pred) -> float:
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     mask = y_true != 0
#     if mask.sum() == 0:
#         return float("nan")

#     return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)