'''
TODO:
Reconfigure XGBoost script to be importable and callable from this main.py file.
Call and test functionality in main.py
Repeat for RF and SARIMA

FUTURE:
Look at refining classification algorithm and how we assign best parameters
'''
#test 
# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from models.xgb import xgb_model, y_pred as xgb_y_pred

BASE_DIR = Path(__file__).resolve().parent.parent


# LOAD TRAIN & TEST DATA
train_df = pd.read_excel(BASE_DIR / "Parameter Optimisation" / "data" / "batchedtrain_dataset.xlsx")
test_df  = pd.read_excel(BASE_DIR / "Parameter Optimisation" / "data" / "batchedtest_dataset.xlsx")

# CALL XGBOOST FUNCTIONS
# CALL RF FUNCTIONS
# CALL SARIMA FUNCTIONS
