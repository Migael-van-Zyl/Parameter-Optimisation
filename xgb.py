# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:02:41 2026

@author: MauduH
"""

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# LOAD DATA
train_df = pd.read_excel(r"C:\Users\MauduH\Documents\Migael\batchedtrain_dataset.xlsx")
test_df  = pd.read_excel(r"C:\Users\MauduH\Documents\Migael\batchedtrain_dataset.xlsx")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# CHECK COLUMN NAME
print("\nColumns available:")
print(train_df.columns)

TARGET = "sic_batch"   

# CONVERT TARGET TO 0-BASED INDEX
train_df[TARGET] = train_df[TARGET] - 1
test_df[TARGET]  = test_df[TARGET] - 1

# REMOVE NON-NUMERIC COLUMNS
X_train = train_df.drop(columns=[TARGET])
X_test  = test_df.drop(columns=[TARGET])

X_train = X_train.select_dtypes(include=["number"])
X_test  = X_test.select_dtypes(include=["number"])

y_train = train_df[TARGET]
y_test  = test_df[TARGET]

# TRAIN MODEL
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train)

print("\nXGBoost training complete")

# PREDICT
y_pred = xgb_model.predict(X_test)

# RESULTS
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

labels = sorted(y_test.unique())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(10,7))
disp.plot(values_format="d")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

print(train_df["sic_batch"].value_counts().sort_index())
