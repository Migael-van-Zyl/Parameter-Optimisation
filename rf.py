# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:25:50 2026

@author: MauduH
"""

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
#running code by Hudivha

# LOAD TRAIN & TEST DATA
train_df = pd.read_excel(r"C:\Users\MauduH\Documents\Migael\batchedtrain_dataset.xlsx")
test_df  = pd.read_excel(r"C:\Users\MauduH\Documents\Migael\batchedtest_dataset.xlsx")

# DEFINE FEATURES & TARGET
TARGET = "sic_batch"

FEATURES = [
    "offpeakconsumption",
    "standardconsumption",
    "peakconsumption",
    "totalconsumption"
]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# TRAIN RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train, y_train)

# PREDICTION & EVALUATION
y_pred = rf.predict(X_test)

print("\n=== RANDOM FOREST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y_test.unique())

disp = ConfusionMatrixDisplay(cm, display_labels=labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest – SIC Batch Confusion Matrix (13 Batches)")
plt.tight_layout()
plt.show()
