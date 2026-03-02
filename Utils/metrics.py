# utils/metrics.py
# from typing import Optional, Dict, Any
# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )
# import matplotlib.pyplot as plt
# import numpy as np

# def evaluate_classifier(
#     y_true,
#     y_pred,
#     label_names: Optional[np.ndarray] = None,
#     title: str = "Confusion Matrix",
#     show_plot: bool = True
# ) -> Dict[str, Any]:
#     """
#     Computes accuracy, prints classification report, and plots confusion matrix.
#     Returns a dict with metrics and the matplotlib Figure (if plotted).
#     """
#     acc = accuracy_score(y_true, y_pred)
#     print(f"Accuracy: {acc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=label_names))

#     cm = confusion_matrix(y_true, y_pred)
#     fig = None
#     if show_plot:
#         disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         disp.plot(ax=ax, cmap="Blues", values_format="d")
#         ax.set_title(title)
#         fig.tight_layout()
#         plt.show()

#     return {"accuracy": acc, "confusion_matrix": cm, "figure": fig}

# utils/metrics.py
# from typing import Optional, Dict, Any
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import numpy as np

# def evaluate_classifier(
#     y_true,
#     y_pred,
#     label_names: Optional[np.ndarray] = None,
#     title: str = "Confusion Matrix",
#     show_plot: bool = True
# ) -> Dict[str, Any]:
#     acc = accuracy_score(y_true, y_pred)
#     print(f"Accuracy: {acc:.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=label_names))

#     cm = confusion_matrix(y_true, y_pred)
#     fig = None
#     if show_plot:
#         disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         disp.plot(ax=ax, cmap="Blues", values_format="d")
#         ax.set_title(title)
#         fig.tight_layout()
#         plt.show()

#     return {"accuracy": acc, "confusion_matrix": cm, "figure": fig}

# utils/metrics.py
from typing import Optional, Dict, Any, Sequence
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import numpy as np

def evaluate_classifier(
    y_true,
    y_pred,
    label_names: Optional[Sequence] = None,  # can be np.ndarray or list
    title: str = "Confusion Matrix",
    show_plot: bool = True,
    zero_division: int = 0
) -> Dict[str, Any]:
    """
    Computes accuracy, prints classification report, and plots confusion matrix.

    Fixes:
    - Converts label_names to strings (sklearn expects strings for target_names).
    - Passes `labels` explicitly to classification_report for consistent ordering.
    - Uses zero_division for stability when some classes have no predictions.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Canonical label order based on ground truth
    unique_labels = np.unique(y_true)
    labels = unique_labels.tolist()  # e.g., [0,1,2,...] numeric labels

    # Prepare display names (strings)
    if label_names is None:
        display_names = [str(c) for c in labels]
    else:
        # Convert any numeric/int labels to strings
        display_names = [str(x) for x in label_names]
        # If length mismatch, fall back to labels-as-strings
        if len(display_names) != len(labels):
            display_names = [str(c) for c in labels]

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,                  # enforce order
            target_names=display_names,     # must be strings
            zero_division=zero_division     # handle no-pred classes cleanly
        )
    )

    # Confusion matrix & plot using the same label order
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = None
    if show_plot:
        disp = ConfusionMatrixDisplay(cm, display_labels=display_names)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

    return {"accuracy": acc, "confusion_matrix": cm, "figure": fig}