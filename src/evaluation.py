"""
evaluation.py
Reusable functions to evaluate model performance.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def get_metrics(y_true, y_pred):
    """Return a dictionary of key performance metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def get_confusion_matrix(y_true, y_pred):
    """Return the confusion matrix as a 2x2 array."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, target_names=None):
    """Return the full text classification report."""
    return classification_report(y_true, y_pred, target_names=target_names)