import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

def evaluate_at_threshold(model, X, y, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return {
        "threshold": threshold,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
def compute_roc_auc(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_proba)
    fpr, tpr, thresholds = roc_curve(y, y_proba)

    return {
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }
def evaluate_multiple_thresholds(model, X, y, thresholds):
    results = []

    y_proba = model.predict_proba(X)[:, 1]

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        report = classification_report(y, y_pred, output_dict=True)

        results.append({
            "threshold": t,
            "precision_churn": report["1"]["precision"],
            "recall_churn": report["1"]["recall"],
            "f1_churn": report["1"]["f1-score"]
        })

    return results