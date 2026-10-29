"""Evaluation metrics used by the PhysioLoad protocol."""

from __future__ import annotations

import numpy as np


def mae(reference, prediction) -> float:
    reference, prediction = np.asarray(reference, float), np.asarray(prediction, float)
    return float(np.mean(np.abs(reference - prediction)))


def rmse(reference, prediction) -> float:
    reference, prediction = np.asarray(reference, float), np.asarray(prediction, float)
    return float(np.sqrt(np.mean((reference - prediction) ** 2)))


def pck(reference_xy, prediction_xy, threshold_px: float) -> float:
    reference, prediction = np.asarray(reference_xy, float), np.asarray(prediction_xy, float)
    if reference.shape != prediction.shape or reference.shape[-1] != 2:
        raise ValueError("reference and prediction must have identical (..., 2) shapes")
    distances = np.linalg.norm(reference - prediction, axis=-1)
    return float(np.mean(distances <= threshold_px))


def risk_coverage(errors, reliability_scores) -> dict[str, np.ndarray | float]:
    """Compute selective risk as lower-confidence samples are rejected."""

    errors = np.asarray(errors, float)
    scores = np.asarray(reliability_scores, float)
    if errors.shape != scores.shape or errors.ndim != 1:
        raise ValueError("errors and scores must be same-length vectors")
    order = np.argsort(scores)[::-1]
    sorted_errors = errors[order]
    cumulative_risk = np.cumsum(sorted_errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    aurc = float(np.trapz(cumulative_risk, coverage))
    return {"coverage": coverage, "risk": cumulative_risk, "aurc": aurc}


def classification_auc(severe_error, severe_error_probability) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(severe_error, int)
    p = np.asarray(severe_error_probability, float)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }

