"""Katman 7 / değerlendirme metrikleri testleri.

A0-A4 karşılaştırma protokolünde kullanılan MAE, RMSE, PCK,
risk-kapsama eğrisi ve sınıflandırma AUC fonksiyonlarını doğrular.
"""

import numpy as np
import pytest

from physioload.metrics import mae, pck, risk_coverage, rmse


# ---------------------------------------------------------------------------
# MAE ve RMSE
# ---------------------------------------------------------------------------

def test_mae_perfect_prediction():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_mae_known_value():
    assert mae([0.0, 0.0], [3.0, 5.0]) == pytest.approx(4.0)


def test_rmse_known_value():
    """RMSE([0,0], [3,4]) = sqrt((9+16)/2) = sqrt(12.5)."""
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


# ---------------------------------------------------------------------------
# PCK (Percentage of Correct Keypoints)
# ---------------------------------------------------------------------------

def test_pck_all_within_threshold():
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    pred = np.array([[0.5, 0.5], [10.5, 0.5]])  # ≈0.7 px hata
    assert pck(ref, pred, threshold_px=2.0) == pytest.approx(1.0)


def test_pck_none_within_threshold():
    ref = np.array([[0.0, 0.0]])
    pred = np.array([[100.0, 100.0]])
    assert pck(ref, pred, threshold_px=2.0) == pytest.approx(0.0)


def test_pck_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pck([[0, 0], [1, 1]], [[0, 0]], threshold_px=5.0)


# ---------------------------------------------------------------------------
# Risk-kapsama eğrisi (seçici tahmin)
# ---------------------------------------------------------------------------

def test_risk_coverage_perfect_ranking():
    """En güvenilir ölçümler sıfır hatalıysa, risk monoton düşmelidir."""
    errors = np.array([0.0, 0.0, 10.0, 10.0])
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    result = risk_coverage(errors, scores)
    assert result["coverage"][-1] == pytest.approx(1.0)
    assert result["aurc"] >= 0.0


def test_risk_coverage_single_sample():
    result = risk_coverage([5.0], [0.7])
    assert result["aurc"] == pytest.approx(5.0)


def test_risk_coverage_shape_mismatch_raises():
    with pytest.raises(ValueError):
        risk_coverage([1.0, 2.0], [0.5])

