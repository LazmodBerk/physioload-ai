"""Katman 6 — Ölçüm Bazlı Kalite Kapısı genişletilmiş testleri.

Fiziksel geçerlilik ve güvenilirlik arasındaki ayrımı, üçlü karar sözleşmesini
ve seçici kabul/ret mantığını doğrular.
"""

import pytest

from physioload.physical_validity import PhysicalValidityContext, assess_physical_validity
from physioload.quality import MeasurementStatus, QualityFeatures, QualityGate


# ---------------------------------------------------------------------------
# Temel üçlü karar sözleşmesi
# ---------------------------------------------------------------------------

def test_accepted_when_all_conditions_met():
    """Fiziksel koşullar ve güvenilirlik yeterliyse ACCEPTED döner."""
    gate = QualityGate()
    decision = gate.evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, True, True, 0.10),
        requested_outputs=["joint_angle"],
    )
    assert decision.outputs["joint_angle"].status is MeasurementStatus.ACCEPTED


def test_rejected_when_reliability_fails():
    """Düşük güvenilirlik REJECTED statüsü üretmeli, NOT_COMPUTABLE değil."""
    decision = QualityGate().evaluate(
        QualityFeatures(0.40, 0.40, 5.0, 0.05, True, True, 0.10),
        requested_outputs=["joint_angle"],
    )
    assert decision.outputs["joint_angle"].status is MeasurementStatus.REJECTED


def test_not_computable_when_metric_scale_missing():
    """Metrik ölçek eksikse dış moment NOT_COMPUTABLE olmalıdır."""
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True, 0.10),
        requested_outputs=["joint_angle", "external_moment"],
    )
    assert decision.outputs["joint_angle"].status is MeasurementStatus.ACCEPTED
    assert decision.outputs["external_moment"].status is MeasurementStatus.NOT_COMPUTABLE


def test_partial_rejection_does_not_invalidate_other_outputs():
    """Bir ölçümün reddi diğer geçerli ölçümleri geçersiz kılmamalıdır."""
    decision = QualityGate().evaluate(
        QualityFeatures(0.40, 0.40, 5.0, 0.05, False, True, 0.10),
        requested_outputs=["joint_angle", "external_moment"],
    )
    # joint_angle: güvenilirlik yetersiz → REJECTED
    # external_moment: ölçek eksik → NOT_COMPUTABLE
    assert decision.outputs["joint_angle"].status is MeasurementStatus.REJECTED
    assert decision.outputs["external_moment"].status is MeasurementStatus.NOT_COMPUTABLE


# ---------------------------------------------------------------------------
# Fiziksel geçerlilik ön denetimi
# ---------------------------------------------------------------------------

def test_external_moment_requires_complete_load_definition():
    context = PhysicalValidityContext(
        metric_scale_valid=True,
        load_magnitude_defined=True,
        load_application_point_defined=False,
        load_direction_defined=True,
    )
    result = assess_physical_validity("external_moment", context)
    assert not result.computable
    assert "load_application_point_missing" in result.reasons


def test_joint_angle_computable_without_metric_scale():
    """Eklem açısı metrik ölçek olmadan hesaplanabilir olmalıdır."""
    context = PhysicalValidityContext(metric_scale_valid=False)
    result = assess_physical_validity("joint_angle", context)
    assert result.computable


def test_high_confidence_cannot_override_physical_invalidity():
    """Yüksek güven puanı fiziksel hesaplanamaz durumu geçersiz kılamaz."""
    # Dış moment için ölçek geçerli, yük tanımı eksik
    features = QualityFeatures(0.99, 0.99, 1.0, 0.01, True, False, 0.05)
    decision = QualityGate().evaluate(
        features,
        requested_outputs=["external_moment"],
    )
    assert decision.outputs["external_moment"].status is MeasurementStatus.NOT_COMPUTABLE


# ---------------------------------------------------------------------------
# Kalite Kapısı karar özellikleri
# ---------------------------------------------------------------------------

def test_valid_outputs_property():
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True, 0.10),
        requested_outputs=["joint_angle", "external_moment"],
    )
    assert "joint_angle" in decision.valid_outputs
    assert "external_moment" not in decision.valid_outputs


def test_not_computable_outputs_property():
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True, 0.10),
        requested_outputs=["joint_angle", "external_moment"],
    )
    assert "external_moment" in decision.not_computable_outputs


def test_score_is_none_for_not_computable():
    """NOT_COMPUTABLE kararının güvenilirlik skoru None olmalıdır."""
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True, 0.10),
        requested_outputs=["external_moment"],
    )
    assert decision.outputs["external_moment"].score is None


def test_rejection_reason_codes_populated():
    """Reddedilen ölçümde gerekçe kodları boş olmamalıdır."""
    decision = QualityGate().evaluate(
        QualityFeatures(0.20, 0.20, 5.0, 0.05, True, True, 0.10),
        requested_outputs=["joint_angle"],
    )
    output = decision.outputs["joint_angle"]
    assert output.status is MeasurementStatus.REJECTED
    assert len(output.reasons) > 0


def test_viewpoint_deviation_triggers_rejection():
    """Aşırı bakış açısı sapması redde neden olmalıdır."""
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 45.0, 0.05, True, True, 0.10),
        requested_outputs=["joint_angle"],
    )
    assert decision.outputs["joint_angle"].status is MeasurementStatus.REJECTED
    assert "viewpoint_out_of_plane" in decision.outputs["joint_angle"].reasons

