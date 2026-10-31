from physioload.physical_validity import PhysicalValidityContext, assess_physical_validity
from physioload.quality import MeasurementStatus, QualityFeatures, QualityGate


def test_valid_angle_survives_missing_metric_scale():
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True, 0.10),
        requested_outputs=["joint_angle", "external_moment"],
    )
    assert decision.outputs["joint_angle"].status is MeasurementStatus.ACCEPTED
    assert decision.outputs["external_moment"].status is MeasurementStatus.NOT_COMPUTABLE


def test_reliability_failure_is_rejected_not_not_computable():
    decision = QualityGate().evaluate(
        QualityFeatures(0.40, 0.40, 5.0, 0.05, True, True, 0.10),
        requested_outputs=["joint_angle"],
    )
    assert decision.outputs["joint_angle"].status is MeasurementStatus.REJECTED


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

