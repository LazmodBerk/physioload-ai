from physioload.marp import RecaptureAction, recommend_recapture
from physioload.quality import QualityFeatures, QualityGate


def test_valid_angle_can_survive_invalid_metric_scale():
    decision = QualityGate().evaluate(
        QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True),
        requested_outputs=["joint_angle", "external_moment"],
    )
    assert "joint_angle" in decision.valid_outputs
    assert "external_moment" in decision.rejected_outputs


def test_marp_prioritizes_scale_for_metric_output():
    recommendation = recommend_recapture(QualityFeatures(0.95, 0.95, 5.0, 0.05, False, True))
    assert recommendation.action is RecaptureAction.CALIBRATE_SCALE

