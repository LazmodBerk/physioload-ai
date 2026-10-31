from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physioload.geometry import external_moment_2d, joint_angle_degrees
from physioload.physical_validity import PhysicalValidityContext
from physioload.quality import MeasurementStatus, QualityFeatures, QualityGate
from physioload.sensitivity import error_impact_map, measurement_sensitivity_profile
from physioload_env import setup_environment


def main() -> None:
    env = setup_environment(root=ROOT)
    angle = joint_angle_degrees((0, 0), (1, 0), (1, 1))
    moment = external_moment_2d((0, 0), (100, 0), (0, 98.0665), metres_per_pixel=0.001)
    features = QualityFeatures(0.92, 0.91, 8.0, 0.08, True, True, 0.10)
    context = PhysicalValidityContext(
        metric_scale_valid=True,
        load_magnitude_defined=True,
        load_application_point_defined=True,
        load_direction_defined=True,
    )
    decision = QualityGate().evaluate(
        features,
        requested_outputs=["joint_angle", "external_moment"],
        physical_context=context,
    )
    points = {"hip": (0, 0), "knee": (1, 0), "ankle": (1, 1)}
    profile = measurement_sensitivity_profile(
        points,
        lambda p: joint_angle_degrees(p["hip"], p["knee"], p["ankle"]),
        sigma_px=0.05,
        samples=100,
    )
    assert abs(angle - 90.0) < 1e-9
    assert abs(moment - 9.80665) < 1e-6
    assert decision.outputs["joint_angle"].status is MeasurementStatus.ACCEPTED
    assert decision.outputs["external_moment"].status is MeasurementStatus.ACCEPTED
    print(env.summary())
    print(f"angle={angle:.1f} deg, moment={moment:.5f} N.m")
    print("quality:", decision)
    print("error impact map:", error_impact_map(profile))


if __name__ == "__main__":
    main()

