from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physioload.geometry import external_moment_2d, joint_angle_degrees
from physioload.marp import recommend_recapture
from physioload.quality import QualityFeatures, QualityGate
from physioload_env import setup_environment


def main() -> None:
    env = setup_environment(root=ROOT)
    angle = joint_angle_degrees((0, 0), (1, 0), (1, 1))
    moment = external_moment_2d((0, 0), (100, 0), (0, 98.0665), metres_per_pixel=0.001)
    features = QualityFeatures(0.92, 0.91, 8.0, 0.08, True, True)
    decision = QualityGate().evaluate(features, requested_outputs=["joint_angle", "external_moment"])
    recommendation = recommend_recapture(features)
    assert abs(angle - 90.0) < 1e-9
    assert abs(moment - 9.80665) < 1e-6
    assert decision.accepted
    print(env.summary())
    print(f"angle={angle:.1f} deg, moment={moment:.5f} N.m")
    print("quality:", decision)
    print("MARP:", recommendation.action.value)


if __name__ == "__main__":
    main()

