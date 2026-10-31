import numpy as np

from physioload.geometry import joint_angle_degrees
from physioload.sensitivity import error_impact_map, measurement_sensitivity_profile


def test_measurement_sensitivity_profile_is_deterministic_and_normalized():
    points = {"hip": (0.0, 0.0), "knee": (1.0, 0.0), "ankle": (1.0, 1.0)}

    def knee_angle(data):
        return joint_angle_degrees(data["hip"], data["knee"], data["ankle"])

    first = measurement_sensitivity_profile(points, knee_angle, sigma_px=0.1, samples=100, seed=7)
    second = measurement_sensitivity_profile(points, knee_angle, sigma_px=0.1, samples=100, seed=7)
    assert first == second
    impact = error_impact_map(first)
    assert set(impact) == set(points)
    assert np.isclose(max(impact.values()), 1.0)

