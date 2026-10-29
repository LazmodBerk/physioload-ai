import numpy as np
import pytest

from physioload.geometry import (
    SegmentMass,
    approximate_com_2d,
    external_moment_2d,
    joint_angle_degrees,
)


def test_right_angle():
    assert joint_angle_degrees((0, 0), (1, 0), (1, 1)) == pytest.approx(90.0)


def test_external_moment_requires_scale():
    with pytest.raises(ValueError):
        external_moment_2d((0, 0), (10, 0), (0, 100), metres_per_pixel=0)


def test_external_moment_value():
    assert external_moment_2d((0, 0), (100, 0), (0, 100), metres_per_pixel=0.001) == pytest.approx(10.0)


def test_projected_com_is_mass_weighted():
    result = approximate_com_2d(
        [
            SegmentMass((0, 0), (2, 0), 0.5, 0.5),
            SegmentMass((0, 2), (2, 2), 0.5, 0.5),
        ]
    )
    assert np.allclose(result, [1, 1])

