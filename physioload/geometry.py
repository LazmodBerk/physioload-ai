"""Geometry and deliberately constrained static-biomechanics calculations.

The functions in this module do not infer force-plate quantities. Metric
outputs require an explicit pixel-to-metre scale and an explicit force vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


ArrayLike2D = Sequence[float] | np.ndarray


def _point(value: ArrayLike2D, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite two-dimensional point")
    return point


def joint_angle_degrees(a: ArrayLike2D, vertex: ArrayLike2D, c: ArrayLike2D) -> float:
    """Return the smaller angle A-vertex-C in degrees, in the range [0, 180]."""

    p_a, p_v, p_c = _point(a, "a"), _point(vertex, "vertex"), _point(c, "c")
    left, right = p_a - p_v, p_c - p_v
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= np.finfo(float).eps:
        raise ValueError("joint angle is undefined for a zero-length segment")
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def segment_orientation_degrees(start: ArrayLike2D, end: ArrayLike2D) -> float:
    """Return segment orientation in image coordinates in [-180, 180] degrees."""

    p_start, p_end = _point(start, "start"), _point(end, "end")
    delta = p_end - p_start
    if np.linalg.norm(delta) <= np.finfo(float).eps:
        raise ValueError("segment orientation is undefined for coincident points")
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def bilateral_asymmetry_degrees(left_angle: float, right_angle: float) -> float:
    """Absolute left-right kinematic angle difference."""

    values = np.asarray([left_angle, right_angle], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("angles must be finite")
    return float(abs(values[0] - values[1]))


@dataclass(frozen=True)
class SegmentMass:
    """One segment for a transparent two-dimensional CoM approximation."""

    proximal: ArrayLike2D
    distal: ArrayLike2D
    mass_fraction: float
    com_fraction_from_proximal: float


def approximate_com_2d(segments: Iterable[SegmentMass]) -> np.ndarray:
    """Estimate projected whole-body CoM from documented segment assumptions.

    The returned value is a projected image-plane position. It is not a force
    measurement and should be reported with the coefficient source.
    """

    weighted = np.zeros(2, dtype=float)
    total_mass = 0.0
    for index, segment in enumerate(segments):
        proximal = _point(segment.proximal, f"segment[{index}].proximal")
        distal = _point(segment.distal, f"segment[{index}].distal")
        if not 0.0 <= segment.com_fraction_from_proximal <= 1.0:
            raise ValueError("segment CoM fraction must be in [0, 1]")
        if not np.isfinite(segment.mass_fraction) or segment.mass_fraction <= 0:
            raise ValueError("segment mass fraction must be positive and finite")
        segment_com = proximal + segment.com_fraction_from_proximal * (distal - proximal)
        weighted += segment.mass_fraction * segment_com
        total_mass += segment.mass_fraction
    if total_mass <= 0:
        raise ValueError("at least one segment is required")
    return weighted / total_mass


def external_moment_2d(
    joint_px: ArrayLike2D,
    load_application_px: ArrayLike2D,
    force_newtons: ArrayLike2D,
    *,
    metres_per_pixel: float,
) -> float:
    """Return signed external moment (N·m) around a joint in the image plane.

    Positive values follow the mathematical counter-clockwise convention for
    the provided coordinate system. Callers must record the image coordinate
    convention and the scale source.
    """

    if not np.isfinite(metres_per_pixel) or metres_per_pixel <= 0:
        raise ValueError("metres_per_pixel must be positive and validated")
    joint = _point(joint_px, "joint_px")
    application = _point(load_application_px, "load_application_px")
    force = _point(force_newtons, "force_newtons")
    radius_m = (application - joint) * metres_per_pixel
    return float(radius_m[0] * force[1] - radius_m[1] * force[0])


def force_from_mass_2d(mass_kg: float, gravity_m_s2: float = 9.80665) -> np.ndarray:
    """Construct a downward image-plane gravity force from an explicit mass."""

    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be positive and finite")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 must be positive and finite")
    return np.array([0.0, mass_kg * gravity_m_s2], dtype=float)


def monte_carlo_angle_uncertainty(
    points: Mapping[str, ArrayLike2D],
    triplet: tuple[str, str, str],
    *,
    landmark_sigma_px: float,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Propagate isotropic landmark noise to a joint-angle distribution."""

    if landmark_sigma_px < 0 or not np.isfinite(landmark_sigma_px):
        raise ValueError("landmark_sigma_px must be finite and non-negative")
    if samples < 2:
        raise ValueError("samples must be at least two")
    base = {name: _point(value, name) for name, value in points.items()}
    missing = [name for name in triplet if name not in base]
    if missing:
        raise KeyError(f"missing landmarks: {missing}")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        perturbed = {
            name: point + rng.normal(0.0, landmark_sigma_px, size=2)
            for name, point in base.items()
        }
        try:
            values.append(joint_angle_degrees(*(perturbed[name] for name in triplet)))
        except ValueError:
            continue
    if len(values) < max(2, samples // 2):
        raise RuntimeError("too many invalid Monte Carlo samples")
    array = np.asarray(values)
    lower, upper = np.quantile(array, [0.025, 0.975])
    return {
        "mean_deg": float(np.mean(array)),
        "std_deg": float(np.std(array, ddof=1)),
        "ci95_low_deg": float(lower),
        "ci95_high_deg": float(upper),
        "valid_samples": int(array.size),
    }

