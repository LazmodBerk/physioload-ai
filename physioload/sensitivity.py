"""Measurement-specific landmark perturbation and error-impact analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


Point2D = Sequence[float] | np.ndarray
MeasurementFunction = Callable[[Mapping[str, np.ndarray]], float]


@dataclass(frozen=True)
class LandmarkImpact:
    landmark: str
    mean_absolute_change: float
    quantile_change: float
    valid_samples: int


def measurement_sensitivity_profile(
    points: Mapping[str, Point2D],
    measurement: MeasurementFunction,
    *,
    sigma_px: float,
    samples: int = 1000,
    quantile: float = 0.95,
    seed: int = 42,
) -> tuple[LandmarkImpact, ...]:
    """Return one perturbation-impact summary per landmark.

    This is an empirical sensitivity indicator, not a calibrated confidence
    interval. Perturbation levels must be justified for the selected data.
    """

    if not np.isfinite(sigma_px) or sigma_px < 0:
        raise ValueError("sigma_px must be finite and non-negative")
    if samples < 2:
        raise ValueError("samples must be at least two")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    base = {name: np.asarray(value, dtype=float) for name, value in points.items()}
    if not base or any(point.shape != (2,) or not np.all(np.isfinite(point)) for point in base.values()):
        raise ValueError("points must contain finite two-dimensional coordinates")
    baseline = float(measurement(base))
    rng = np.random.default_rng(seed)
    impacts: list[LandmarkImpact] = []
    for name in base:
        changes: list[float] = []
        for _ in range(samples):
            perturbed = {key: value.copy() for key, value in base.items()}
            perturbed[name] += rng.normal(0.0, sigma_px, size=2)
            try:
                value = float(measurement(perturbed))
            except (ValueError, FloatingPointError):
                continue
            if np.isfinite(value):
                changes.append(abs(value - baseline))
        if not changes:
            raise RuntimeError(f"no valid perturbation sample for {name}")
        array = np.asarray(changes)
        impacts.append(LandmarkImpact(name, float(np.mean(array)), float(np.quantile(array, quantile)), int(array.size)))
    return tuple(sorted(impacts, key=lambda item: item.quantile_change, reverse=True))


def error_impact_map(profile: Sequence[LandmarkImpact]) -> dict[str, float]:
    """Normalize landmark impacts for an explainable measurement-level map."""

    if not profile:
        return {}
    maximum = max(item.quantile_change for item in profile)
    if maximum <= np.finfo(float).eps:
        return {item.landmark: 0.0 for item in profile}
    return {item.landmark: item.quantile_change / maximum for item in profile}

