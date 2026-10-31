"""Measurement-level reliability scoring and selective Quality Gate decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np

from .physical_validity import PhysicalValidityContext, assess_physical_validity


class MeasurementStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    NOT_COMPUTABLE = "NOT_COMPUTABLE"


@dataclass(frozen=True)
class QualityFeatures:
    visibility: float
    pose_confidence: float
    view_deviation_deg: float
    geometry_penalty: float
    scale_valid: bool
    load_definition_valid: bool = True
    sensitivity_penalty: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray([
            self.visibility, self.pose_confidence, self.view_deviation_deg,
            self.geometry_penalty, self.sensitivity_penalty,
            float(self.scale_valid), float(self.load_definition_valid),
        ], dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OutputDecision:
    output: str
    status: MeasurementStatus
    score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class QualityDecision:
    outputs: Mapping[str, OutputDecision]

    @property
    def accepted(self) -> bool:
        return bool(self.outputs) and all(item.status is MeasurementStatus.ACCEPTED for item in self.outputs.values())

    @property
    def valid_outputs(self) -> tuple[str, ...]:
        return tuple(name for name, item in self.outputs.items() if item.status is MeasurementStatus.ACCEPTED)

    @property
    def rejected_outputs(self) -> tuple[str, ...]:
        return tuple(name for name, item in self.outputs.items() if item.status is MeasurementStatus.REJECTED)

    @property
    def not_computable_outputs(self) -> tuple[str, ...]:
        return tuple(name for name, item in self.outputs.items() if item.status is MeasurementStatus.NOT_COMPUTABLE)


@dataclass(frozen=True)
class QualityGate:
    # Initial engineering defaults only. Calibrate them on validation data.
    minimum_visibility: float = 0.70
    minimum_pose_confidence: float = 0.70
    maximum_view_deviation_deg: float = 20.0
    maximum_geometry_penalty: float = 0.35
    maximum_sensitivity_penalty: float = 0.50
    acceptance_threshold: float = 0.65

    def score(self, features: QualityFeatures) -> float:
        if not np.all(np.isfinite(features.vector())):
            return 0.0
        visibility = np.clip(features.visibility, 0.0, 1.0)
        confidence = np.clip(features.pose_confidence, 0.0, 1.0)
        view = 1.0 - np.clip(features.view_deviation_deg / 45.0, 0.0, 1.0)
        geometry = 1.0 - np.clip(features.geometry_penalty, 0.0, 1.0)
        sensitivity = 1.0 - np.clip(features.sensitivity_penalty, 0.0, 1.0)
        return float(0.25 * visibility + 0.20 * confidence + 0.20 * view + 0.15 * geometry + 0.20 * sensitivity)

    def _reliability_reasons(self, features: QualityFeatures, score: float) -> tuple[str, ...]:
        reasons: list[str] = []
        if features.visibility < self.minimum_visibility:
            reasons.append("low_visibility")
        if features.pose_confidence < self.minimum_pose_confidence:
            reasons.append("low_pose_confidence")
        if features.view_deviation_deg > self.maximum_view_deviation_deg:
            reasons.append("viewpoint_out_of_plane")
        if features.geometry_penalty > self.maximum_geometry_penalty:
            reasons.append("anatomical_geometry_inconsistent")
        if features.sensitivity_penalty > self.maximum_sensitivity_penalty:
            reasons.append("measurement_too_sensitive_to_landmark_error")
        if score < self.acceptance_threshold:
            reasons.append("reliability_score_below_threshold")
        return tuple(dict.fromkeys(reasons))

    def evaluate(self, features: QualityFeatures, *, requested_outputs: Iterable[str], physical_context: PhysicalValidityContext | None = None) -> QualityDecision:
        requested = tuple(dict.fromkeys(requested_outputs))
        context = physical_context or PhysicalValidityContext(
            metric_scale_valid=features.scale_valid,
            load_magnitude_defined=features.load_definition_valid,
            load_application_point_defined=features.load_definition_valid,
            load_direction_defined=features.load_definition_valid,
        )
        score = self.score(features)
        reliability_reasons = self._reliability_reasons(features, score)
        decisions: dict[str, OutputDecision] = {}
        for output in requested:
            physical = assess_physical_validity(output, context)
            if not physical.computable:
                decisions[output] = OutputDecision(output, MeasurementStatus.NOT_COMPUTABLE, None, physical.reasons)
            elif reliability_reasons:
                decisions[output] = OutputDecision(output, MeasurementStatus.REJECTED, score, reliability_reasons)
            else:
                decisions[output] = OutputDecision(output, MeasurementStatus.ACCEPTED, score, ())
        return QualityDecision(decisions)


def fit_calibrator(features: np.ndarray, severe_error: np.ndarray, *, random_state: int = 42):
    """Fit a reproducible baseline severe-error probability model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.asarray(features, dtype=float)
    y = np.asarray(severe_error, dtype=int)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and labels have incompatible shapes")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both reliable and severe-error examples are required")
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state)),
    ]).fit(x, y)

