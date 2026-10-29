"""Output-level reliability scoring and selective Quality Gate decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QualityFeatures:
    visibility: float
    pose_confidence: float
    view_deviation_deg: float
    geometry_penalty: float
    scale_valid: bool
    load_definition_valid: bool = True

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.visibility,
                self.pose_confidence,
                self.view_deviation_deg,
                self.geometry_penalty,
                float(self.scale_valid),
                float(self.load_definition_valid),
            ],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualityDecision:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    valid_outputs: tuple[str, ...]
    rejected_outputs: tuple[str, ...]


@dataclass(frozen=True)
class QualityGate:
    minimum_visibility: float = 0.70
    minimum_pose_confidence: float = 0.70
    maximum_view_deviation_deg: float = 20.0
    maximum_geometry_penalty: float = 0.35
    acceptance_threshold: float = 0.65

    def score(self, features: QualityFeatures) -> float:
        values = features.vector()
        if not np.all(np.isfinite(values)):
            return 0.0
        visibility = np.clip(features.visibility, 0.0, 1.0)
        confidence = np.clip(features.pose_confidence, 0.0, 1.0)
        view = 1.0 - np.clip(features.view_deviation_deg / 45.0, 0.0, 1.0)
        geometry = 1.0 - np.clip(features.geometry_penalty, 0.0, 1.0)
        return float(0.30 * visibility + 0.25 * confidence + 0.25 * view + 0.20 * geometry)

    def evaluate(self, features: QualityFeatures, *, requested_outputs: Iterable[str]) -> QualityDecision:
        requested = tuple(dict.fromkeys(requested_outputs))
        reasons: list[str] = []
        if features.visibility < self.minimum_visibility:
            reasons.append("low_visibility")
        if features.pose_confidence < self.minimum_pose_confidence:
            reasons.append("low_pose_confidence")
        if features.view_deviation_deg > self.maximum_view_deviation_deg:
            reasons.append("viewpoint_out_of_plane")
        if features.geometry_penalty > self.maximum_geometry_penalty:
            reasons.append("anatomical_geometry_inconsistent")
        score = self.score(features)
        if score < self.acceptance_threshold:
            reasons.append("reliability_score_below_threshold")

        valid: list[str] = []
        rejected: list[str] = []
        base_valid = not reasons
        for output in requested:
            output_reasons = list(reasons)
            if output in {"metric_distance", "external_moment", "moment_arm"} and not features.scale_valid:
                output_reasons.append("metric_scale_not_validated")
            if output == "external_moment" and not features.load_definition_valid:
                output_reasons.append("external_load_not_fully_defined")
            if output_reasons:
                rejected.append(output)
            else:
                valid.append(output)

        accepted = base_valid and not rejected
        return QualityDecision(
            accepted=accepted,
            score=score,
            reasons=tuple(dict.fromkeys(reasons)),
            valid_outputs=tuple(valid),
            rejected_outputs=tuple(rejected),
        )


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
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state)),
        ]
    )
    return model.fit(x, y)

