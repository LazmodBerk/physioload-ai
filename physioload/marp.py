"""Mechanics-Aware Re-Capture Policy (MARP)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .quality import QualityFeatures


class RecaptureAction(str, Enum):
    ALIGN_CAMERA = "align_camera_to_measurement_plane"
    INCREASE_DISTANCE = "increase_camera_distance"
    IMPROVE_LIGHTING = "improve_lighting"
    CLEAR_OCCLUSION = "clear_occlusion"
    CALIBRATE_SCALE = "calibrate_metric_scale"
    DEFINE_LOAD = "define_external_load"
    RETAKE = "retake_same_setup"


@dataclass(frozen=True)
class RecaptureRecommendation:
    action: RecaptureAction
    expected_error_reduction: float
    reason: str
    evidence: Mapping[str, float | bool]


def recommend_recapture(features: QualityFeatures) -> RecaptureRecommendation:
    """Return an auditable baseline action before a learned policy is available."""

    candidates: list[RecaptureRecommendation] = []
    if features.view_deviation_deg > 10:
        candidates.append(
            RecaptureRecommendation(
                RecaptureAction.ALIGN_CAMERA,
                min(features.view_deviation_deg / 45.0, 1.0),
                "camera view is likely to distort the measurement plane",
                {"view_deviation_deg": features.view_deviation_deg},
            )
        )
    if features.visibility < 0.80:
        candidates.append(
            RecaptureRecommendation(
                RecaptureAction.CLEAR_OCCLUSION,
                min(1.0 - features.visibility, 1.0),
                "one or more required landmarks are poorly visible",
                {"visibility": features.visibility},
            )
        )
    if features.pose_confidence < 0.75:
        candidates.append(
            RecaptureRecommendation(
                RecaptureAction.IMPROVE_LIGHTING,
                min(1.0 - features.pose_confidence, 1.0),
                "pose confidence is low; lighting and contrast should be checked",
                {"pose_confidence": features.pose_confidence},
            )
        )
    if features.geometry_penalty > 0.25:
        candidates.append(
            RecaptureRecommendation(
                RecaptureAction.INCREASE_DISTANCE,
                min(features.geometry_penalty, 1.0),
                "body geometry is inconsistent or cropped",
                {"geometry_penalty": features.geometry_penalty},
            )
        )
    if not features.scale_valid:
        candidates.append(
            RecaptureRecommendation(
                RecaptureAction.CALIBRATE_SCALE,
                1.0,
                "metric output requires a validated image scale",
                {"scale_valid": False},
            )
        )
    if not features.load_definition_valid:
        candidates.append(
            RecaptureRecommendation(
                RecaptureAction.DEFINE_LOAD,
                1.0,
                "external moment requires load magnitude, application point and direction",
                {"load_definition_valid": False},
            )
        )
    if not candidates:
        return RecaptureRecommendation(
            RecaptureAction.RETAKE,
            0.0,
            "no dominant correctable cause was identified",
            {},
        )
    return max(candidates, key=lambda candidate: candidate.expected_error_reduction)

