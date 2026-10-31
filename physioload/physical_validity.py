"""Physical computability checks for measurement-level biomechanical outputs.

These checks run before reliability scoring. A high pose confidence cannot make
an output physically identifiable when scale, load definition or required
landmarks are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PhysicalValidityContext:
    available_landmarks: frozenset[str] = frozenset()
    metric_scale_valid: bool = False
    load_magnitude_defined: bool = False
    load_application_point_defined: bool = False
    load_direction_defined: bool = False

    @classmethod
    def from_landmarks(cls, landmarks: Iterable[str], **kwargs):
        return cls(available_landmarks=frozenset(landmarks), **kwargs)


@dataclass(frozen=True)
class PhysicalValidityResult:
    computable: bool
    reasons: tuple[str, ...]


REQUIRED_LANDMARKS: dict[str, frozenset[str]] = {
    "elbow_angle": frozenset({"shoulder", "elbow", "wrist"}),
    "knee_angle": frozenset({"hip", "knee", "ankle"}),
    "joint_angle": frozenset(),
    "segment_orientation": frozenset(),
    "bilateral_asymmetry": frozenset(),
    "com_projection": frozenset(),
    "moment_arm": frozenset(),
    "external_moment": frozenset(),
}


def assess_physical_validity(output: str, context: PhysicalValidityContext) -> PhysicalValidityResult:
    reasons: list[str] = []
    required = REQUIRED_LANDMARKS.get(output, frozenset())
    if context.available_landmarks and not required.issubset(context.available_landmarks):
        reasons.append("required_landmarks_missing")
    if output in {"metric_distance", "moment_arm", "external_moment"} and not context.metric_scale_valid:
        reasons.append("metric_scale_not_validated")
    if output == "external_moment":
        if not context.load_magnitude_defined:
            reasons.append("load_magnitude_missing")
        if not context.load_application_point_defined:
            reasons.append("load_application_point_missing")
        if not context.load_direction_defined:
            reasons.append("load_direction_missing")
    return PhysicalValidityResult(not reasons, tuple(reasons))

