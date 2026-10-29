"""Optional pose-estimator adapters with a common result schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol
from urllib.request import urlretrieve

import numpy as np


@dataclass(frozen=True)
class PoseResult:
    keypoints_xy: np.ndarray
    confidence: np.ndarray
    names: tuple[str, ...]
    backend: str
    inference_ms: float


class PoseEstimator(Protocol):
    name: str

    def predict(self, rgb_image: np.ndarray) -> PoseResult: ...


class MediaPipePoseEstimator:
    name = "mediapipe"

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )

    LANDMARK_NAMES = (
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
        "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
        "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
        "right_heel", "left_foot_index", "right_foot_index",
    )

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise RuntimeError("install the 'mediapipe' package for this backend") from error
        self._mp = mp
        if model_path is None:
            cache_root = Path.home() / ".cache" / "physioload" / "models"
            cache_root.mkdir(parents=True, exist_ok=True)
            model_path = cache_root / "pose_landmarker_heavy.task"
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Downloading MediaPipe pose model -> {model_path}")
            urlretrieve(self.MODEL_URL, model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._vision = vision
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        image = np.asarray(rgb_image)
        height, width = image.shape[:2]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("rgb_image must have shape (height, width, 3)")
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image.astype(np.uint8, copy=False)),
        )
        started = perf_counter()
        output = self._landmarker.detect(mp_image)
        elapsed_ms = (perf_counter() - started) * 1000
        if not output.pose_landmarks:
            raise RuntimeError("MediaPipe did not detect a pose")
        landmarks = output.pose_landmarks[0]
        keypoints = np.asarray([[p.x * width, p.y * height] for p in landmarks], dtype=float)
        confidence = np.asarray([p.visibility or p.presence for p in landmarks], dtype=float)
        names = self.LANDMARK_NAMES
        return PoseResult(keypoints, confidence, names, self.name, elapsed_ms)


class RTMPoseEstimator:
    name = "rtmpose"

    def __init__(self, *, mode: str = "balanced", backend: str = "onnxruntime") -> None:
        try:
            from rtmlib import Body
        except ImportError as error:
            raise RuntimeError("install the 'rtmlib' package for this backend") from error
        self._model = Body(to_openpose=False, mode=mode, backend=backend, device="cpu")

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        image = np.asarray(rgb_image)
        started = perf_counter()
        keypoints, scores = self._model(image[:, :, ::-1])
        elapsed_ms = (perf_counter() - started) * 1000
        keypoints = np.asarray(keypoints, dtype=float)
        scores = np.asarray(scores, dtype=float)
        if keypoints.ndim == 3:
            keypoints, scores = keypoints[0], scores[0]
        if keypoints.size == 0:
            raise RuntimeError("RTMPose did not detect a pose")
        names = tuple(f"coco_{index}" for index in range(keypoints.shape[0]))
        return PoseResult(keypoints, scores, names, self.name, elapsed_ms)

