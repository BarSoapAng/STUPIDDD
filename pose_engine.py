from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseResult:
    is_dab: bool
    landmarks: dict[str, tuple[int, int]]
    extended_side: Optional[str]
    nose_px: Optional[tuple[int, int]]


class PoseEngine:
    _POSE_TASK_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    )
    _REQUIRED_LANDMARKS = (
        "LEFT_WRIST",
        "LEFT_ELBOW",
        "LEFT_SHOULDER",
        "RIGHT_WRIST",
        "RIGHT_ELBOW",
        "RIGHT_SHOULDER",
        "NOSE",
    )
    _OPTIONAL_LANDMARKS = ("LEFT_EYE", "RIGHT_EYE")

    def __init__(self, task_model_path: str = "assets/pose_landmarker_heavy.task") -> None:
        self._backend: str = "disabled"
        self._legacy_pose_module = None
        self._legacy_pose = None
        self._task_pose = None
        self._task_pose_landmark_enum = None
        self._task_model_path = Path(task_model_path)

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._init_legacy_backend()
        else:
            self._init_tasks_backend()

    def classify_frame(self, bgr_frame: np.ndarray) -> PoseResult:
        return self.classify(bgr_frame)

    def classify(self, bgr_frame: np.ndarray) -> PoseResult:
        if self._backend == "disabled":
            return PoseResult(False, {}, None, None)

        if self._backend == "legacy":
            return self._classify_with_legacy(bgr_frame)
        return self._classify_with_tasks(bgr_frame)

    def _classify_with_legacy(self, bgr_frame: np.ndarray) -> PoseResult:
        frame_h, frame_w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._legacy_pose.process(rgb_frame)

        if not result.pose_landmarks:
            return PoseResult(False, {}, None, None)

        landmarks = result.pose_landmarks.landmark
        pixel_landmarks = self._to_pixel_landmarks(
            landmarks=landmarks,
            frame_w=frame_w,
            frame_h=frame_h,
            index_for_name=lambda name: self._legacy_pose_module.PoseLandmark[name].value,
        )
        nose_px = pixel_landmarks.get("NOSE")

        if not self._visibility_guard_passed(
            landmarks=landmarks,
            landmark_for_name=lambda name: landmarks[self._legacy_pose_module.PoseLandmark[name].value],
        ):
            return PoseResult(False, pixel_landmarks, None, nose_px)

        left_ok, left_angle = self._check_orientation(
            landmark_for_name=lambda name: landmarks[self._legacy_pose_module.PoseLandmark[name].value],
            extended_side="LEFT",
        )
        right_ok, right_angle = self._check_orientation(
            landmark_for_name=lambda name: landmarks[self._legacy_pose_module.PoseLandmark[name].value],
            extended_side="RIGHT",
        )

        return self._build_pose_result_from_orientation_checks(
            left_ok=left_ok,
            right_ok=right_ok,
            left_angle=left_angle,
            right_angle=right_angle,
            pixel_landmarks=pixel_landmarks,
            nose_px=nose_px,
        )

    def _classify_with_tasks(self, bgr_frame: np.ndarray) -> PoseResult:
        frame_h, frame_w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._task_pose.detect(mp_image)
        if not result.pose_landmarks:
            return PoseResult(False, {}, None, None)

        landmarks = result.pose_landmarks[0]
        pixel_landmarks = self._to_pixel_landmarks(
            landmarks=landmarks,
            frame_w=frame_w,
            frame_h=frame_h,
            index_for_name=lambda name: self._task_pose_landmark_enum[name].value,
        )
        nose_px = pixel_landmarks.get("NOSE")

        if not self._visibility_guard_passed(
            landmarks=landmarks,
            landmark_for_name=lambda name: landmarks[self._task_pose_landmark_enum[name].value],
        ):
            return PoseResult(False, pixel_landmarks, None, nose_px)

        left_ok, left_angle = self._check_orientation(
            landmark_for_name=lambda name: landmarks[self._task_pose_landmark_enum[name].value],
            extended_side="LEFT",
        )
        right_ok, right_angle = self._check_orientation(
            landmark_for_name=lambda name: landmarks[self._task_pose_landmark_enum[name].value],
            extended_side="RIGHT",
        )

        return self._build_pose_result_from_orientation_checks(
            left_ok=left_ok,
            right_ok=right_ok,
            left_angle=left_angle,
            right_angle=right_angle,
            pixel_landmarks=pixel_landmarks,
            nose_px=nose_px,
        )

    @staticmethod
    def _build_pose_result_from_orientation_checks(
        left_ok: bool,
        right_ok: bool,
        left_angle: float,
        right_angle: float,
        pixel_landmarks: dict[str, tuple[int, int]],
        nose_px: Optional[tuple[int, int]],
    ) -> PoseResult:
        is_dab = left_ok or right_ok
        if not is_dab:
            return PoseResult(False, pixel_landmarks, None, nose_px)

        if left_ok and right_ok:
            extended_side = "LEFT" if left_angle >= right_angle else "RIGHT"
        elif left_ok:
            extended_side = "LEFT"
        else:
            extended_side = "RIGHT"

        return PoseResult(True, pixel_landmarks, extended_side, nose_px)

    def _to_pixel_landmarks(
        self,
        landmarks: list,
        frame_w: int,
        frame_h: int,
        index_for_name,
    ) -> dict[str, tuple[int, int]]:
        output: dict[str, tuple[int, int]] = {}
        names = self._REQUIRED_LANDMARKS + self._OPTIONAL_LANDMARKS
        for name in names:
            idx = index_for_name(name)
            lm = landmarks[idx]
            output[name] = self._landmark_to_px(lm.x, lm.y, frame_w, frame_h)
        return output

    def _visibility_guard_passed(self, landmarks: list, landmark_for_name) -> bool:
        for name in self._REQUIRED_LANDMARKS:
            lm = landmark_for_name(name)
            visibility = self._effective_visibility(lm)
            if visibility <= 0.5:
                return False
        return True

    def _check_orientation(self, landmark_for_name, extended_side: str) -> tuple[bool, float]:
        tucked_side = "RIGHT" if extended_side == "LEFT" else "LEFT"

        ext_shoulder = landmark_for_name(f"{extended_side}_SHOULDER")
        ext_elbow = landmark_for_name(f"{extended_side}_ELBOW")
        ext_wrist = landmark_for_name(f"{extended_side}_WRIST")
        tucked_elbow = landmark_for_name(f"{tucked_side}_ELBOW")
        nose = landmark_for_name("NOSE")

        elbow_angle = self._elbow_angle(
            shoulder=(ext_shoulder.x, ext_shoulder.y),
            elbow=(ext_elbow.x, ext_elbow.y),
            wrist=(ext_wrist.x, ext_wrist.y),
        )
        is_straight = elbow_angle > 155.0
        wrist_raised = ext_wrist.y < ext_shoulder.y
        tuck_dist = self._distance_2d(
            (nose.x, nose.y),
            (tucked_elbow.x, tucked_elbow.y),
        )
        face_tucked = tuck_dist <= 0.18

        return (is_straight and wrist_raised and face_tucked), elbow_angle

    def _init_legacy_backend(self) -> None:
        self._legacy_pose_module = mp.solutions.pose
        self._legacy_pose = self._legacy_pose_module.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._backend = "legacy"
        print("[pose] Backend: mediapipe legacy solutions")

    def _init_tasks_backend(self) -> None:
        self._task_model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._task_model_path.exists():
            self._download_pose_task_model()

        if not self._task_model_path.exists():
            print("[pose] Pose model is missing. Dab detection disabled.")
            self._backend = "disabled"
            return

        try:
            vision = mp.tasks.vision
            options = vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self._task_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
            self._task_pose = vision.PoseLandmarker.create_from_options(options)
            self._task_pose_landmark_enum = vision.PoseLandmark
            self._backend = "tasks"
            print("[pose] Backend: mediapipe tasks PoseLandmarker")
        except Exception as exc:  # noqa: BLE001
            print(f"[pose] Failed to initialize PoseLandmarker ({exc}). Dab detection disabled.")
            self._backend = "disabled"

    def _download_pose_task_model(self) -> None:
        try:
            print("[pose] Downloading pose model for MediaPipe Tasks...")
            urllib.request.urlretrieve(self._POSE_TASK_MODEL_URL, self._task_model_path)
            print(f"[pose] Model saved to {self._task_model_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[pose] Model download failed ({exc})")

    @staticmethod
    def _effective_visibility(landmark) -> float:
        visibility = getattr(landmark, "visibility", None)
        if visibility is not None:
            return float(visibility)
        presence = getattr(landmark, "presence", None)
        if presence is not None:
            return float(presence)
        return 1.0

    @staticmethod
    def _landmark_to_px(x: float, y: float, frame_w: int, frame_h: int) -> tuple[int, int]:
        px = int(np.clip(x * frame_w, 0, frame_w - 1))
        py = int(np.clip(y * frame_h, 0, frame_h - 1))
        return px, py

    @staticmethod
    def _distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    @staticmethod
    def _elbow_angle(
        shoulder: tuple[float, float],
        elbow: tuple[float, float],
        wrist: tuple[float, float],
    ) -> float:
        # Elbow-centered vectors produce an interior angle near 180 deg for a straight arm.
        vec_a = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]], dtype=np.float32)
        vec_b = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]], dtype=np.float32)

        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cos_theta = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))
