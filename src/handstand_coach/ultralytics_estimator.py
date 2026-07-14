"""Ultralytics implementation of the pose-estimator contract."""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from handstand_coach.models import Keypoint, KeypointName, Pose

COCO_KEYPOINT_NAMES = (
    KeypointName.NOSE,
    KeypointName.LEFT_EYE,
    KeypointName.RIGHT_EYE,
    KeypointName.LEFT_EAR,
    KeypointName.RIGHT_EAR,
    KeypointName.LEFT_SHOULDER,
    KeypointName.RIGHT_SHOULDER,
    KeypointName.LEFT_ELBOW,
    KeypointName.RIGHT_ELBOW,
    KeypointName.LEFT_WRIST,
    KeypointName.RIGHT_WRIST,
    KeypointName.LEFT_HIP,
    KeypointName.RIGHT_HIP,
    KeypointName.LEFT_KNEE,
    KeypointName.RIGHT_KNEE,
    KeypointName.LEFT_ANKLE,
    KeypointName.RIGHT_ANKLE,
)


class UltralyticsPoseEstimator:
    """Estimate poses using an Ultralytics YOLO pose model."""

    def __init__(self, model_path: str | Path) -> None:
        self._model = YOLO(str(model_path))

    def estimate(self, frame: NDArray[np.uint8]) -> Pose | None:
        """Estimate the highest-confidence person's pose."""

        self._validate_frame(frame)

        result = self._predict(frame)
        if result is None:
            return None

        return self._convert_result(result)

    def _predict(self, frame: NDArray[np.uint8]) -> Results | None:
        """Run Ultralytics inference and return the first image result."""
        results = self._model.predict(frame, verbose=False)

        if not results:
            return None

        return results[0]

    def _convert_result(self, result: Results) -> Pose | None:
        """Convert an Ultralytics result into an application-owned Pose."""

        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return None

        person_index = int(result.boxes.conf.argmax().item())
        normalized_xy = result.keypoints.xyn[person_index].cpu().numpy()
        confidence_tensor = result.keypoints.conf

        if confidence_tensor is None:
            raise RuntimeError("Ultralytics result did not include keypoint confidence")

        confidences = confidence_tensor[person_index].cpu().numpy()

        keypoints = self._build_keypoints(
            normalized_xy=normalized_xy,
            confidences=confidences,
        )

        if not keypoints:
            return None

        return Pose(keypoints=keypoints)

    @staticmethod
    def _build_keypoints(
        normalized_xy: NDArray[np.floating],
        confidences: NDArray[np.floating],
    ) -> tuple[Keypoint, ...]:
        """Convert COCO-ordered arrays into named Keypoint objects."""

        if len(normalized_xy) != len(COCO_KEYPOINT_NAMES):
            raise RuntimeError("Unexpected number of keypoints.")

        keypoints: list[Keypoint] = []

        for name, coordinates, confidence in zip(
            COCO_KEYPOINT_NAMES,
            normalized_xy,
            confidences,
            strict=True,
        ):
            x = float(coordinates[0])
            y = float(coordinates[1])
            confidence_value = float(confidence)

            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                continue

            keypoints.append(
                Keypoint(
                    name=name,
                    x=x,
                    y=y,
                    confidence=confidence_value,
                )
            )

        return tuple(keypoints)

    @staticmethod
    def _validate_frame(frame: NDArray[np.uint8]) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must have shape (height, width, 3); received {frame.shape}")

        if frame.dtype != np.uint8:
            raise ValueError(f"frame must use uint8 pixels; received {frame.dtype}")
