"""Application-owned pose data types."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class KeypointName(StrEnum):
    """Body keypoints supported by the application."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


@dataclass(frozen=True, slots=True)
class Keypoint:
    """A body keypoint with coordinates normalized to its source frame."""

    name: KeypointName
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        values = {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }

        for field_name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0; received {value}")


@dataclass(frozen=True, slots=True)
class Pose:
    """Keypoints belonging to one detected person."""

    keypoints: tuple[Keypoint, ...]

    def __post_init__(self) -> None:
        if not self.keypoints:
            raise ValueError("Pose must contain at least one keypoint")

        names = [keypoint.name for keypoint in self.keypoints]
        if len(names) != len(set(names)):
            raise ValueError("Pose cannot contain duplicate keypoint names")

    def get(self, name: KeypointName) -> Keypoint | None:
        """Return a keypoint by name, or None when it is unavailable."""

        for keypoint in self.keypoints:
            if keypoint.name is name:
                return keypoint

        return None


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Structured pose information produced for one source frame."""

    frame_index: int
    timestamp_s: float
    image_width: int
    image_height: int
    pose: Pose | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")

        if not isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be a finite non-negative value")

        if self.image_width <= 0:
            raise ValueError("image_width must be positive")

        if self.image_height <= 0:
            raise ValueError("image_height must be positive")
