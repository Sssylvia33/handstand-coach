"""Pose skeleton rendering for OpenCV images."""

from collections.abc import Iterable
import cv2
import numpy as np
from numpy.typing import NDArray


from handstand_coach.models import Keypoint, KeypointName, PoseFrame

SKELETON_CONNECTIONS = (
    # Head
    (KeypointName.NOSE, KeypointName.LEFT_EYE),
    (KeypointName.NOSE, KeypointName.RIGHT_EYE),
    (KeypointName.LEFT_EYE, KeypointName.LEFT_EAR),
    (KeypointName.RIGHT_EYE, KeypointName.RIGHT_EAR),
    # Shoulders and arms
    (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW),
    (KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_ELBOW),
    (KeypointName.RIGHT_ELBOW, KeypointName.RIGHT_WRIST),
    # Torso
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_HIP),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP),
    # Legs
    (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE),
    (KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE),
    (KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
)


class PoseRenderer:
    """Draw confidence-filtered poses on BGR image frames."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        connection_color: tuple[int, int, int] = (0, 255, 0),
        keypoint_color: tuple[int, int, int] = (0, 165, 255),
        connection_thickness: int = 2,
        keypoint_radius: int = 4,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        if connection_thickness <= 0:
            raise ValueError("connection_thickness must be positive")

        if keypoint_radius <= 0:
            raise ValueError("keypoint_radius must be positive")

        self._confidence_threshold = confidence_threshold
        self._connection_color = connection_color
        self._keypoint_color = keypoint_color
        self._connection_thickness = connection_thickness
        self._keypoint_radius = keypoint_radius

    def render(
        self,
        image: NDArray[np.uint8],
        pose_frame: PoseFrame,
    ) -> NDArray[np.uint8]:
        """Return an annotated copy of an image."""

        self._validate_image(image, pose_frame)
        rendered = image.copy()

        if pose_frame.pose is None:
            return rendered

        reliable_keypoints = {
            keypoint.name: keypoint
            for keypoint in pose_frame.pose.keypoints
            if keypoint.confidence >= self._confidence_threshold
        }

        self._draw_connections(
            rendered,
            reliable_keypoints,
            pose_frame.image_width,
            pose_frame.image_height,
        )
        self._draw_keypoints(
            rendered,
            reliable_keypoints.values(),
            pose_frame.image_width,
            pose_frame.image_height,
        )

        return rendered

    def _draw_connections(
        self,
        image: NDArray[np.uint8],
        keypoints: dict[KeypointName, Keypoint],
        image_width: int,
        image_height: int,
    ) -> None:
        for start_name, end_name in SKELETON_CONNECTIONS:
            start = keypoints.get(start_name)
            end = keypoints.get(end_name)

            if start is None or end is None:
                continue

            cv2.line(
                image,
                self._to_pixel(start, image_width, image_height),
                self._to_pixel(end, image_width, image_height),
                self._connection_color,
                self._connection_thickness,
                cv2.LINE_AA,
            )

    def _draw_keypoints(
        self,
        image: NDArray[np.uint8],
        keypoints: Iterable[Keypoint],
        image_width: int,
        image_height: int,
    ) -> None:
        for keypoint in keypoints:
            cv2.circle(
                image,
                self._to_pixel(keypoint, image_width, image_height),
                self._keypoint_radius,
                self._keypoint_color,
                -1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _to_pixel(
        keypoint: Keypoint,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        return (
            round(keypoint.x * (image_width - 1)),
            round(keypoint.y * (image_height - 1)),
        )

    @staticmethod
    def _validate_image(
        image: NDArray[np.uint8],
        pose_frame: PoseFrame,
    ) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (height, width, 3); received {image.shape}")

        if image.dtype != np.uint8:
            raise ValueError(f"image must use uint8 pixels; received {image.dtype}")

        image_height, image_width = image.shape[:2]
        if image_width != pose_frame.image_width or image_height != pose_frame.image_height:
            raise ValueError(
                "image dimensions do not match PoseFrame metadata: "
                f"image={image_width}x{image_height}, "
                f"pose_frame={pose_frame.image_width}x{pose_frame.image_height}"
            )
