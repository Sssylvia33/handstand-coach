import numpy as np
import pytest

from handstand_coach.models import KeypointName
from handstand_coach.ultralytics_estimator import (
    COCO_KEYPOINT_NAMES,
    UltralyticsPoseEstimator,
)

def test_build_keypoints_maps_coco_order_to_names() -> None:
    coordinates = np.full(
        (len(COCO_KEYPOINT_NAMES), 2),
        0.5,
        dtype=np.float32,
    )
    confidences = np.linspace(
        0.1,
        0.9,
        len(COCO_KEYPOINT_NAMES),
        dtype=np.float32,
    )

    keypoints = UltralyticsPoseEstimator._build_keypoints(
        normalized_xy=coordinates,
        confidences=confidences,
    )

    left_shoulder = keypoints[5]

    assert len(keypoints) == 17
    assert left_shoulder.name is KeypointName.LEFT_SHOULDER
    assert left_shoulder.x == pytest.approx(0.5)
    assert left_shoulder.y == pytest.approx(0.5)
    assert left_shoulder.confidence == pytest.approx(confidences[5])

def test_build_keypoints_omits_out_of_frame_coordinates() -> None:
    coordinates = np.full(
        (len(COCO_KEYPOINT_NAMES), 2),
        0.5,
        dtype=np.float32,
    )
    confidences = np.full(
        len(COCO_KEYPOINT_NAMES),
        0.9,
        dtype=np.float32,
    )
    coordinates[9] = (-0.1, 0.8)  # COCO index 9 is the left wrist.

    keypoints = UltralyticsPoseEstimator._build_keypoints(
        normalized_xy=coordinates,
        confidences=confidences,
    )
    names = {keypoint.name for keypoint in keypoints}

    assert len(keypoints) == 16
    assert KeypointName.LEFT_WRIST not in names

def test_build_keypoints_rejects_unexpected_keypoint_count() -> None:
    coordinates = np.full((16, 2), 0.5, dtype=np.float32)
    confidences = np.full(16, 0.9, dtype=np.float32)

    with pytest.raises(RuntimeError, match="Unexpected number of keypoints"):
        UltralyticsPoseEstimator._build_keypoints(
            normalized_xy=coordinates,
            confidences=confidences,
        )

@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((100, 100), dtype=np.uint8),
        np.zeros((100, 100, 4), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.float32),
    ],
)
def test_validate_frame_rejects_invalid_images(frame: np.ndarray) -> None:
    with pytest.raises(ValueError):
        UltralyticsPoseEstimator._validate_frame(frame)