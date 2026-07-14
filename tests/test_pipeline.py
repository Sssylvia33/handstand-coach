import numpy as np
from numpy.typing import NDArray

from handstand_coach.models import Keypoint, KeypointName, Pose
from handstand_coach.pipeline import analyze_frame

class FakePoseEstimator:
    """Controllable estimator used only by pipeline tests."""

    def __init__(self, pose: Pose | None) -> None:
        self._pose = pose
        self.received_frame: NDArray[np.uint8] | None = None

    def estimate(self, frame: NDArray[np.uint8]) -> Pose | None:
        self.received_frame = frame
        return self._pose

def test_analyze_frame_combines_pose_and_frame_metadata() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist = Keypoint(
        name=KeypointName.LEFT_WRIST,
        x=0.40,
        y=0.80,
        confidence=0.95,
    )
    pose = Pose(keypoints=(wrist,))
    estimator = FakePoseEstimator(pose)

    result = analyze_frame(
        frame,
        estimator,
        frame_index=7,
        timestamp_s=0.25,
    )

    assert result.frame_index == 7
    assert result.timestamp_s == 0.25
    assert result.image_width == 640
    assert result.image_height == 480
    assert result.pose is pose
    assert estimator.received_frame is frame

def test_analyze_frame_preserves_frame_when_no_pose_is_detected() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    estimator = FakePoseEstimator(None)

    result = analyze_frame(
        frame,
        estimator,
        frame_index=8,
        timestamp_s=0.30,
    )

    assert result.frame_index == 8
    assert result.image_width == 1280
    assert result.image_height == 720
    assert result.pose is None