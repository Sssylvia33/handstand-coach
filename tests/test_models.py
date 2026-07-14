import pytest

from handstand_coach.models import Keypoint, KeypointName, Pose, PoseFrame


def test_keypoint_stores_normalized_values() -> None:
    keypoint = Keypoint(
        name=KeypointName.LEFT_SHOULDER,
        x=0.25,
        y=0.75,
        confidence=0.90,
    )

    assert keypoint.name is KeypointName.LEFT_SHOULDER
    assert keypoint.x == 0.25
    assert keypoint.y == 0.75
    assert keypoint.confidence == 0.90


@pytest.mark.parametrize(
    ("x", "y", "confidence"),
    [
        (-0.01, 0.50, 0.90),
        (1.01, 0.50, 0.90),
        (0.50, -0.01, 0.90),
        (0.50, 1.01, 0.90),
        (0.50, 0.50, -0.01),
        (0.50, 0.50, 1.01),
    ],
)
def test_keypoint_rejects_values_outside_normalized_range(
    x: float,
    y: float,
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="must be between"):
        Keypoint(
            name=KeypointName.LEFT_SHOULDER,
            x=x,
            y=y,
            confidence=confidence,
        )


def test_pose_returns_keypoint_by_name() -> None:
    shoulder = Keypoint(
        name=KeypointName.LEFT_SHOULDER,
        x=0.25,
        y=0.40,
        confidence=0.95,
    )
    wrist = Keypoint(
        name=KeypointName.LEFT_WRIST,
        x=0.30,
        y=0.80,
        confidence=0.90,
    )
    pose = Pose(keypoints=(shoulder, wrist))

    assert pose.get(KeypointName.LEFT_SHOULDER) is shoulder
    assert pose.get(KeypointName.LEFT_WRIST) is wrist
    assert pose.get(KeypointName.RIGHT_WRIST) is None


def test_pose_rejects_empty_keypoints() -> None:
    with pytest.raises(ValueError, match="at least one keypoint"):
        Pose(keypoints=())


def test_pose_rejects_duplicate_keypoint_names() -> None:
    first_shoulder = Keypoint(
        name=KeypointName.LEFT_SHOULDER,
        x=0.25,
        y=0.40,
        confidence=0.95,
    )
    second_shoulder = Keypoint(
        name=KeypointName.LEFT_SHOULDER,
        x=0.30,
        y=0.45,
        confidence=0.85,
    )

    with pytest.raises(ValueError, match="duplicate"):
        Pose(keypoints=(first_shoulder, second_shoulder))


def test_pose_frame_can_represent_detected_pose() -> None:
    wrist = Keypoint(
        name=KeypointName.LEFT_WRIST,
        x=0.40,
        y=0.80,
        confidence=0.95,
    )
    pose = Pose(keypoints=(wrist,))

    pose_frame = PoseFrame(
        frame_index=3,
        timestamp_s=0.10,
        image_width=1920,
        image_height=1080,
        pose=pose,
    )

    assert pose_frame.frame_index == 3
    assert pose_frame.timestamp_s == 0.10
    assert pose_frame.pose is pose


def test_pose_frame_can_represent_no_detection() -> None:
    pose_frame = PoseFrame(
        frame_index=4,
        timestamp_s=0.13,
        image_width=1920,
        image_height=1080,
        pose=None,
    )

    assert pose_frame.pose is None
    assert pose_frame.image_width == 1920
    assert pose_frame.image_height == 1080


@pytest.mark.parametrize(
    ("frame_index", "timestamp_s", "image_width", "image_height"),
    [
        (-1, 0.0, 1920, 1080),
        (0, -0.01, 1920, 1080),
        (0, float("nan"), 1920, 1080),
        (0, 0.0, 0, 1080),
        (0, 0.0, 1920, -1),
    ],
)
def test_pose_frame_rejects_invalid_metadata(
    frame_index: int,
    timestamp_s: float,
    image_width: int,
    image_height: int,
) -> None:
    with pytest.raises(ValueError):
        PoseFrame(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            image_width=image_width,
            image_height=image_height,
        )
