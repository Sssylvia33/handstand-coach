import pytest

from handstand_coach.metrics import calculate_joint_angle
from handstand_coach.models import Keypoint, KeypointName, Pose, PoseFrame


def test_calculate_joint_angle_returns_structured_result() -> None:
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=101,
        image_height=101,
        pose=Pose(
            keypoints=(
                Keypoint(
                    name=KeypointName.LEFT_SHOULDER,
                    x=0.5,
                    y=0.8,
                    confidence=0.95,
                ),
                Keypoint(
                    name=KeypointName.LEFT_ELBOW,
                    x=0.5,
                    y=0.5,
                    confidence=0.90,
                ),
                Keypoint(
                    name=KeypointName.LEFT_WRIST,
                    x=0.8,
                    y=0.5,
                    confidence=0.80,
                ),
            )
        ),
    )

    result = calculate_joint_angle(
        pose_frame,
        first_name=KeypointName.LEFT_SHOULDER,
        vertex_name=KeypointName.LEFT_ELBOW,
        third_name=KeypointName.LEFT_WRIST,
        confidence_threshold=0.5,
    )

    assert result is not None
    assert result.joint is KeypointName.LEFT_ELBOW
    assert result.degrees == pytest.approx(90.0)
    assert result.confidence == pytest.approx(0.80)


def test_calculate_joint_angle_returns_none_when_pose_is_missing() -> None:
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=640,
        image_height=480,
        pose=None,
    )

    result = calculate_joint_angle(
        pose_frame,
        first_name=KeypointName.LEFT_SHOULDER,
        vertex_name=KeypointName.LEFT_ELBOW,
        third_name=KeypointName.LEFT_WRIST,
        confidence_threshold=0.5,
    )

    assert result is None


@pytest.mark.parametrize(
    "missing_name",
    [
        KeypointName.LEFT_SHOULDER,
        KeypointName.LEFT_ELBOW,
        KeypointName.LEFT_WRIST,
    ],
)
def test_calculate_joint_angle_returns_none_when_keypoint_is_missing(
    missing_name: KeypointName,
) -> None:
    complete_keypoints = (
        Keypoint(
            name=KeypointName.LEFT_SHOULDER,
            x=0.5,
            y=0.8,
            confidence=0.9,
        ),
        Keypoint(
            name=KeypointName.LEFT_ELBOW,
            x=0.5,
            y=0.5,
            confidence=0.9,
        ),
        Keypoint(
            name=KeypointName.LEFT_WRIST,
            x=0.8,
            y=0.5,
            confidence=0.9,
        ),
    )
    available_keypoints = tuple(
        keypoint for keypoint in complete_keypoints if keypoint.name is not missing_name
    )
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=101,
        image_height=101,
        pose=Pose(keypoints=available_keypoints),
    )

    result = calculate_joint_angle(
        pose_frame,
        first_name=KeypointName.LEFT_SHOULDER,
        vertex_name=KeypointName.LEFT_ELBOW,
        third_name=KeypointName.LEFT_WRIST,
        confidence_threshold=0.5,
    )

    assert result is None


def test_calculate_joint_angle_returns_none_when_confidence_is_low() -> None:
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=101,
        image_height=101,
        pose=Pose(
            keypoints=(
                Keypoint(
                    name=KeypointName.LEFT_SHOULDER,
                    x=0.5,
                    y=0.8,
                    confidence=0.95,
                ),
                Keypoint(
                    name=KeypointName.LEFT_ELBOW,
                    x=0.5,
                    y=0.5,
                    confidence=0.90,
                ),
                Keypoint(
                    name=KeypointName.LEFT_WRIST,
                    x=0.8,
                    y=0.5,
                    confidence=0.49,
                ),
            )
        ),
    )

    result = calculate_joint_angle(
        pose_frame,
        first_name=KeypointName.LEFT_SHOULDER,
        vertex_name=KeypointName.LEFT_ELBOW,
        third_name=KeypointName.LEFT_WRIST,
        confidence_threshold=0.5,
    )

    assert result is None
