import pytest

from handstand_coach.metrics import JointAngle, calculate_joint_angle, select_joint_angle
from handstand_coach.models import BodySide, Keypoint, KeypointName, Pose, PoseFrame


def make_joint_angle(
    *,
    joint: KeypointName,
    degrees: float,
    confidence: float,
) -> JointAngle:
    return JointAngle(
        joint=joint,
        degrees=degrees,
        confidence=confidence,
    )


def test_select_joint_angle_returns_only_available_side() -> None:
    right = make_joint_angle(
        joint=KeypointName.RIGHT_HIP,
        degrees=92.0,
        confidence=0.8,
    )

    result = select_joint_angle(left=None, right=right)

    assert result is not None
    assert result.angle is right
    assert result.source_side is BodySide.RIGHT


def test_select_joint_angle_prefers_higher_confidence() -> None:
    left = make_joint_angle(
        joint=KeypointName.LEFT_HIP,
        degrees=90.0,
        confidence=0.9,
    )
    right = make_joint_angle(
        joint=KeypointName.RIGHT_HIP,
        degrees=94.0,
        confidence=0.7,
    )

    result = select_joint_angle(left=left, right=right)

    assert result is not None
    assert result.angle is left
    assert result.source_side is BodySide.LEFT


def test_select_joint_angle_returns_none_when_both_sides_are_unavailable() -> None:
    assert select_joint_angle(left=None, right=None) is None


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
