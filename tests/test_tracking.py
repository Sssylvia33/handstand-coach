from math import log
from unittest.mock import Mock

import pytest

from handstand_coach.metrics import JointAngle, SelectedJointAngle
from handstand_coach.models import BodySide, Keypoint, KeypointName, Pose, PoseFrame
from handstand_coach.tracking import (
    BilateralJointAngleTracker,
    JointAngleTracker,
    PoseMetricsTracker,
)


def make_left_elbow_frame(
    *,
    timestamp_s: float,
    shoulder: tuple[float, float],
    wrist: tuple[float, float],
) -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        timestamp_s=timestamp_s,
        image_width=101,
        image_height=101,
        pose=Pose(
            keypoints=(
                Keypoint(
                    name=KeypointName.LEFT_SHOULDER,
                    x=shoulder[0],
                    y=shoulder[1],
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
                    x=wrist[0],
                    y=wrist[1],
                    confidence=0.9,
                ),
            )
        ),
    )


def make_left_elbow_tracker() -> JointAngleTracker:
    return JointAngleTracker(
        first_name=KeypointName.LEFT_SHOULDER,
        vertex_name=KeypointName.LEFT_ELBOW,
        third_name=KeypointName.LEFT_WRIST,
        confidence_threshold=0.5,
        smoothing_time_constant_s=1.0,
    )


def make_elbow_tracker(side: BodySide) -> JointAngleTracker:
    if side is BodySide.LEFT:
        names = (
            KeypointName.LEFT_SHOULDER,
            KeypointName.LEFT_ELBOW,
            KeypointName.LEFT_WRIST,
        )
    else:
        names = (
            KeypointName.RIGHT_SHOULDER,
            KeypointName.RIGHT_ELBOW,
            KeypointName.RIGHT_WRIST,
        )

    return JointAngleTracker(
        first_name=names[0],
        vertex_name=names[1],
        third_name=names[2],
        confidence_threshold=0.5,
        smoothing_time_constant_s=1.0,
    )


def test_bilateral_tracker_selects_stronger_side() -> None:
    tracker = BilateralJointAngleTracker(
        left_tracker=make_elbow_tracker(BodySide.LEFT),
        right_tracker=make_elbow_tracker(BodySide.RIGHT),
    )
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=101,
        image_height=101,
        pose=Pose(
            keypoints=(
                Keypoint(KeypointName.LEFT_SHOULDER, 0.2, 0.5, 0.7),
                Keypoint(KeypointName.LEFT_ELBOW, 0.5, 0.5, 0.7),
                Keypoint(KeypointName.LEFT_WRIST, 0.5, 0.8, 0.7),
                Keypoint(KeypointName.RIGHT_SHOULDER, 0.2, 0.5, 0.9),
                Keypoint(KeypointName.RIGHT_ELBOW, 0.5, 0.5, 0.9),
                Keypoint(KeypointName.RIGHT_WRIST, 0.5, 0.8, 0.9),
            )
        ),
    )

    result = tracker.update(pose_frame)

    assert result is not None
    assert result.source_side is BodySide.RIGHT
    assert result.angle.joint is KeypointName.RIGHT_ELBOW
    assert result.angle.confidence == pytest.approx(0.9)


def test_pose_metrics_tracker_assembles_frame_measurements() -> None:
    elbow_result = SelectedJointAngle(
        angle=JointAngle(KeypointName.LEFT_ELBOW, 170.0, 0.9),
        source_side=BodySide.LEFT,
    )
    hip_result = SelectedJointAngle(
        angle=JointAngle(KeypointName.RIGHT_HIP, 160.0, 0.8),
        source_side=BodySide.RIGHT,
    )
    elbow_tracker = Mock(spec=BilateralJointAngleTracker)
    elbow_tracker.update.return_value = elbow_result
    hip_tracker = Mock(spec=BilateralJointAngleTracker)
    hip_tracker.update.return_value = hip_result
    tracker = PoseMetricsTracker(
        elbow_tracker=elbow_tracker,
        hip_tracker=hip_tracker,
    )
    pose_frame = PoseFrame(
        frame_index=12,
        timestamp_s=1.5,
        image_width=640,
        image_height=480,
        pose=Pose(keypoints=(Keypoint(KeypointName.NOSE, 0.5, 0.2, 0.9),)),
    )

    result = tracker.update(pose_frame)

    assert result.frame_index == 12
    assert result.timestamp_s == pytest.approx(1.5)
    assert result.pose_detected is True
    assert result.elbow_angle is elbow_result
    assert result.hip_angle is hip_result
    elbow_tracker.update.assert_called_once_with(pose_frame)
    hip_tracker.update.assert_called_once_with(pose_frame)


def test_joint_angle_tracker_smooths_using_frame_timestamp() -> None:
    tracker = make_left_elbow_tracker()
    right_angle_frame = make_left_elbow_frame(
        timestamp_s=0.0,
        shoulder=(0.5, 0.8),
        wrist=(0.8, 0.5),
    )
    straight_angle_frame = make_left_elbow_frame(
        timestamp_s=log(2.0),
        shoulder=(0.2, 0.5),
        wrist=(0.8, 0.5),
    )

    first_result = tracker.update(right_angle_frame)
    second_result = tracker.update(straight_angle_frame)

    assert first_result is not None
    assert first_result.degrees == pytest.approx(90.0)
    assert second_result is not None
    assert second_result.joint is KeypointName.LEFT_ELBOW
    assert second_result.degrees == pytest.approx(135.0)
    assert second_result.confidence == pytest.approx(0.9)


def test_joint_angle_tracker_resets_after_unavailable_angle() -> None:
    tracker = make_left_elbow_tracker()
    tracker.update(
        make_left_elbow_frame(
            timestamp_s=0.0,
            shoulder=(0.5, 0.8),
            wrist=(0.8, 0.5),
        )
    )
    missing_pose_frame = PoseFrame(
        frame_index=1,
        timestamp_s=1.0,
        image_width=101,
        image_height=101,
        pose=None,
    )

    unavailable_result = tracker.update(missing_pose_frame)
    result_after_gap = tracker.update(
        make_left_elbow_frame(
            timestamp_s=2.0,
            shoulder=(0.2, 0.5),
            wrist=(0.8, 0.5),
        )
    )

    assert unavailable_result is None
    assert result_after_gap is not None
    assert result_after_gap.degrees == pytest.approx(180.0)
