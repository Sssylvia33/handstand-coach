from math import log

import pytest

from handstand_coach.models import Keypoint, KeypointName, Pose, PoseFrame
from handstand_coach.tracking import JointAngleTracker


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
