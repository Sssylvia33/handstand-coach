"""Stateful tracking of posture measurements."""

from handstand_coach.metrics import (
    JointAngle,
    PoseMetrics,
    SelectedJointAngle,
    calculate_joint_angle,
    select_joint_angle,
)
from handstand_coach.models import KeypointName, PoseFrame
from handstand_coach.temporal import ExponentialSmoother


class JointAngleTracker:
    """Calculate and temporally smooth one configured joint angle."""

    def __init__(
        self,
        *,
        first_name: KeypointName,
        vertex_name: KeypointName,
        third_name: KeypointName,
        confidence_threshold: float,
        smoothing_time_constant_s: float,
    ) -> None:
        self._first_name = first_name
        self._vertex_name = vertex_name
        self._third_name = third_name
        self._confidence_threshold = confidence_threshold
        self._smoother = ExponentialSmoother(time_constant_s=smoothing_time_constant_s)

    def update(self, pose_frame: PoseFrame) -> JointAngle | None:
        """Return a smoothed angle, or None when the angle is unavailable."""

        joint_angle = calculate_joint_angle(
            pose_frame=pose_frame,
            first_name=self._first_name,
            vertex_name=self._vertex_name,
            third_name=self._third_name,
            confidence_threshold=self._confidence_threshold,
        )

        if joint_angle is None:
            self._smoother.reset()
            return None
        smoothed_degrees = self._smoother.update(
            value=joint_angle.degrees, timestamp_s=pose_frame.timestamp_s
        )

        return JointAngle(
            joint=joint_angle.joint, degrees=smoothed_degrees, confidence=joint_angle.confidence
        )


class BilateralJointAngleTracker:
    """Track both anatomical sides and return the stronger measurement."""

    def __init__(
        self,
        *,
        left_tracker: JointAngleTracker,
        right_tracker: JointAngleTracker,
    ) -> None:
        self._left_tracker = left_tracker
        self._right_tracker = right_tracker

    def update(self, pose_frame: PoseFrame) -> SelectedJointAngle | None:
        """Return the confidence-selected angle for one frame."""
        left_result = self._left_tracker.update(pose_frame)
        right_result = self._right_tracker.update(pose_frame)
        return select_joint_angle(left=left_result, right=right_result)


class PoseMetricsTracker:
    """Produce exercise-independent posture metrics for each pose frame."""

    def __init__(
        self,
        *,
        elbow_tracker: BilateralJointAngleTracker,
        hip_tracker: BilateralJointAngleTracker,
    ) -> None:
        self._elbow_tracker = elbow_tracker
        self._hip_tracker = hip_tracker

    def update(self, pose_frame: PoseFrame) -> PoseMetrics:
        """Return all currently supported metrics for one frame."""

        elbow_result = self._elbow_tracker.update(pose_frame)
        hip_result = self._hip_tracker.update(pose_frame)
        pose = pose_frame.pose is not None

        result = PoseMetrics(
            frame_index=pose_frame.frame_index,
            timestamp_s=pose_frame.timestamp_s,
            pose_detected=pose,
            elbow_angle=elbow_result,
            hip_angle=hip_result,
        )
        return result
