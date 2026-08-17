"""Stateful tracking of posture measurements."""

from handstand_coach.metrics import JointAngle, calculate_joint_angle
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
