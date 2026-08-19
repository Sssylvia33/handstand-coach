"""Confidence-aware posture metrics."""

from dataclasses import dataclass

from handstand_coach.geometry import (
    Point2D,
    angle_degrees,
    normalized_to_pixel,
)
from handstand_coach.models import BodySide, KeypointName, PoseFrame


@dataclass(frozen=True, slots=True)
class JointAngle:
    """A calculated joint angle and its input confidence."""

    joint: KeypointName
    degrees: float
    confidence: float


@dataclass(frozen=True, slots=True)
class SelectedJointAngle:
    """A joint angle selected from the available anatomical sides."""

    angle: JointAngle
    source_side: BodySide


@dataclass(frozen=True, slots=True)
class PoseMetrics:
    """Exercise-independent posture measurements for one pose frame."""

    frame_index: int
    timestamp_s: float
    pose_detected: bool
    elbow_angle: SelectedJointAngle | None
    hip_angle: SelectedJointAngle | None


def select_joint_angle(
    *,
    left: JointAngle | None,
    right: JointAngle | None,
) -> SelectedJointAngle | None:
    """Select the available joint angle with the stronger confidence."""

    if left is None and right is None:
        return None
    if left is not None and right is None:
        return SelectedJointAngle(
            angle=left,
            source_side=BodySide.LEFT,
        )
    if left is None and right is not None:
        return SelectedJointAngle(
            angle=right,
            source_side=BodySide.RIGHT,
        )
    if left.confidence >= right.confidence:
        return SelectedJointAngle(
            angle=left,
            source_side=BodySide.LEFT,
        )
    else:
        return SelectedJointAngle(
            angle=right,
            source_side=BodySide.RIGHT,
        )


def calculate_joint_angle(
    pose_frame: PoseFrame,
    *,
    first_name: KeypointName,
    vertex_name: KeypointName,
    third_name: KeypointName,
    confidence_threshold: float,
) -> JointAngle | None:
    """Calculate an angle from three confidence-qualified keypoints."""

    pose = pose_frame.pose
    if pose is None:
        return None

    first = pose.get(first_name)
    vertex = pose.get(vertex_name)
    third = pose.get(third_name)

    if first is None or vertex is None or third is None:
        return None

    confidence = min(
        first.confidence,
        vertex.confidence,
        third.confidence,
    )
    if confidence < confidence_threshold:
        return None

    first_point = normalized_to_pixel(
        Point2D(x=first.x, y=first.y),
        image_width=pose_frame.image_width,
        image_height=pose_frame.image_height,
    )
    vertex_point = normalized_to_pixel(
        Point2D(x=vertex.x, y=vertex.y),
        image_width=pose_frame.image_width,
        image_height=pose_frame.image_height,
    )
    third_point = normalized_to_pixel(
        Point2D(x=third.x, y=third.y),
        image_width=pose_frame.image_width,
        image_height=pose_frame.image_height,
    )

    return JointAngle(
        joint=vertex_name,
        degrees=angle_degrees(first_point, vertex_point, third_point),
        confidence=confidence,
    )
