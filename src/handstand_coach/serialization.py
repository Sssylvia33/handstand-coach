"""Convert application-owned pose data into storage records."""

from datetime import UTC

from handstand_coach.models import Keypoint, Pose, PoseFrame
from handstand_coach.session import (
    SESSION_SCHEMA_VERSION,
    SessionMetadata,
)


def pose_frame_to_record(pose_frame: PoseFrame) -> dict[str, object]:
    """Convert a pose frame into a JSON-compatible record."""

    pose_record = None if pose_frame.pose is None else _pose_to_record(pose_frame.pose)

    return {
        "frame_index": pose_frame.frame_index,
        "timestamp_s": pose_frame.timestamp_s,
        "image_width": pose_frame.image_width,
        "image_height": pose_frame.image_height,
        "pose": pose_record,
    }


def session_metadata_to_record(
    metadata: SessionMetadata,
) -> dict[str, object]:
    """Convert session metadata into a JSON-compatible record."""

    started_at_utc = metadata.started_at_utc.astimezone(UTC)
    started_at_text = started_at_utc.isoformat().replace("+00:00", "Z")

    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": metadata.session_id,
        "started_at_utc": started_at_text,
        "source": str(metadata.source),
        "model": metadata.model,
        "confidence_threshold": metadata.confidence_threshold,
    }


def _pose_to_record(pose: Pose) -> dict[str, object]:
    """Convert a detected pose into a storage record."""

    return {"keypoints": [_keypoint_to_record(keypoint) for keypoint in pose.keypoints]}


def _keypoint_to_record(keypoint: Keypoint) -> dict[str, object]:
    """Convert one keypoint into a storage record."""

    return {
        "name": keypoint.name.value,
        "x": keypoint.x,
        "y": keypoint.y,
        "confidence": keypoint.confidence,
    }
