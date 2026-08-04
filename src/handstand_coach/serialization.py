"""Convert between application pose objects and storage records."""

from collections.abc import Mapping
from datetime import UTC, datetime

from handstand_coach.models import (
    Keypoint,
    KeypointName,
    Pose,
    PoseFrame,
)
from handstand_coach.session import (
    SESSION_SCHEMA_VERSION,
    SessionMetadata,
)


class RecordValidationError(ValueError):
    """Raised when a storage record violates the data contract."""


class UnsupportedSchemaVersionError(RecordValidationError):
    """Raised when no reader exists for a record's schema version."""


def pose_frame_to_record(pose_frame: PoseFrame) -> dict[str, object]:
    """Convert a pose frame into a JSON-compatible storage record."""
    pose_record = None if pose_frame.pose is None else _pose_to_record(pose_frame.pose)

    return {
        "frame_index": pose_frame.frame_index,
        "timestamp_s": pose_frame.timestamp_s,
        "image_width": pose_frame.image_width,
        "image_height": pose_frame.image_height,
        "pose": pose_record,
    }


def pose_frame_from_record(
    record: Mapping[str, object],
) -> PoseFrame:
    """Restore a validated pose frame from a storage record."""

    try:
        pose_value = _require_field(record, "pose")
        pose = (
            None if pose_value is None else _pose_from_record(_require_mapping(pose_value, "pose"))
        )

        return PoseFrame(
            frame_index=_require_int(
                _require_field(record, "frame_index"),
                "frame_index",
            ),
            timestamp_s=_require_number(
                _require_field(record, "timestamp_s"),
                "timestamp_s",
            ),
            image_width=_require_int(
                _require_field(record, "image_width"),
                "image_width",
            ),
            image_height=_require_int(
                _require_field(record, "image_height"),
                "image_height",
            ),
            pose=pose,
        )
    except RecordValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise RecordValidationError(f"Invalid pose frame record: {error}") from error


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


def session_metadata_from_record(
    record: Mapping[str, object],
) -> SessionMetadata:
    """Restore validated session metadata from a storage record."""

    try:
        schema_version = _require_int(
            _require_field(record, "schema_version"),
            "schema_version",
        )

        if schema_version != SESSION_SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(f"Unsupported schema version: {schema_version}")
        started_at_text = _require_string(
            _require_field(record, "started_at_utc"),
            "started_at_utc",
        )
        iso_timestamp = (
            f"{started_at_text[:-1]}+00:00" if started_at_text.endswith("Z") else started_at_text
        )
        started_at_utc = datetime.fromisoformat(iso_timestamp).astimezone(UTC)

        source_text = _require_string(
            _require_field(record, "source"),
            "source",
        )
        source: int | str = int(source_text) if source_text.isdecimal() else source_text

        return SessionMetadata(
            session_id=_require_string(
                _require_field(record, "session_id"),
                "session_id",
            ),
            started_at_utc=started_at_utc,
            source=source,
            model=_require_string(
                _require_field(record, "model"),
                "model",
            ),
            confidence_threshold=_require_number(
                _require_field(
                    record,
                    "confidence_threshold",
                ),
                "confidence_threshold",
            ),
        )
    except RecordValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise RecordValidationError(f"Invalid session metadata record: {error}") from error


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


def _pose_from_record(
    record: Mapping[str, object],
) -> Pose:
    keypoints_value = _require_field(
        record,
        "keypoints",
    )

    if not isinstance(keypoints_value, list):
        raise TypeError("keypoints must be a list")

    keypoints = tuple(
        _keypoint_from_record(_require_mapping(value, "keypoint")) for value in keypoints_value
    )

    return Pose(keypoints=keypoints)


def _keypoint_from_record(
    record: Mapping[str, object],
) -> Keypoint:
    name_value = _require_string(
        _require_field(record, "name"),
        "name",
    )

    return Keypoint(
        name=KeypointName(name_value),
        x=_require_number(
            _require_field(record, "x"),
            "x",
        ),
        y=_require_number(
            _require_field(record, "y"),
            "y",
        ),
        confidence=_require_number(
            _require_field(record, "confidence"),
            "confidence",
        ),
    )


def _require_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecordValidationError(f"{field_name} must be an object")

    return value


def _require_int(
    value: object,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordValidationError(f"{field_name} must be an integer")

    return value


def _require_number(
    value: object,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{field_name} must be a number")

    return float(value)


def _require_string(
    value: object,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field_name} must be a string")

    return value


def _require_field(
    record: Mapping[str, object],
    field_name: str,
) -> object:
    try:
        return record[field_name]
    except KeyError as error:
        raise RecordValidationError(f"Missing required field: {field_name}") from error
