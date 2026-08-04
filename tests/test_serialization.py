import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from handstand_coach.models import (
    Keypoint,
    KeypointName,
    Pose,
    PoseFrame,
)
from handstand_coach.serialization import (
    RecordValidationError,
    UnsupportedSchemaVersionError,
    pose_frame_from_record,
    pose_frame_to_record,
    session_metadata_from_record,
    session_metadata_to_record,
)
from handstand_coach.session import SessionMetadata


def test_pose_frame_to_record_preserves_detected_pose() -> None:
    pose_frame = PoseFrame(
        frame_index=1,
        timestamp_s=0.08,
        image_width=1280,
        image_height=720,
        pose=Pose(
            keypoints=(
                Keypoint(
                    name=KeypointName.LEFT_SHOULDER,
                    x=0.42,
                    y=0.31,
                    confidence=0.96,
                ),
            )
        ),
    )

    record = pose_frame_to_record(pose_frame)

    assert record == {
        "frame_index": 1,
        "timestamp_s": 0.08,
        "image_width": 1280,
        "image_height": 720,
        "pose": {
            "keypoints": [
                {
                    "name": "left_shoulder",
                    "x": 0.42,
                    "y": 0.31,
                    "confidence": 0.96,
                }
            ]
        },
    }


def test_pose_frame_to_record_preserves_missing_pose() -> None:
    pose_frame = PoseFrame(
        frame_index=2,
        timestamp_s=0.16,
        image_width=1280,
        image_height=720,
        pose=None,
    )

    record = pose_frame_to_record(pose_frame)

    assert record == {
        "frame_index": 2,
        "timestamp_s": 0.16,
        "image_width": 1280,
        "image_height": 720,
        "pose": None,
    }


def test_pose_frame_record_is_json_serializable() -> None:
    pose_frame = PoseFrame(
        frame_index=3,
        timestamp_s=0.24,
        image_width=1280,
        image_height=720,
        pose=Pose(
            keypoints=(
                Keypoint(
                    name=KeypointName.LEFT_WRIST,
                    x=0.40,
                    y=0.82,
                    confidence=0.91,
                ),
            )
        ),
    )

    record = pose_frame_to_record(pose_frame)

    json.dumps(record)


def test_session_metadata_to_record_preserves_session_configuration() -> None:
    metadata = SessionMetadata(
        session_id="2026-07-20T183000Z",
        started_at_utc=datetime(
            2026,
            7,
            20,
            18,
            30,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )

    record = session_metadata_to_record(metadata)

    assert record == {
        "schema_version": 1,
        "session_id": "2026-07-20T183000Z",
        "started_at_utc": "2026-07-20T18:30:00Z",
        "source": "0",
        "model": "yolov8n-pose.pt",
        "confidence_threshold": 0.5,
    }


def test_session_metadata_to_record_normalizes_start_time_to_utc() -> None:
    helsinki_summer_time = timezone(timedelta(hours=3))
    metadata = SessionMetadata(
        session_id="2026-07-20T183000Z",
        started_at_utc=datetime(
            2026,
            7,
            20,
            21,
            30,
            tzinfo=helsinki_summer_time,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )

    record = session_metadata_to_record(metadata)

    assert record["started_at_utc"] == "2026-07-20T18:30:00Z"


def test_pose_frame_round_trip_preserves_detected_pose() -> None:
    original = PoseFrame(
        frame_index=8,
        timestamp_s=0.64,
        image_width=1280,
        image_height=720,
        pose=Pose(
            keypoints=(
                Keypoint(
                    name=KeypointName.LEFT_SHOULDER,
                    x=0.42,
                    y=0.31,
                    confidence=0.96,
                ),
                Keypoint(
                    name=KeypointName.RIGHT_WRIST,
                    x=0.61,
                    y=0.78,
                    confidence=0.88,
                ),
            )
        ),
    )

    record = pose_frame_to_record(original)
    restored = pose_frame_from_record(record)

    assert restored == original


def test_pose_frame_round_trip_preserves_missing_pose() -> None:
    original = PoseFrame(
        frame_index=9,
        timestamp_s=0.72,
        image_width=1280,
        image_height=720,
        pose=None,
    )

    record = pose_frame_to_record(original)
    restored = pose_frame_from_record(record)

    assert restored == original


@pytest.mark.parametrize(
    "missing_field",
    [
        "frame_index",
        "timestamp_s",
        "image_width",
        "image_height",
        "pose",
    ],
)
def test_pose_frame_from_record_rejects_missing_required_field(
    missing_field: str,
) -> None:
    original = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=1280,
        image_height=720,
        pose=None,
    )
    record = pose_frame_to_record(original)
    del record[missing_field]

    with pytest.raises(
        RecordValidationError,
        match=missing_field,
    ):
        pose_frame_from_record(record)


def test_pose_frame_from_record_rejects_unknown_keypoint_name() -> None:
    record = {
        "frame_index": 0,
        "timestamp_s": 0.0,
        "image_width": 1280,
        "image_height": 720,
        "pose": {
            "keypoints": [
                {
                    "name": "tail",
                    "x": 0.5,
                    "y": 0.5,
                    "confidence": 0.9,
                }
            ]
        },
    }

    with pytest.raises(
        RecordValidationError,
        match="Invalid pose frame record",
    ):
        pose_frame_from_record(record)


def test_session_metadata_round_trip_preserves_configuration() -> None:
    original = SessionMetadata(
        session_id="2026-07-23T090000.123456Z",
        started_at_utc=datetime(
            2026,
            7,
            23,
            9,
            0,
            0,
            123456,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )

    record = session_metadata_to_record(original)
    restored = session_metadata_from_record(record)

    assert restored == original


def test_session_metadata_from_record_rejects_unsupported_schema() -> None:
    metadata = SessionMetadata(
        session_id="2026-07-23T090000Z",
        started_at_utc=datetime(
            2026,
            7,
            23,
            9,
            0,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )
    record = session_metadata_to_record(metadata)
    record["schema_version"] = 2

    with pytest.raises(
        UnsupportedSchemaVersionError,
        match="2",
    ):
        session_metadata_from_record(record)
