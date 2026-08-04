from datetime import UTC, datetime
from pathlib import Path

import pytest

from handstand_coach.models import (
    Keypoint,
    KeypointName,
    Pose,
    PoseFrame,
)
from handstand_coach.reading import SessionReader, SessionReadError
from handstand_coach.recording import SessionWriter
from handstand_coach.session import SessionMetadata


def test_read_metadata_restores_metadata_written_by_writer(
    tmp_path: Path,
) -> None:
    metadata = SessionMetadata(
        session_id="2026-07-23T120000Z",
        started_at_utc=datetime(
            2026,
            7,
            23,
            12,
            0,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )

    with SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=metadata,
    ) as writer:
        session_directory = writer.session_directory

    reader = SessionReader(session_directory)

    assert reader.read_metadata() == metadata


def test_iter_frames_restores_frames_written_by_writer(
    tmp_path: Path,
) -> None:
    metadata = SessionMetadata(
        session_id="2026-07-23T130000Z",
        started_at_utc=datetime(
            2026,
            7,
            23,
            13,
            0,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )
    frames = [
        PoseFrame(
            frame_index=0,
            timestamp_s=0.0,
            image_width=640,
            image_height=480,
            pose=None,
        ),
        PoseFrame(
            frame_index=1,
            timestamp_s=0.08,
            image_width=640,
            image_height=480,
            pose=Pose(
                keypoints=(
                    Keypoint(
                        name=KeypointName.LEFT_SHOULDER,
                        x=0.4,
                        y=0.3,
                        confidence=0.95,
                    ),
                )
            ),
        ),
    ]

    with SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=metadata,
    ) as writer:
        for frame in frames:
            writer.write_frame(frame)

        session_directory = writer.session_directory

    reader = SessionReader(session_directory)

    assert list(reader.iter_frames()) == frames


def test_read_metadata_reports_missing_file(
    tmp_path: Path,
) -> None:
    reader = SessionReader(tmp_path / "missing-session")

    with pytest.raises(
        SessionReadError,
        match="Unable to read session metadata",
    ):
        reader.read_metadata()


def test_read_metadata_reports_invalid_json(
    tmp_path: Path,
) -> None:
    session_directory = tmp_path / "invalid-session"
    session_directory.mkdir()

    metadata_path = session_directory / "metadata.json"
    metadata_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    reader = SessionReader(session_directory)

    with pytest.raises(
        SessionReadError,
        match="Invalid session metadata",
    ):
        reader.read_metadata()


def test_iter_frames_reports_missing_file(
    tmp_path: Path,
) -> None:
    reader = SessionReader(tmp_path / "missing-session")

    with pytest.raises(
        SessionReadError,
        match="Unable to read pose frames",
    ):
        list(reader.iter_frames())


def test_iter_frames_reports_invalid_line_number(
    tmp_path: Path,
) -> None:
    session_directory = tmp_path / "invalid-session"
    session_directory.mkdir()

    frames_path = session_directory / "poses.jsonl"
    frames_path.write_text(
        '{"frame_index":0,"timestamp_s":0.0,'
        '"image_width":640,"image_height":480,"pose":null}\n'
        "{invalid json\n",
        encoding="utf-8",
    )

    reader = SessionReader(session_directory)
    frames = reader.iter_frames()

    first_frame = next(frames)

    assert first_frame.frame_index == 0

    with pytest.raises(
        SessionReadError,
        match="line 2",
    ):
        next(frames)
