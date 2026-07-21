import json
from datetime import UTC, datetime
from pathlib import Path
import pytest

from handstand_coach.recording import (
    SessionWriteError,
    SessionWriter,
)
from handstand_coach.serialization import (
    pose_frame_to_record,
    session_metadata_to_record,
)
from handstand_coach.session import SessionMetadata
from handstand_coach.models import (
    Keypoint,
    KeypointName,
    Pose,
    PoseFrame,
)


def make_metadata() -> SessionMetadata:
    return SessionMetadata(
        session_id="2026-07-21T090000Z",
        started_at_utc=datetime(
            2026,
            7,
            21,
            9,
            0,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )


def make_pose_frame() -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=1280,
        image_height=720,
        pose=None,
    )


def test_construction_does_not_create_output_directory(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "sessions"

    SessionWriter(
        output_directory=output_directory,
        metadata=make_metadata(),
    )

    assert not output_directory.exists()


def test_enter_creates_session_files_and_metadata(
    tmp_path: Path,
) -> None:
    metadata = make_metadata()
    writer = SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=metadata,
    )

    with writer:
        metadata_path = writer.session_directory / "metadata.json"
        frames_path = writer.session_directory / "poses.jsonl"

        assert writer.session_directory.is_dir()
        assert metadata_path.is_file()
        assert frames_path.is_file()
        assert frames_path.read_text(encoding="utf-8") == ""

        stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert stored_metadata == session_metadata_to_record(metadata)


def test_write_frame_appends_and_flushes_frames_in_order(
    tmp_path: Path,
) -> None:
    first_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
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
    second_frame = PoseFrame(
        frame_index=1,
        timestamp_s=0.08,
        image_width=1280,
        image_height=720,
        pose=None,
    )
    writer = SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=make_metadata(),
    )

    with writer:
        writer.write_frame(first_frame)
        writer.write_frame(second_frame)

        frames_path = writer.session_directory / "poses.jsonl"
        stored_records = [
            json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()
        ]

    assert stored_records == [
        pose_frame_to_record(first_frame),
        pose_frame_to_record(second_frame),
    ]


def test_write_frame_is_allowed_only_while_open(
    tmp_path: Path,
) -> None:
    writer = SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=make_metadata(),
    )
    pose_frame = make_pose_frame()

    with pytest.raises(RuntimeError, match="not open"):
        writer.write_frame(pose_frame)

    with writer:
        writer.write_frame(pose_frame)

    with pytest.raises(RuntimeError, match="not open"):
        writer.write_frame(pose_frame)


def test_writer_cannot_be_entered_twice(
    tmp_path: Path,
) -> None:
    writer = SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=make_metadata(),
    )

    with writer:
        pass

    with pytest.raises(RuntimeError, match="only be entered once"):
        with writer:
            pass


def test_exception_inside_context_closes_writer(
    tmp_path: Path,
) -> None:
    writer = SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=make_metadata(),
    )

    with pytest.raises(ValueError, match="processing failed"):
        with writer:
            raise ValueError("processing failed")

    with pytest.raises(RuntimeError, match="not open"):
        writer.write_frame(make_pose_frame())


def test_close_is_safe_to_call_repeatedly(
    tmp_path: Path,
) -> None:
    writer = SessionWriter(
        output_directory=tmp_path / "sessions",
        metadata=make_metadata(),
    )

    with writer:
        pass

    writer.close()
    writer.close()


def test_enter_refuses_to_overwrite_existing_session(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "sessions"
    existing_session = output_directory / make_metadata().session_id
    existing_session.mkdir(parents=True)

    marker_path = existing_session / "keep.txt"
    marker_path.write_text(
        "existing session data",
        encoding="utf-8",
    )

    writer = SessionWriter(
        output_directory=output_directory,
        metadata=make_metadata(),
    )

    with pytest.raises(
        SessionWriteError,
        match="already exists",
    ):
        with writer:
            pass

    assert marker_path.read_text(encoding="utf-8") == "existing session data"


def test_enter_translates_output_filesystem_error(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "sessions"
    output_directory.write_text(
        "this path is a file",
        encoding="utf-8",
    )

    writer = SessionWriter(
        output_directory=output_directory,
        metadata=make_metadata(),
    )

    with pytest.raises(
        SessionWriteError,
        match="Unable to initialize session",
    ) as error_result:
        with writer:
            pass

    assert isinstance(error_result.value.__cause__, OSError)
