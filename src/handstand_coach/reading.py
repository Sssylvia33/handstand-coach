"""Load structured pose sessions from the filesystem."""

import json
from collections.abc import Iterator
from pathlib import Path

from handstand_coach.models import PoseFrame
from handstand_coach.serialization import (
    RecordValidationError,
    pose_frame_from_record,
    session_metadata_from_record,
)
from handstand_coach.session import SessionMetadata


class SessionReadError(RuntimeError):
    """Raised when a structured session cannot be read."""


class SessionReader:
    """Read one structured pose session."""

    def __init__(self, session_directory: str | Path) -> None:
        self._session_directory = Path(session_directory)

    @property
    def session_directory(self) -> Path:
        """Return the directory containing this session."""

        return self._session_directory

    def read_metadata(self) -> SessionMetadata:
        """Read and deserialize this session's metadata."""

        metadata_path = self._session_directory / "metadata.json"

        try:
            metadata_text = metadata_path.read_text(encoding="utf-8")
        except OSError as error:
            raise SessionReadError(f"Unable to read session metadata: {metadata_path}") from error

        try:
            metadata_record = json.loads(metadata_text)
            return session_metadata_from_record(metadata_record)
        except (
            json.JSONDecodeError,
            RecordValidationError,
        ) as error:
            raise SessionReadError(f"Invalid session metadata: {metadata_path}") from error

    def iter_frames(self) -> Iterator[PoseFrame]:
        """Yield deserialized pose frames in recording order."""

        frames_path = self._session_directory / "poses.jsonl"

        try:
            with frames_path.open(
                "r",
                encoding="utf-8",
            ) as frames_file:
                for line_number, line in enumerate(
                    frames_file,
                    start=1,
                ):
                    try:
                        frame_record = json.loads(line)
                        yield pose_frame_from_record(frame_record)
                    except (
                        json.JSONDecodeError,
                        RecordValidationError,
                    ) as error:
                        raise SessionReadError(
                            f"Invalid pose frame at line {line_number}: {frames_path}"
                        ) from error
        except OSError as error:
            raise SessionReadError(f"Unable to read pose frames: {frames_path}") from error
