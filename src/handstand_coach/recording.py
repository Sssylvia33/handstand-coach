"""Persist structured pose sessions to the filesystem."""

import json
from enum import Enum, auto
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

from handstand_coach.serialization import (
    pose_frame_to_record,
    session_metadata_to_record,
)
from handstand_coach.session import SessionMetadata
from handstand_coach.models import PoseFrame


class _WriterState(Enum):
    NEW = auto()
    OPEN = auto()
    CLOSED = auto()


class SessionWriteError(RuntimeError):
    """Raised when a structured session cannot be written."""


class SessionWriter:
    """Write one structured pose session."""

    def __init__(
        self,
        *,
        output_directory: str | Path,
        metadata: SessionMetadata,
    ) -> None:
        self._output_directory = Path(output_directory)
        self._metadata = metadata
        self._state = _WriterState.NEW
        self._frames_file: TextIO | None = None

    @property
    def session_directory(self) -> Path:
        """Return the directory belonging to this session."""

        return self._output_directory / self._metadata.session_id

    def __enter__(self) -> Self:
        if self._state is not _WriterState.NEW:
            raise RuntimeError("SessionWriter can only be entered once")

        try:
            self.session_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            metadata_record = session_metadata_to_record(self._metadata)
            metadata_text = json.dumps(
                metadata_record,
                indent=2,
            )

            metadata_path = self.session_directory / "metadata.json"
            metadata_path.write_text(
                f"{metadata_text}\n",
                encoding="utf-8",
                newline="\n",
            )

            frames_path = self.session_directory / "poses.jsonl"
            self._frames_file = frames_path.open(
                "x",
                encoding="utf-8",
                newline="\n",
            )
        except OSError as error:
            self.close()

            session_already_exists = (
                isinstance(error, FileExistsError) and self.session_directory.exists()
            )

            if session_already_exists:
                message = f"Session directory already exists: {self.session_directory}"
            else:
                message = f"Unable to initialize session: {self.session_directory}"

            raise SessionWriteError(message) from error
        except Exception:
            self.close()
            raise

        self._state = _WriterState.OPEN
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close resources owned by this writer."""
        frames_file = self._frames_file
        self._frames_file = None
        self._state = _WriterState.CLOSED

        if frames_file is not None:
            frames_file.close()

    def write_frame(self, pose_frame: PoseFrame) -> None:
        """Append one pose-frame record to this session."""

        if self._state is not _WriterState.OPEN or self._frames_file is None:
            raise RuntimeError("SessionWriter is not open")

        record = pose_frame_to_record(pose_frame)
        line = json.dumps(
            record,
            separators=(",", ":"),
        )

        self._frames_file.write(line)
        self._frames_file.write("\n")
        self._frames_file.flush()
