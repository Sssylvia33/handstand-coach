"""OpenCV video-source lifecycle and frame acquisition."""

from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np
from numpy.typing import NDArray


class FrameSource(Protocol):
    """Source that supplies sequential BGR image frames."""

    def read(self) -> NDArray[np.uint8] | None:
        """Read the next frame, or return None when finished."""
        ...


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be used."""


class OpenCVVideoSource:
    """Read sequential frames from an OpenCV video source."""

    def __init__(self, source: int | str | Path) -> None:
        self._source = source
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> Self:
        if self._capture is not None:
            raise VideoSourceError("Video source is already open")

        source = str(self._source) if isinstance(self._source, Path) else self._source
        capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Unable to open video source: {self._source!r}")

        self._capture = capture
        return self

    def read(self) -> NDArray[np.uint8] | None:
        """Read the next frame, or return None at the end of the stream."""

        if self._capture is None:
            raise VideoSourceError("Video source is not open")

        success, frame = self._capture.read()

        if not success or frame is None:
            return None

        return frame

    def close(self) -> None:
        """Release the video source if it is currently open."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
