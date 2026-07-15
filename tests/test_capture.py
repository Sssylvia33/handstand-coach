from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import handstand_coach.capture as capture_module
from handstand_coach.capture import OpenCVVideoSource, VideoSourceError


class FakeVideoCapture:
    """In-memory substitute for cv2.VideoCapture."""

    def __init__(
        self,
        *,
        opened: bool = True,
        frames: tuple[NDArray[np.uint8], ...] = (),
    ) -> None:
        self._opened = opened
        self._frames = list(frames)
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if not self._opened or not self._frames:
            return False, None

        return True, self._frames.pop(0)

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False


def replace_video_capture(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeVideoCapture,
) -> list[int | str]:
    """Replace OpenCV construction and record requested sources."""

    requested_sources: list[int | str] = []

    def factory(source: int | str) -> FakeVideoCapture:
        requested_sources.append(source)
        return fake

    monkeypatch.setattr(
        capture_module.cv2,
        "VideoCapture",
        factory,
    )
    return requested_sources


def test_video_source_reads_frames_and_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake = FakeVideoCapture(frames=(frame,))
    requested_sources = replace_video_capture(monkeypatch, fake)

    with OpenCVVideoSource(0) as source:
        result = source.read()
        end_of_stream = source.read()

    assert result is frame
    assert end_of_stream is None
    assert requested_sources == [0]
    assert fake.release_calls == 1


def test_video_source_converts_path_to_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeVideoCapture()
    requested_sources = replace_video_capture(monkeypatch, fake)

    with OpenCVVideoSource(Path("example.mp4")):
        pass

    assert requested_sources == ["example.mp4"]


def test_video_source_releases_capture_that_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeVideoCapture(opened=False)
    replace_video_capture(monkeypatch, fake)

    with pytest.raises(VideoSourceError, match="Unable to open"):
        with OpenCVVideoSource(0):
            pass

    assert fake.release_calls == 1


def test_video_source_releases_capture_after_processing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeVideoCapture()
    replace_video_capture(monkeypatch, fake)

    with pytest.raises(RuntimeError, match="Inference failed"):
        with OpenCVVideoSource(0):
            raise RuntimeError("Inference failed")

    assert fake.release_calls == 1


def test_video_source_close_is_safe_to_call_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeVideoCapture()
    replace_video_capture(monkeypatch, fake)
    source = OpenCVVideoSource(0)

    with source:
        source.close()
        source.close()

    assert fake.release_calls == 1


def test_video_source_rejects_read_before_opening() -> None:
    source = OpenCVVideoSource(0)

    with pytest.raises(VideoSourceError, match="not open"):
        source.read()
