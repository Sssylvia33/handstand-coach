import numpy as np
import pytest
from numpy.typing import NDArray

from handstand_coach.models import Pose, PoseFrame
from handstand_coach.stream import AnalyzedFrame, analyze_stream


class FakeFrameSource:
    """Provide a finite sequence of in-memory frames."""

    def __init__(self, frames: tuple[NDArray[np.uint8], ...]) -> None:
        self._frames = iter(frames)

    def read(self) -> NDArray[np.uint8] | None:
        return next(self._frames, None)


class FakePoseEstimator:
    """Record received frames and return no detected pose."""

    def __init__(self) -> None:
        self.received_frames: list[NDArray[np.uint8]] = []

    def estimate(self, frame: NDArray[np.uint8]) -> Pose | None:
        self.received_frames.append(frame)
        return None


class FakeClock:
    """Return predefined clock readings in sequence."""

    def __init__(self, *readings: float) -> None:
        self._readings = iter(readings)

    def __call__(self) -> float:
        return next(self._readings)


def test_analyze_stream_processes_frames_sequentially() -> None:
    first_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    second_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    source = FakeFrameSource((first_frame, second_frame))
    estimator = FakePoseEstimator()
    clock = FakeClock(
        100.0,  # Session start
        100.1,  # First frame processing starts
        100.3,  # First frame processing ends
        100.4,  # Second frame processing starts
        100.5,  # Second frame processing ends
    )

    results = list(analyze_stream(source, estimator, clock=clock))

    assert len(results) == 2

    first_result = results[0]
    assert first_result.image is first_frame
    assert first_result.pose_frame.frame_index == 0
    assert first_result.pose_frame.timestamp_s == pytest.approx(0.1)
    assert first_result.pose_frame.image_width == 640
    assert first_result.pose_frame.image_height == 480
    assert first_result.pose_frame.pose is None
    assert first_result.processing_time_s == pytest.approx(0.2)
    assert first_result.processing_fps == pytest.approx(5.0)

    second_result = results[1]
    assert second_result.image is second_frame
    assert second_result.pose_frame.frame_index == 1
    assert second_result.pose_frame.timestamp_s == pytest.approx(0.4)
    assert second_result.pose_frame.image_width == 1280
    assert second_result.pose_frame.image_height == 720
    assert second_result.processing_time_s == pytest.approx(0.1)
    assert second_result.processing_fps == pytest.approx(10.0)

    assert len(estimator.received_frames) == 2
    assert estimator.received_frames[0] is first_frame
    assert estimator.received_frames[1] is second_frame


def test_analyze_stream_stops_when_source_is_empty() -> None:
    source = FakeFrameSource(())
    estimator = FakePoseEstimator()
    clock = FakeClock(50.0)

    results = list(analyze_stream(source, estimator, clock=clock))

    assert results == []
    assert estimator.received_frames == []


@pytest.mark.parametrize(
    "processing_time_s",
    [
        -0.01,
        float("nan"),
        float("inf"),
    ],
)
def test_analyzed_frame_rejects_invalid_processing_time(
    processing_time_s: float,
) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=100,
        image_height=100,
    )

    with pytest.raises(ValueError, match="processing_time_s"):
        AnalyzedFrame(
            image=image,
            pose_frame=pose_frame,
            processing_time_s=processing_time_s,
        )


def test_zero_processing_time_has_infinite_processing_fps() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pose_frame = PoseFrame(
        frame_index=0,
        timestamp_s=0.0,
        image_width=100,
        image_height=100,
    )

    result = AnalyzedFrame(
        image=image,
        pose_frame=pose_frame,
        processing_time_s=0.0,
    )

    assert result.processing_fps == float("inf")
