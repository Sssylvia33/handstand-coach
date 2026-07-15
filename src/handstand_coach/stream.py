"""Continuous frame-analysis pipeline."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from handstand_coach.capture import FrameSource
from handstand_coach.estimation import PoseEstimator
from handstand_coach.models import PoseFrame
from handstand_coach.pipeline import analyze_frame


@dataclass(frozen=True, slots=True)
class AnalyzedFrame:
    """Image, structured pose result, and processing performance."""

    image: NDArray[np.uint8]
    pose_frame: PoseFrame
    processing_time_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.processing_time_s) or self.processing_time_s < 0.0:
            raise ValueError("processing_time_s must be finite and non-negative")

    @property
    def processing_fps(self) -> float:
        """Maximum FPS implied by this frame's processing time."""

        if self.processing_time_s == 0.0:
            return float("inf")

        return 1.0 / self.processing_time_s


def analyze_stream(
    source: FrameSource,
    estimator: PoseEstimator,
    *,
    clock: Callable[[], float] = perf_counter,
) -> Iterator[AnalyzedFrame]:
    """Analyze frames sequentially until the source is exhausted."""

    start_time = clock()
    frame_index = 0

    while True:
        image = source.read()

        if image is None:
            break

        processing_started = clock()
        timestamp_s = processing_started - start_time

        pose_frame = analyze_frame(
            image,
            estimator,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )

        processing_time_s = clock() - processing_started

        yield AnalyzedFrame(
            image=image,
            pose_frame=pose_frame,
            processing_time_s=processing_time_s,
        )

        frame_index += 1
