"""Application pipeline for processing individual image frames."""

import numpy as np
from numpy.typing import NDArray

from handstand_coach.estimation import PoseEstimator
from handstand_coach.models import PoseFrame

def analyze_frame(
    frame: NDArray[np.uint8],
    estimator: PoseEstimator,
    *,
    frame_index: int,
    timestamp_s: float,
) -> PoseFrame:
    """Estimate a pose and attach its source-frame metadata."""

    image_height, image_width = frame.shape[:2]
    pose = estimator.estimate(frame)

    return PoseFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        image_width=image_width,
        image_height=image_height,
        pose=pose,
    )