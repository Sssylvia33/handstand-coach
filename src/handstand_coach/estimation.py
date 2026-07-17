"""Interfaces for pose-estimation implementations."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from handstand_coach.models import Pose


class PoseEstimator(Protocol):
    """Contract implemented by pose-estimation providers."""

    def estimate(self, frame: NDArray[np.uint8]) -> Pose | None:
        """Estimate one person's pose from a BGR image frame."""
        ...


class PoseModelLoadError(RuntimeError):
    """Raised when a pose-estimation model cannot be loaded."""
