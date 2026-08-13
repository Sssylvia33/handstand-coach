"""Temporal processing for posture measurements."""

from math import exp, isfinite


class ExponentialSmoother:
    """Smooth sequential measurements using elapsed time."""

    def __init__(self, *, time_constant_s: float) -> None:
        if not isfinite(time_constant_s) or time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be a finite positive value")
        self._time_constant_s = time_constant_s
        self._previous_value: float | None = None
        self._previous_timestamp_s: float | None = None

    def update(
        self,
        *,
        value: float,
        timestamp_s: float,
    ) -> float:
        """Update the filter and return the smoothed value."""

        if self._previous_value is None or self._previous_timestamp_s is None:
            smoothed_value = value
        else:
            delta_time_s = timestamp_s - self._previous_timestamp_s
            alpha = 1.0 - exp(-delta_time_s / self._time_constant_s)
            smoothed_value = alpha * value + (1.0 - alpha) * self._previous_value

        self._previous_value = smoothed_value
        self._previous_timestamp_s = timestamp_s

        return smoothed_value

    def reset(self) -> None:
        """Discard previous measurements."""

        self._previous_value = None
        self._previous_timestamp_s = None
