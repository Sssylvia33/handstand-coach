from math import log

import pytest

from handstand_coach.temporal import ExponentialSmoother


def test_exponential_smoother_uses_elapsed_time() -> None:
    smoother = ExponentialSmoother(time_constant_s=1.0)

    first_result = smoother.update(
        value=100.0,
        timestamp_s=0.0,
    )
    second_result = smoother.update(
        value=180.0,
        timestamp_s=log(2.0),
    )

    assert first_result == pytest.approx(100.0)
    assert second_result == pytest.approx(140.0)


def test_exponential_smoother_reset_starts_a_new_sequence() -> None:
    smoother = ExponentialSmoother(time_constant_s=1.0)

    smoother.update(
        value=100.0,
        timestamp_s=0.0,
    )
    smoother.update(
        value=180.0,
        timestamp_s=1.0,
    )

    smoother.reset()

    result = smoother.update(
        value=90.0,
        timestamp_s=5.0,
    )

    assert result == pytest.approx(90.0)
