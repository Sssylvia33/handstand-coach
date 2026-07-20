from datetime import UTC, datetime, timedelta, timezone

import pytest

from handstand_coach.session import SessionMetadata

VALID_START_TIME = datetime(2026, 7, 20, 18, 30, tzinfo=UTC)


def make_metadata(
    *,
    session_id: str = "2026-07-20T183000Z",
    started_at_utc: datetime = VALID_START_TIME,
    source: int | str = 0,
    model: str = "yolov8n-pose.pt",
    confidence_threshold: float = 0.5,
) -> SessionMetadata:
    """Create valid metadata with selected fields overridden."""

    return SessionMetadata(
        session_id=session_id,
        started_at_utc=started_at_utc,
        source=source,
        model=model,
        confidence_threshold=confidence_threshold,
    )


@pytest.mark.parametrize("session_id", ["", "   "])
def test_session_metadata_rejects_empty_session_id(
    session_id: str,
) -> None:
    with pytest.raises(ValueError, match="session_id"):
        make_metadata(session_id=session_id)


def test_session_metadata_rejects_datetime_without_timezone() -> None:
    naive_datetime = datetime(2026, 7, 20, 18, 30)

    with pytest.raises(ValueError, match="timezone-aware"):
        make_metadata(started_at_utc=naive_datetime)


@pytest.mark.parametrize("source", ["", "   "])
def test_session_metadata_rejects_empty_string_source(
    source: str,
) -> None:
    with pytest.raises(ValueError, match="source"):
        make_metadata(source=source)


@pytest.mark.parametrize("model", ["", "   "])
def test_session_metadata_rejects_empty_model(
    model: str,
) -> None:
    with pytest.raises(ValueError, match="model"):
        make_metadata(model=model)


@pytest.mark.parametrize(
    "confidence_threshold",
    [-0.01, 1.01, float("nan")],
)
def test_session_metadata_rejects_invalid_confidence_threshold(
    confidence_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        make_metadata(confidence_threshold=confidence_threshold)


def test_session_metadata_accepts_timezone_aware_non_utc_start() -> None:
    helsinki_summer_time = timezone(timedelta(hours=3))

    metadata = make_metadata(
        started_at_utc=datetime(
            2026,
            7,
            20,
            21,
            30,
            tzinfo=helsinki_summer_time,
        )
    )

    assert metadata.started_at_utc.utcoffset() == timedelta(hours=3)


@pytest.mark.parametrize(
    "confidence_threshold",
    [0.0, 0.5, 1.0],
)
def test_session_metadata_accepts_valid_confidence_threshold(
    confidence_threshold: float,
) -> None:
    metadata = make_metadata(confidence_threshold=confidence_threshold)

    assert metadata.confidence_threshold == confidence_threshold
