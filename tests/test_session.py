from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from handstand_coach.session import (
    SessionMetadata,
    create_session_metadata,
)

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


def test_create_session_metadata_normalizes_time_and_generates_id() -> None:
    helsinki_summer_time = timezone(timedelta(hours=3))
    started_at = datetime(
        2026,
        7,
        22,
        12,
        30,
        15,
        123456,
        tzinfo=helsinki_summer_time,
    )

    metadata = create_session_metadata(
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
        started_at=started_at,
    )

    assert metadata == SessionMetadata(
        session_id="2026-07-22T093015.123456Z",
        started_at_utc=datetime(
            2026,
            7,
            22,
            9,
            30,
            15,
            123456,
            tzinfo=UTC,
        ),
        source=0,
        model="yolov8n-pose.pt",
        confidence_threshold=0.5,
    )


def test_create_session_metadata_rejects_datetime_without_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        create_session_metadata(
            source=0,
            model="yolov8n-pose.pt",
            confidence_threshold=0.5,
            started_at=datetime(2026, 7, 22, 12, 30),
        )


def test_create_session_metadata_converts_paths_to_strings() -> None:
    metadata = create_session_metadata(
        source=Path("videos/handstand.mp4"),
        model=Path("models/yolov8n-pose.pt"),
        confidence_threshold=0.5,
        started_at=VALID_START_TIME,
    )

    assert metadata.source == str(Path("videos/handstand.mp4"))
    assert metadata.model == str(Path("models/yolov8n-pose.pt"))
