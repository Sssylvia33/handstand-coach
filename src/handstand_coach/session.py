"""Session-level domain models."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """Configuration and identity shared by every frame in a session."""

    session_id: str
    started_at_utc: datetime
    source: int | str
    model: str
    confidence_threshold: float

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")

        if self.started_at_utc.tzinfo is None or self.started_at_utc.utcoffset() is None:
            raise ValueError("started_at_utc must be timezone-aware")

        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("source must not be empty")

        if not self.model.strip():
            raise ValueError("model must not be empty")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")


def create_session_metadata(
    *,
    source: int | str | Path,
    model: str | Path,
    confidence_threshold: float,
    started_at: datetime,
) -> SessionMetadata:
    """Create normalized metadata for a new recording session."""

    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")

    started_at_utc = started_at.astimezone(UTC)
    session_id = started_at_utc.strftime("%Y-%m-%dT%H%M%S.%fZ")

    normalized_source = str(source) if isinstance(source, Path) else source

    return SessionMetadata(
        session_id=session_id,
        started_at_utc=started_at_utc,
        source=normalized_source,
        model=str(model),
        confidence_threshold=confidence_threshold,
    )
