"""Session-level domain models."""

from dataclasses import dataclass
from datetime import datetime

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
