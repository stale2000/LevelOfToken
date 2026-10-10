"""Pure deadline and admission helpers for bounded LoT training campaigns.

All wall-clock deadlines are UTC. Request deadlines use a monotonic clock so
NTP adjustments cannot extend or prematurely expire an admitted request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

CAMPAIGN_SECONDS = 5 * 60 * 60
OPTIMIZER_SECONDS = 3 * 60 * 60
LAST_SUBMISSION_SECONDS = 4 * 60 * 60
MAX_SERVER_START_TIMEOUT_SECONDS = 900
MAX_STARTUP_TIMEOUT_SECONDS = 900
MAX_REQUEST_TIMEOUT_SECONDS = 2700
MAX_TERMINATION_GRACE_SECONDS = 180
MIN_SAFETY_MARGIN_SECONDS = 300
MAX_EXPORT_RESERVE_SECONDS = 900


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("wall-clock times must include a timezone")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class WallDeadline:
    """An immutable absolute UTC deadline."""

    at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _utc(self.at))

    def remaining_seconds(self, now_utc: datetime) -> float:
        return (self.at - _utc(now_utc)).total_seconds()

    def expired(self, now_utc: datetime) -> bool:
        return self.remaining_seconds(now_utc) <= 0


@dataclass(frozen=True)
class MonotonicDeadline:
    """An immutable deadline measured by a monotonic clock."""

    at: float

    def remaining_seconds(self, now_monotonic: float) -> float:
        return self.at - now_monotonic

    def expired(self, now_monotonic: float) -> bool:
        return self.remaining_seconds(now_monotonic) <= 0


def make_monotonic_deadline(timeout_seconds: float, now_monotonic: float) -> MonotonicDeadline:
    """Create a request deadline from an injected monotonic timestamp."""
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")
    return MonotonicDeadline(now_monotonic + timeout_seconds)


@dataclass(frozen=True)
class CampaignDeadlines:
    """Fixed campaign wall deadlines derived from its UTC start time."""

    started_at: datetime
    optimizer_cutoff: WallDeadline
    last_submission: WallDeadline
    final_deadline: WallDeadline

    @classmethod
    def from_start(cls, started_at: datetime) -> CampaignDeadlines:
        start = _utc(started_at)
        return cls(
            started_at=start,
            optimizer_cutoff=WallDeadline(start + timedelta(seconds=OPTIMIZER_SECONDS)),
            last_submission=WallDeadline(start + timedelta(seconds=LAST_SUBMISSION_SECONDS)),
            final_deadline=WallDeadline(start + timedelta(seconds=CAMPAIGN_SECONDS)),
        )


@dataclass(frozen=True)
class FalLimits:
    """Fal lifecycle limits used for admission, in seconds."""

    server_start_timeout: int = MAX_SERVER_START_TIMEOUT_SECONDS
    startup_timeout: int = MAX_STARTUP_TIMEOUT_SECONDS
    request_timeout: int = MAX_REQUEST_TIMEOUT_SECONDS
    termination_grace: int = MAX_TERMINATION_GRACE_SECONDS
    safety_margin: int = MIN_SAFETY_MARGIN_SECONDS

    def validate(self) -> str | None:
        checks = (
            (self.server_start_timeout <= MAX_SERVER_START_TIMEOUT_SECONDS,
             f"server start timeout must be at most {MAX_SERVER_START_TIMEOUT_SECONDS}s"),
            (self.startup_timeout <= MAX_STARTUP_TIMEOUT_SECONDS,
             f"startup timeout must be at most {MAX_STARTUP_TIMEOUT_SECONDS}s"),
            (self.request_timeout <= MAX_REQUEST_TIMEOUT_SECONDS,
             f"request timeout must be at most {MAX_REQUEST_TIMEOUT_SECONDS}s"),
            (self.termination_grace <= MAX_TERMINATION_GRACE_SECONDS,
             f"termination grace must be at most {MAX_TERMINATION_GRACE_SECONDS}s"),
            (self.safety_margin >= MIN_SAFETY_MARGIN_SECONDS,
             f"safety margin must be at least {MIN_SAFETY_MARGIN_SECONDS}s"),
        )
        for valid, reason in checks:
            if not valid:
                return reason
        if min(self.server_start_timeout, self.startup_timeout,
               self.request_timeout, self.termination_grace) < 0:
            return "Fal timeouts and termination grace must be non-negative"
        return None


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str
    processing_seconds: float
    latest_finish: datetime


def admit_request(
    *,
    now_utc: datetime,
    deadlines: CampaignDeadlines,
    optimizer_seconds: float,
    evaluation_seconds: float = 0,
    export_reserve_seconds: float = MAX_EXPORT_RESERVE_SECONDS,
    phase: str = "optimizer",
    limits: FalLimits | None = None,
) -> Admission:
    """Decide whether a request fits Fal and campaign deadlines.

    Evaluation and export are part of request processing: their durations are
    added to optimizer work and must fit the 45-minute Fal request cap. The
    latest finish also reserves startup, termination grace, and safety margin.
    ``now_utc`` is injected to keep scheduling decisions deterministic.
    """
    now = _utc(now_utc)
    limits = limits or FalLimits()
    durations = (optimizer_seconds, evaluation_seconds, export_reserve_seconds)
    if any(seconds < 0 for seconds in durations):
        return Admission(False, "request phase durations must be non-negative", 0, now)
    if export_reserve_seconds > MAX_EXPORT_RESERVE_SECONDS:
        return Admission(False, f"export reserve must be at most {MAX_EXPORT_RESERVE_SECONDS}s",
                         sum(durations), now)
    if phase not in {"optimizer", "evaluation"}:
        return Admission(False, "phase must be 'optimizer' or 'evaluation'", sum(durations), now)
    invalid_limits = limits.validate()
    if invalid_limits:
        return Admission(False, invalid_limits, sum(durations), now)

    processing = sum(durations)
    startup = limits.server_start_timeout + limits.startup_timeout
    latest_finish = now + timedelta(
        seconds=startup + processing + limits.termination_grace + limits.safety_margin
    )
    if processing > limits.request_timeout:
        return Admission(False,
                         f"processing budget {processing:g}s exceeds Fal request timeout "
                         f"{limits.request_timeout}s (includes evaluation and export)",
                         processing, latest_finish)

    submission_deadline = (deadlines.optimizer_cutoff if phase == "optimizer"
                           else deadlines.last_submission)
    if now > submission_deadline.at:
        name = "optimizer cutoff" if phase == "optimizer" else "last submission cutoff"
        return Admission(False, f"cannot submit after {name} at {submission_deadline.at.isoformat()}",
                         processing, latest_finish)

    if phase == "optimizer":
        # Queue/routing time is part of the campaign wall-clock envelope too.
        optimizer_finish = now + timedelta(
            seconds=limits.server_start_timeout + limits.startup_timeout + optimizer_seconds
        )
        if optimizer_finish > deadlines.optimizer_cutoff.at:
            return Admission(False,
                             "optimizer work plus startup would finish after the T+4h optimizer cutoff",
                             processing, latest_finish)
    if latest_finish > deadlines.final_deadline.at:
        return Admission(False,
                         "request plus startup, termination grace, and safety margin would exceed "
                         "the T+6h final deadline",
                         processing, latest_finish)
    return Admission(True, "fits campaign and Fal request deadlines", processing, latest_finish)
