"""Causal, non-trading labels for five-minute crypto flow anomalies.

The thresholds here are fixed taxonomy rules, not fitted alpha parameters.
Their output must never be interpreted as a return forecast or trade order.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math

from .schemas import CryptoObservation, ObservationClassification, require_utc


MIN_BUY_SHARE = 0.55
MIN_VOLUME_MULTIPLE = 2.0
CONTINUATION_WINDOW = timedelta(hours=1)
REQUIRED_FLOW_FIELDS = (
    "aggressor_buy_share_5m",
    "quote_volume_5m_usd",
    "quote_volume_baseline_5m_usd",
)


def classify_observation(
    observation: CryptoObservation,
    asof_time: datetime,
    recent_alert_time: datetime | None = None,
    recent_alert_available_time: datetime | None = None,
) -> ObservationClassification:
    """Classify one already completed observation using no future outcomes."""
    if not isinstance(observation, CryptoObservation):
        raise TypeError("observation must be a CryptoObservation")
    require_utc(asof_time, "asof_time")
    if observation.available_time > asof_time:
        raise ValueError("observation is not yet visible at asof_time")
    if (recent_alert_time is None) != (recent_alert_available_time is None):
        raise ValueError("prior alert event and availability times must be supplied together")
    if recent_alert_time is not None:
        require_utc(recent_alert_time, "recent_alert_time")
        require_utc(recent_alert_available_time, "recent_alert_available_time")
        if recent_alert_time > observation.event_time:
            raise ValueError("recent_alert_time cannot be after the observation")
        if recent_alert_available_time < recent_alert_time:
            raise ValueError("prior alert availability cannot precede its event")
        if recent_alert_available_time > asof_time:
            raise ValueError("prior alert was not visible at asof_time")

    def result(label: str, *reasons: str) -> ObservationClassification:
        return ObservationClassification(
            event_id=observation.event_id,
            market=observation.market,
            venue=observation.venue,
            symbol=observation.symbol,
            family=observation.family,
            label=label,
            reasons=tuple(reasons),
            asof_time=asof_time,
        )

    if asof_time >= observation.expires_at:
        return result("STALE", "observation_expired")
    if observation.family != "flow_volume_position":
        return result("NO_ALERT", "family_not_supported_by_flow_classifier")

    fields = observation.observed_fields
    missing = [name for name in REQUIRED_FLOW_FIELDS if name not in fields]
    if missing:
        return result("NO_ALERT", *(f"missing_{name}" for name in missing))

    reasons: list[str] = []
    buy_share = fields["aggressor_buy_share_5m"]
    volume_multiple = (
        fields["quote_volume_5m_usd"] / fields["quote_volume_baseline_5m_usd"]
    )
    if not math.isfinite(volume_multiple):
        return result("NO_ALERT", "volume_multiple_nonfinite")
    if buy_share < MIN_BUY_SHARE:
        reasons.append("buy_share_below_0_55")
    else:
        reasons.append("buy_share_at_least_0_55")
    if volume_multiple < MIN_VOLUME_MULTIPLE:
        reasons.append("volume_multiple_below_2")
    else:
        reasons.append("volume_multiple_at_least_2")
    open_interest = fields.get("open_interest_change_pct_5m")
    if open_interest is None:
        reasons.append("open_interest_unavailable")
    elif open_interest > 0:
        reasons.append("open_interest_increasing")
    elif open_interest < 0:
        reasons.append("open_interest_decreasing")
    else:
        reasons.append("open_interest_unchanged")
    if buy_share < MIN_BUY_SHARE or volume_multiple < MIN_VOLUME_MULTIPLE:
        return result("NO_ALERT", *reasons)

    if recent_alert_time is not None and (
        observation.event_time - recent_alert_time < CONTINUATION_WINDOW
    ):
        return result("CONTINUATION", *reasons, "prior_alert_within_one_hour")
    return result("FIRST_ALERT", *reasons, "no_prior_alert_within_one_hour")


class SignalIntake:
    """Small deterministic alert-history state machine, not a trade engine."""

    def __init__(self) -> None:
        self._seen: set[tuple] = set()
        self._last_event: dict[tuple[str, ...], datetime] = {}
        self._last_asof: dict[tuple[str, ...], datetime] = {}
        self._last_alert: dict[
            tuple[str, ...], tuple[datetime, datetime]
        ] = {}

    def classify(
        self, observation: CryptoObservation, asof_time: datetime,
    ) -> ObservationClassification:
        if not isinstance(observation, CryptoObservation):
            raise TypeError("observation must be a CryptoObservation")
        duplicate_key = observation.duplicate_key
        if duplicate_key in self._seen:
            raise ValueError("duplicate source observation for the same window")
        stream_key = observation.stream_key
        previous_event = self._last_event.get(stream_key)
        if previous_event is not None and observation.event_time < previous_event:
            raise ValueError("observations must arrive in event-time order per stream")
        previous_asof = self._last_asof.get(stream_key)
        if previous_asof is not None and asof_time < previous_asof:
            raise ValueError("decision asof_time cannot go backwards per stream")
        prior_alert = self._last_alert.get(stream_key)
        label = classify_observation(
            observation, asof_time,
            prior_alert[0] if prior_alert is not None else None,
            prior_alert[1] if prior_alert is not None else None,
        )
        self._seen.add(duplicate_key)
        self._last_event[stream_key] = observation.event_time
        self._last_asof[stream_key] = asof_time
        if label.label in {"FIRST_ALERT", "CONTINUATION"}:
            self._last_alert[stream_key] = (
                observation.event_time, observation.available_time,
            )
        return label
