"""Read WhaleWatcher external-intelligence v2 for prospective shadow research.

The compact WhaleWatcher snapshot does not retain verification timestamps.
Consequently, a verified claim first becomes available to this adapter when
Quant actually reads the snapshot.  It must never be backfilled to the source
publication time or used as a historical out-of-sample observation.  The
adapter copies only provenance and source assertions, never text, prices,
post-event studies, AI reviews, probabilities, or execution instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time


_MAX_SNAPSHOT_BYTES = 2_000_000
_MAX_AGE_MS = 10 * 60 * 1000
_SIGNAL_ROLE = "external_intelligence_shadow"
_SOURCE_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}\Z")
_OBSERVATION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_VERIFICATION_STATUSES = {
    "pending", "verified", "partially_verified", "contradicted", "unverifiable",
}


@dataclass(frozen=True)
class ShadowClaim:
    """A source assertion, not an instruction to open a long/short position."""

    observation_id: str
    source_group_id: str
    fact_group_sha256: str
    asserted_side: str
    source_available_at_ms: int
    quant_available_at_ms: int
    execution_eligible: bool = field(default=False, init=False)


@dataclass(frozen=True)
class ShadowImport:
    symbol: str
    status: str
    captured_at_ms: int
    snapshot_generated_at_ms: int | None = None
    snapshot_sha256: str | None = None
    claims: tuple[ShadowClaim, ...] = ()
    signal_role: str = field(default="whalewatcher_shadow_research", init=False)
    research_only: bool = field(default=True, init=False)
    execution_eligible: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)


class _InvalidSnapshot(ValueError):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidSnapshot("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise _InvalidSnapshot(f"nonfinite JSON value: {value}")


def _valid_ms(value, upper_bound: int) -> bool:
    return type(value) is int and 0 < value <= upper_bound


def _safe_fact_group(value) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and value == value.strip().lower()
        and not any(character.isspace() or ord(character) < 32 for character in value)
    )


def _claims_for_symbol(row, *, generated_at_ms: int, captured_at_ms: int):
    if (
        not isinstance(row, dict)
        or row.get("manual_review_required") is not True
        or row.get("execution_eligible") is not False
    ):
        raise _InvalidSnapshot("unsafe symbol record")
    observations = row.get("observations")
    if not isinstance(observations, list) or len(observations) > 2_000:
        raise _InvalidSnapshot("invalid observation list")

    claims = []
    seen_ids = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise _InvalidSnapshot("invalid observation")
        observation_id = observation.get("observation_id")
        source = observation.get("source")
        fact_group = observation.get("fact_group_id")
        source_available = observation.get("available_at_ms")
        side = observation.get("side")
        verification = observation.get("verification_status")
        eligible = observation.get("consensus_eligible")
        if (
            not isinstance(observation_id, str)
            or not _OBSERVATION_ID_RE.fullmatch(observation_id)
            or observation_id in seen_ids
            or not isinstance(source, str)
            or not _SOURCE_RE.fullmatch(source)
            or not _safe_fact_group(fact_group)
            or not _valid_ms(source_available, generated_at_ms)
            or side not in {"long", "short", "neutral"}
            or verification not in _VERIFICATION_STATUSES
            or type(eligible) is not bool
            or observation.get("execution_eligible") is not False
        ):
            raise _InvalidSnapshot("malformed or unsafe observation")
        seen_ids.add(observation_id)
        if not eligible:
            continue
        if (
            verification != "verified"
            or side not in {"long", "short"}
            or source == "unknown"
            or fact_group == "shared_unknown"
            or fact_group.endswith(":shared_unknown")
        ):
            raise _InvalidSnapshot("unverified or dependent assertion")
        claims.append(ShadowClaim(
            observation_id=observation_id,
            source_group_id=source,
            fact_group_sha256=hashlib.sha256(fact_group.encode("utf-8")).hexdigest(),
            asserted_side=side,
            source_available_at_ms=source_available,
            # The source snapshot omits verified_at_ms; Quant's read time is
            # the earliest defensible timestamp for this imported assertion.
            quant_available_at_ms=captured_at_ms,
        ))
    return tuple(claims)


def read_whalewatcher_shadow_snapshot(
    path: str | Path,
    symbol: str,
    *,
    max_age_ms: int = _MAX_AGE_MS,
) -> ShadowImport:
    """Import a fresh local v2 snapshot; fail closed to an empty shadow context.

    This function uses the real clock at ingestion and does not accept a
    caller-supplied historical decision time.  It provides no execution path.
    A trustworthy historical study needs separately archived first-seen and
    verification timestamps, source rights, and independent validation.
    """

    normalized = str(symbol).strip().upper()
    if normalized.endswith("USDT"):
        normalized = normalized[:-4]
    if normalized not in {"BTC", "ETH"}:
        raise ValueError("WhaleWatcher shadow snapshot supports BTC and ETH only")
    if type(max_age_ms) is not int or not 0 < max_age_ms <= _MAX_AGE_MS:
        raise ValueError("max_age_ms must be within (0, 600000]")

    started_at_ms = time.time_ns() // 1_000_000
    unavailable = ShadowImport(normalized, "unavailable", started_at_ms)
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(_MAX_SNAPSHOT_BYTES + 1)
        if not raw or len(raw) > _MAX_SNAPSHOT_BYTES:
            return unavailable
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
            parse_constant=_reject_nonfinite,
        )
        # Reading and parsing may take time.  The imported claim becomes
        # visible to Quant only after that work has finished.
        captured_at_ms = time.time_ns() // 1_000_000
        unavailable = ShadowImport(normalized, "unavailable", captured_at_ms)
        if not isinstance(payload, dict):
            return unavailable
        if (
            type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 2
            or payload.get("signal_role") != _SIGNAL_ROLE
            or payload.get("execution_eligible") is not False
            or payload.get("operator_execution_eligible") is not False
            or payload.get("promotion_eligible") is not False
            or payload.get("manual_review_required") is not True
            or payload.get("performance_calibration_used") is not False
        ):
            return unavailable
        generated_at_ms = payload.get("generated_at_ms")
        if (
            not _valid_ms(generated_at_ms, captured_at_ms)
            or captured_at_ms - generated_at_ms > max_age_ms
        ):
            return unavailable
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict) or set(symbols) != {"BTC", "ETH"}:
            return unavailable
        # Validate both records so a corrupted or promoted peer symbol does
        # not pass merely because this call requested the other symbol.
        claims_by_symbol = {
            asset: _claims_for_symbol(
                symbols[asset], generated_at_ms=generated_at_ms,
                captured_at_ms=captured_at_ms,
            )
            for asset in ("BTC", "ETH")
        }
        claims = claims_by_symbol[normalized]
        return ShadowImport(
            symbol=normalized,
            status="fresh" if claims else "fresh_no_claims",
            captured_at_ms=captured_at_ms,
            snapshot_generated_at_ms=generated_at_ms,
            snapshot_sha256=hashlib.sha256(raw).hexdigest(),
            claims=claims,
        )
    except (OSError, TypeError, ValueError, UnicodeError, _InvalidSnapshot):
        return unavailable
