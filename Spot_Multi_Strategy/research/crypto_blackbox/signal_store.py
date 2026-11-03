"""Small append-only SQLite store for crypto observations and audit labels.

This database holds no raw market packets, screenshots, health information,
orders or returns. Historical queries reveal a classification only once both
its underlying observation and the classification itself were available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3

from .schemas import (
    CRYPTO_MARKETS,
    CryptoObservation,
    ObservationClassification,
    require_utc,
)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_OUTCOME_REASON_FRAGMENTS = (
    "future", "forward", "post_alert", "after_alert", "profit", "pnl",
    "return", "gain", "win_rate", "screenshot", "image", "health",
)


def _to_us(value: datetime, name: str) -> int:
    require_utc(value, name)
    delta = value - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _from_us(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(value))


@dataclass(frozen=True)
class StoredSignal:
    """A visible observation and its already-available non-trading label."""

    observation: CryptoObservation
    classification: ObservationClassification


class SignalStore:
    """Append-only storage; the optional file path is supplied by the caller."""

    def __init__(self, path: str | Path | None = ":memory:") -> None:
        if path is None:
            path = ":memory:"
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("path must be a nonempty filesystem path or :memory:")
        if str(path).startswith("file:"):
            raise ValueError("SQLite URI paths are not accepted")
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL,
                source TEXT NOT NULL,
                source_version TEXT NOT NULL,
                license_id TEXT NOT NULL,
                market TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                family TEXT NOT NULL,
                window_start_us INTEGER NOT NULL,
                event_us INTEGER NOT NULL,
                available_us INTEGER NOT NULL,
                expires_us INTEGER NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                observed_json TEXT NOT NULL,
                UNIQUE(source,source_version,market,venue,symbol,family,event_us),
                CHECK(window_start_us < event_us),
                CHECK(event_us <= available_us),
                CHECK(available_us < expires_us),
                CHECK(horizon_seconds > 0)
            );
            CREATE INDEX IF NOT EXISTS observations_asof
                ON observations(available_us,market,event_us);
            CREATE INDEX IF NOT EXISTS observations_identity
                ON observations(event_id,market,venue,symbol,family);
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY,
                observation_id INTEGER NOT NULL UNIQUE,
                asof_us INTEGER NOT NULL,
                label TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                FOREIGN KEY(observation_id) REFERENCES observations(id),
                CHECK(label IN ('FIRST_ALERT','CONTINUATION','NO_ALERT','STALE'))
            );
            CREATE INDEX IF NOT EXISTS classifications_asof
                ON classifications(asof_us);
            CREATE TRIGGER IF NOT EXISTS immutable_observations_update
            BEFORE UPDATE ON observations BEGIN
                SELECT RAISE(ABORT, 'observations are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS immutable_observations_delete
            BEFORE DELETE ON observations BEGIN
                SELECT RAISE(ABORT, 'observations are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS immutable_classifications_update
            BEFORE UPDATE ON classifications BEGIN
                SELECT RAISE(ABORT, 'classifications are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS immutable_classifications_delete
            BEFORE DELETE ON classifications BEGIN
                SELECT RAISE(ABORT, 'classifications are append-only');
            END;
            """
        )

    def __enter__(self) -> SignalStore:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def append_observation(self, observation: CryptoObservation) -> None:
        if not isinstance(observation, CryptoObservation):
            raise TypeError("observation must be a CryptoObservation")
        observed_json = json.dumps(
            dict(observation.observed_fields), sort_keys=True, allow_nan=False,
            separators=(",", ":"),
        )
        values = (
            observation.event_id, observation.source, observation.source_version,
            observation.license_id, observation.market, observation.venue,
            observation.symbol, observation.family,
            _to_us(observation.window_start, "window_start"),
            _to_us(observation.event_time, "event_time"),
            _to_us(observation.available_time, "available_time"),
            _to_us(observation.expires_at, "expires_at"),
            observation.horizon_seconds, observed_json,
        )
        try:
            with self.db:
                self.db.execute(
                    """INSERT INTO observations (
                        event_id,source,source_version,license_id,market,venue,
                        symbol,family,window_start_us,event_us,available_us,
                        expires_us,horizon_seconds,observed_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate or invalid observation") from exc

    def append_classification(
        self, classification: ObservationClassification,
    ) -> None:
        if not isinstance(classification, ObservationClassification):
            raise TypeError("classification must be an ObservationClassification")
        if classification.tradable is not False:
            raise ValueError("trade decisions cannot be stored as observation labels")
        if len(classification.reasons) > 16 or not all(
            _REASON_CODE.fullmatch(reason) for reason in classification.reasons
        ):
            raise ValueError("classification reasons must be bounded machine codes")
        if any(
            fragment in reason
            for reason in classification.reasons
            for fragment in _OUTCOME_REASON_FRAGMENTS
        ):
            raise ValueError("outcome-derived or unrelated classification reason forbidden")
        rows = self.db.execute(
            """SELECT id,available_us,expires_us FROM observations
            WHERE event_id=? AND market=? AND venue=? AND symbol=? AND family=?""",
            (
                classification.event_id, classification.market,
                classification.venue, classification.symbol, classification.family,
            ),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("classification needs one unambiguous stored observation")
        observation_id, available_us, expires_us = rows[0]
        asof_us = _to_us(classification.asof_time, "classification.asof_time")
        if asof_us < available_us:
            raise ValueError("classification precedes observation availability")
        if (asof_us >= expires_us) != (classification.label == "STALE"):
            raise ValueError("stale label must agree with observation expiry")
        reasons_json = json.dumps(
            list(classification.reasons), allow_nan=False, separators=(",", ":"),
        )
        try:
            with self.db:
                self.db.execute(
                    """INSERT INTO classifications
                    (observation_id,asof_us,label,reasons_json) VALUES (?,?,?,?)""",
                    (observation_id, asof_us, classification.label, reasons_json),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("classification already exists or is invalid") from exc

    def list_asof(
        self, asof_time: datetime, market: str | None = None,
        limit: int = 100,
    ) -> tuple[StoredSignal, ...]:
        """Only return observations and labels both visible at ``asof_time``."""
        asof_us = _to_us(asof_time, "asof_time")
        if market is not None and market not in CRYPTO_MARKETS:
            raise ValueError("market filter must be a crypto market")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer in [1, 1000]")
        conditions = "o.available_us<=? AND c.asof_us<=?"
        parameters: list[object] = [asof_us, asof_us]
        if market is not None:
            conditions += " AND o.market=?"
            parameters.append(market)
        parameters.append(limit)
        rows = self.db.execute(
            f"""SELECT o.event_id,o.source,o.source_version,o.license_id,
                o.market,o.venue,o.symbol,o.family,o.window_start_us,o.event_us,
                o.available_us,o.expires_us,o.horizon_seconds,o.observed_json,
                c.asof_us,c.label,c.reasons_json
            FROM observations o JOIN classifications c ON c.observation_id=o.id
            WHERE {conditions}
            ORDER BY o.event_us DESC,o.id DESC LIMIT ?""",
            parameters,
        ).fetchall()
        result: list[StoredSignal] = []
        for row in rows:
            observation = CryptoObservation(
                event_id=row[0], source=row[1], source_version=row[2],
                license_id=row[3], market=row[4], venue=row[5], symbol=row[6],
                family=row[7], window_start=_from_us(row[8]),
                event_time=_from_us(row[9]), available_time=_from_us(row[10]),
                expires_at=_from_us(row[11]), horizon_seconds=row[12],
                observed_fields=json.loads(row[13]),
            )
            classification = ObservationClassification(
                event_id=row[0], market=row[4], venue=row[5], symbol=row[6],
                family=row[7], asof_time=_from_us(row[14]), label=row[15],
                reasons=tuple(json.loads(row[16])),
            )
            result.append(StoredSignal(observation, classification))
        return tuple(result)
