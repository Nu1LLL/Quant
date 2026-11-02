"""Provider-neutral point-in-time quote and OHLCV storage."""

from datetime import datetime, timezone
import hashlib
import json
import math

import pandas as pd


def utc_us(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be ISO-8601 text with timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone missing")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


class MarketDataStore:
    """Persist L1 quotes and bars with explicit event and availability time."""

    def __init__(self, registry):
        self.registry = registry
        self.db = registry.db
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_events (
              provider TEXT NOT NULL, dataset_version TEXT NOT NULL,
              venue TEXT NOT NULL, instrument TEXT NOT NULL,
              event_id TEXT NOT NULL, kind TEXT NOT NULL,
              fingerprint TEXT NOT NULL, market TEXT NOT NULL,
              event_us INTEGER NOT NULL, available_us INTEGER NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY(provider,dataset_version,venue,instrument,event_id,kind));
            CREATE INDEX IF NOT EXISTS market_event_lookup ON market_events
              (provider,dataset_version,venue,instrument,kind,available_us,event_us);
            CREATE TABLE IF NOT EXISTS quarantine (
              id INTEGER PRIMARY KEY, raw TEXT NOT NULL,
              reason TEXT NOT NULL, quarantined_us INTEGER NOT NULL);
            """
        )

    def _quarantine(self, raw_text, reason):
        now_us = int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000)
        with self.db:
            self.db.execute(
                "INSERT INTO quarantine(raw,reason,quarantined_us) VALUES (?,?,?)",
                (raw_text, str(reason), now_us),
            )
        return "quarantined"

    def _ingest(self, raw, kind):
        raw_text = json.dumps(raw, sort_keys=True, default=str)
        try:
            row = dict(raw)
            identity_fields = (
                "provider", "dataset_version", "venue", "instrument", "event_id"
            )
            for key in (*identity_fields, "market"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"missing identity: {key}")
            event = utc_us(row["event_time"])
            available = utc_us(row["available_time"])
            if available < event:
                raise ValueError("availability precedes event")

            if kind == "quote":
                numeric = ("bid", "ask", "bid_size", "ask_size")
                payload = {key: row[key] for key in numeric}
                if row["bid"] <= 0 or row["ask"] <= row["bid"]:
                    raise ValueError("nonpositive, locked or crossed quote")
                if min(row["bid_size"], row["ask_size"]) < 0:
                    raise ValueError("negative size")
                payload["capabilities"] = ["l1_quote"]
            elif kind == "bar":
                numeric = ("open", "high", "low", "close", "volume")
                payload = {key: row[key] for key in numeric}
                if min(row["open"], row["high"], row["low"], row["close"]) <= 0:
                    raise ValueError("nonpositive OHLC")
                if not row["low"] <= min(row["open"], row["close"]):
                    raise ValueError("invalid OHLC low")
                if not row["high"] >= max(row["open"], row["close"]):
                    raise ValueError("invalid OHLC high")
                if row["volume"] < 0:
                    raise ValueError("negative volume")
                if not isinstance(row.get("interval"), str) or not row["interval"].strip():
                    raise ValueError("bar interval is required")
                payload["interval"] = row["interval"]
                payload["capabilities"] = ["ohlcv_bar"]
            else:
                raise ValueError("unknown market event kind")

            for key in numeric:
                if type(row[key]) not in (float, int) or not math.isfinite(row[key]):
                    raise ValueError(f"invalid {key}")
            payload["latency_us"] = available - event
            canonical = json.dumps(
                [kind, row["market"], event, available, payload], sort_keys=True
            )
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            identity = tuple(row[key] for key in identity_fields)
            prior = self.db.execute(
                """SELECT fingerprint FROM market_events WHERE provider=?
                AND dataset_version=? AND venue=? AND instrument=?
                AND event_id=? AND kind=?""",
                (*identity, kind),
            ).fetchone()
            if prior:
                if prior[0] != fingerprint:
                    raise ValueError("conflicting duplicate event")
                return "duplicate"
            with self.db:
                self.db.execute(
                    "INSERT INTO market_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        *identity, kind, fingerprint, row["market"], event,
                        available, json.dumps(payload, sort_keys=True),
                    ),
                )
            return "accepted"
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            return self._quarantine(raw_text, exc)

    def ingest_quote(self, raw):
        return self._ingest(raw, "quote")

    def ingest_bar(self, raw):
        return self._ingest(raw, "bar")

    def latest_quote(
        self, provider, dataset_version, venue, instrument,
        decision_us, max_age_us, required_capabilities=("l1_quote",)
    ):
        row = self.db.execute(
            """SELECT event_us,available_us,payload FROM market_events
            WHERE provider=? AND dataset_version=? AND venue=? AND instrument=?
            AND kind='quote' AND available_us<=?
            ORDER BY event_us DESC,available_us DESC,event_id DESC LIMIT 1""",
            (provider, dataset_version, venue, instrument, decision_us),
        ).fetchone()
        if row is None or decision_us - row[0] > max_age_us:
            return None
        payload = json.loads(row[2])
        if not set(required_capabilities).issubset(payload["capabilities"]):
            return None
        return {"event_us": row[0], "available_us": row[1], **payload}

    def bars_asof(
        self, provider, dataset_version, venue, instrument, decision_us
    ) -> pd.DataFrame:
        rows = self.db.execute(
            """SELECT event_us,available_us,payload FROM (
              SELECT event_us,available_us,payload,event_id,
              ROW_NUMBER() OVER (
                PARTITION BY event_us
                ORDER BY available_us DESC,event_id DESC
              ) AS revision_rank
              FROM market_events
              WHERE provider=? AND dataset_version=? AND venue=? AND instrument=?
              AND kind='bar' AND available_us<=?
            ) WHERE revision_rank=1 ORDER BY event_us""",
            (provider, dataset_version, venue, instrument, decision_us),
        ).fetchall()
        records = []
        for event_us, available_us, payload_text in rows:
            payload = json.loads(payload_text)
            records.append(
                {
                    "open_time": pd.Timestamp(event_us, unit="us", tz="UTC"),
                    "available_time": pd.Timestamp(available_us, unit="us", tz="UTC"),
                    **{key: payload[key] for key in ("open", "high", "low", "close", "volume")},
                }
            )
        return pd.DataFrame(records)
