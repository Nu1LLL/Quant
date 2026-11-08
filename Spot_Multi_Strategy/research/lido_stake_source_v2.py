"""Timestamp-corrected, pinned Lido Sunday source; original sealed loader untouched."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math

from research.eth_defi_tvl_source import SourceFailure
from research.lido_stake_source import FIRST, LAST, STAKED_SHA, URL, load_spot
from research.stablecoin_liquidity_source import _fetch


TIMESTAMP_SHA = "787adfdb0e39b5d3e3de3047bf33503e7f82376a4395eec5dcc821e0b3664947"
EXPECTED_CLOCK_COUNTS = {"00:00:00": 239, "22:00:00": 7, "23:00:00": 17}


def parse_stake(body: bytes) -> tuple[dict[date, float], dict]:
    try:
        payload = json.loads(body)
        if payload.get("name") != "Lido":
            raise SourceFailure("Lido protocol identity changed")
        rows = payload["chainTvls"]["Ethereum"]["tokens"]
        if not isinstance(rows, list):
            raise SourceFailure("Lido Ethereum token history absent")
        selected = []
        for row in rows:
            if not isinstance(row, dict) or type(row.get("date")) is not int:
                raise SourceFailure("Lido token row schema invalid")
            moment = datetime.fromtimestamp(row["date"], timezone.utc)
            if FIRST <= moment.date() <= LAST and moment.weekday() == 6:
                tokens = row.get("tokens")
                if not isinstance(tokens, dict) or "WETH" not in tokens:
                    raise SourceFailure("Lido Sunday WETH entry absent")
                selected.append([row["date"], tokens["WETH"]])
        content_hash = sha256(json.dumps(selected, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
        if content_hash != STAKED_SHA:
            raise SourceFailure(f"Lido Sunday content revised: {content_hash}")
        timestamps = [row[0] for row in selected]
        timestamp_hash = sha256(json.dumps(timestamps, separators=(",", ":")).encode()).hexdigest()
        if timestamp_hash != TIMESTAMP_SHA:
            raise SourceFailure(f"Lido Sunday timestamps revised: {timestamp_hash}")
        expected_count = (LAST - FIRST).days // 7 + 1
        if len(selected) != 263 or expected_count != 263:
            raise SourceFailure("Lido Sunday record count changed")
        clock_counts = {clock: 0 for clock in EXPECTED_CLOCK_COUNTS}
        out = {}
        previous_stamp = -1
        for i, (stamp, raw) in enumerate(selected):
            moment = datetime.fromtimestamp(stamp, timezone.utc)
            expected_day = FIRST + timedelta(days=7 * i)
            clock = moment.strftime("%H:%M:%S")
            if (moment.date() != expected_day or stamp <= previous_stamp
                    or clock not in EXPECTED_CLOCK_COUNTS):
                raise SourceFailure(f"Lido UTC Sunday date/order/clock invalid: {expected_day}")
            if type(raw) not in (int, float) or not math.isfinite(raw) or raw <= 0:
                raise SourceFailure(f"Lido WETH units invalid: {expected_day}")
            clock_counts[clock] += 1
            out[expected_day] = float(raw)
            previous_stamp = stamp
        if clock_counts != EXPECTED_CLOCK_COUNTS:
            raise SourceFailure("Lido Sunday clock distribution changed")
        return out, {"source": URL, "ethereum_weth_sunday_sha256": content_hash,
                     "sunday_timestamp_sha256": timestamp_hash,
                     "utc_clock_counts": clock_counts,
                     "sundays": len(out), "first": str(FIRST), "last": str(LAST),
                     "raw_response_bytes": len(body),
                     "first_published_vintage_unavailable": True}
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, OSError) as exc:
        raise SourceFailure("malformed timestamp-corrected Lido history") from exc


def load_stake() -> tuple[dict[date, float], dict]:
    return parse_stake(_fetch(URL, limit=4_000_000))
