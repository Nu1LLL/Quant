"""Frozen Snapshot proposal and Binance spot sources; no strategy decisions."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import re
import time
from typing import Any

import pandas as pd
import requests

from research.stablecoin_liquidity_source import _fetch, _load_month


SPACES = {
    "UNIUSDT": "uniswapgovernance.eth",
    "COMPUSDT": "comp-vote.eth",
    "SUSHIUSDT": "sushigov.eth",
    "1INCHUSDT": "1inch.eth",
    "AAVEUSDT": "aavedao.eth",
}
SYMBOLS = tuple(SPACES)
PRICE_SYMBOLS = SYMBOLS + ("BTCUSDT",)
START = date(2021, 1, 1)
END = date(2026, 1, 1)
WARMUP = date(2020, 1, 1)
GRAPHQL = "https://hub.snapshot.org/graphql"
QUERY = '''query($space:String!,$skip:Int!){proposals(first:1000,skip:$skip,
  where:{space:$space},orderBy:"created",orderDirection:asc){
    id created start end votes scores_updated space{id}}}'''
MAX_PAGES = 20
PAGE_SIZE = 1000


class SourceFailure(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    symbol: str
    space: str
    created_s: int
    start_s: int
    end_s: int
    votes: int
    scores_updated_s: int | None

    @property
    def decision_s(self) -> int:
        return self.end_s + 86_400

    @property
    def score_available(self) -> bool:
        return self.scores_updated_s is not None and self.scores_updated_s <= self.decision_s


def _parse_proposal(raw: Any, symbol: str, space: str) -> Proposal:
    try:
        if not isinstance(raw, dict) or raw.get("space", {}).get("id") != space:
            raise SourceFailure("proposal space changed")
        ident = raw["id"]
        if not isinstance(ident, str) or not ident or len(ident) > 128:
            raise SourceFailure("proposal ID invalid")
        created, start, end, votes = (raw[name] for name in
                                      ("created", "start", "end", "votes"))
        updated = raw["scores_updated"]
        if (any(type(x) is not int for x in (created, start, end, votes))
                or updated is not None and type(updated) is not int
                or not 0 <= created < 4_102_444_800
                or not 0 <= start < end < 4_102_444_800 or votes < 0
                or updated is not None and not 0 <= updated < 4_102_444_800):
            raise SourceFailure("proposal timestamp or vote count invalid")
        return Proposal(ident, symbol, space, created, start, end, votes, updated)
    except (KeyError, TypeError, AttributeError) as exc:
        raise SourceFailure("proposal schema invalid") from exc


def _fetch_space(symbol: str, space: str) -> tuple[list[Proposal], str]:
    proposals: list[Proposal] = []
    previous_created = -1
    seen: set[str] = set()
    for page in range(MAX_PAGES):
        payload = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = requests.post(
                    GRAPHQL, json={"query": QUERY,
                                   "variables": {"space": space, "skip": page * PAGE_SIZE}},
                    timeout=30)
                response.raise_for_status()
                if len(response.content) > 5_000_000:
                    raise SourceFailure("oversized Snapshot response")
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(attempt + 1)
        if payload is None:
            raise SourceFailure(f"Snapshot transport failed: {space}") from last_error
        if (not isinstance(payload, dict) or payload.get("errors")
                or not isinstance(payload.get("data", {}).get("proposals"), list)):
            raise SourceFailure(f"Snapshot GraphQL failed: {space}")
        batch = payload["data"]["proposals"]
        if len(batch) > PAGE_SIZE:
            raise SourceFailure("Snapshot page too large")
        for raw in batch:
            proposal = _parse_proposal(raw, symbol, space)
            if proposal.created_s < previous_created or proposal.proposal_id in seen:
                raise SourceFailure("Snapshot pagination order/duplicate failure")
            previous_created = proposal.created_s
            seen.add(proposal.proposal_id)
            if (datetime.fromtimestamp(proposal.end_s, timezone.utc).date() >= WARMUP
                    and proposal.created_s < int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())):
                proposals.append(proposal)
        if len(batch) < PAGE_SIZE:
            break
    else:
        raise SourceFailure("Snapshot pagination did not exhaust")
    canonical = json.dumps(
        [[p.proposal_id, p.symbol, p.space, p.created_s, p.start_s,
          p.end_s, p.votes, p.scores_updated_s] for p in proposals],
        separators=(",", ":"), ensure_ascii=False).encode()
    return proposals, sha256(canonical).hexdigest()


def load_proposals() -> tuple[list[Proposal], dict]:
    """Read the fixed historical subset twice; reject a moving source vintage."""
    first = [_fetch_space(symbol, space) for symbol, space in SPACES.items()]
    second = [_fetch_space(symbol, space) for symbol, space in SPACES.items()]
    if [digest for _, digest in first] != [digest for _, digest in second]:
        raise SourceFailure("Snapshot historical subset changed between reads")
    rows = [proposal for proposals, _ in first for proposal in proposals]
    audit = {symbol: {"space": space, "rows": len(first[i][0]),
                      "sha256": first[i][1]}
             for i, (symbol, space) in enumerate(SPACES.items())}
    return rows, audit


def _months(symbol: str) -> list[str]:
    begin = (2020, 10) if symbol == "BTCUSDT" else (2021, 1)
    return [f"{year}-{month:02d}" for year in range(2020, 2027)
            for month in range(1, 13)
            if begin <= (year, month) <= (2026, 1)]


def _price_month(spec: tuple[str, str]):
    symbol, month = spec
    name = f"{symbol}-1d-{month}.zip"
    base = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{name}"
    try:
        statement = _fetch(base + ".CHECKSUM", limit=200).decode("ascii").strip()
    except (UnicodeError, ValueError) as exc:
        raise SourceFailure(f"checksum unavailable: {symbol} {month}") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+([^\s]+)", statement)
    if match is None or match.group(2) != name:
        raise SourceFailure(f"invalid Binance checksum: {symbol} {month}")
    year, mon = map(int, month.split("-"))
    row = {"month": month, "symbol": symbol, "sha256": match.group(1).lower(),
           "days": str(monthrange(year, mon)[1]),
           "timestamp_unit": "us" if month >= "2025-01" else "ms",
           "market": "spot", "dataset": "1d"}
    return _load_month(row)


def load_prices() -> tuple[dict[str, pd.DataFrame], dict[str, float], list[dict]]:
    specs = [(symbol, month) for symbol in PRICE_SYMBOLS for month in _months(symbol)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(_price_month, specs))
    frames: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in PRICE_SYMBOLS}
    terminal: dict[str, float] = {}
    audit = []
    for month, symbol, data, meta in parts:
        audit.append(meta)
        if month == "2026-01":
            if not isinstance(data, float):
                raise SourceFailure("terminal price missing")
            terminal[symbol] = data
        else:
            if not isinstance(data, pd.DataFrame):
                raise SourceFailure("daily price frame missing")
            frames[symbol].append(data)
    merged = {symbol: pd.concat(frames[symbol]) for symbol in PRICE_SYMBOLS}
    for symbol, frame in merged.items():
        first = "2020-10-01" if symbol == "BTCUSDT" else "2021-01-01"
        if not frame.index.equals(pd.date_range(first, "2025-12-31", tz="UTC", freq="D")):
            raise SourceFailure(f"{symbol}: incomplete daily UTC grid")
    if set(terminal) != set(PRICE_SYMBOLS):
        raise SourceFailure("terminal opening prices incomplete")
    return merged, terminal, audit
