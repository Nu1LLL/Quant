"""One-shot, bounded GitHub issue-count reader for the frozen BTC/ETH study.

No title, label, closure state, price, or strategy outcome enters this source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
import subprocess
import time


FIRST_MONDAY = date(2020, 10, 5)
LAST_MONDAY = date(2026, 9, 14)
REPOS = (("BTCUSDT", "bitcoin/bitcoin"),
         ("ETHUSDT", "ethereum/go-ethereum"))
WEEKS_PER_BATCH = 12


class IssueSourceError(ValueError):
    pass


@dataclass(frozen=True)
class IssueSource:
    counts: dict[date, dict[str, int]]
    sha256: str
    queries: int
    batches: int


def mondays() -> tuple[date, ...]:
    days = (LAST_MONDAY - FIRST_MONDAY).days
    if days % 7 or FIRST_MONDAY.weekday() != 0 or LAST_MONDAY.weekday() != 0:
        raise IssueSourceError("registered week bounds changed")
    return tuple(FIRST_MONDAY + timedelta(days=7 * i)
                 for i in range(days // 7 + 1))


def _query(batch: tuple[date, ...]) -> tuple[str, dict[str, tuple[date, str]]]:
    fields = []
    aliases = {}
    for index, monday in enumerate(batch):
        sunday = monday + timedelta(days=6)
        for asset, repo in REPOS:
            alias = f"w{index}{'b' if asset == 'BTCUSDT' else 'e'}"
            search = (f"repo:{repo} is:issue "
                      f"created:{monday.isoformat()}..{sunday.isoformat()}")
            fields.append(f'{alias}:search(query:{json.dumps(search)},'
                          'type:ISSUE,first:0){issueCount}')
            aliases[alias] = (monday, asset)
    fields.append("rateLimit{remaining cost}")
    return "query{" + " ".join(fields) + "}", aliases


def load_counts() -> IssueSource:
    weeks = mondays()
    counts: dict[date, dict[str, int]] = {}
    batches = 0
    for offset in range(0, len(weeks), WEEKS_PER_BATCH):
        query, aliases = _query(weeks[offset:offset + WEEKS_PER_BATCH])
        completed = None
        for attempt in range(3):
            try:
                completed = subprocess.run(
                    ["gh", "api", "graphql", "-f", "query=" + query],
                    capture_output=True, text=True, timeout=45, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                completed = None
            if completed is not None and completed.returncode == 0:
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
        if completed is None or completed.returncode:
            raise IssueSourceError(f"GitHub GraphQL transport failed at batch {batches}")
        try:
            payload = json.loads(completed.stdout)
        except (ValueError, UnicodeError) as exc:
            raise IssueSourceError(f"invalid GraphQL JSON at batch {batches}") from exc
        if payload.get("errors") or not isinstance(payload.get("data"), dict):
            raise IssueSourceError(f"GraphQL errors at batch {batches}")
        data = payload["data"]
        if set(data) != set(aliases) | {"rateLimit"}:
            raise IssueSourceError(f"GraphQL aliases missing at batch {batches}")
        rate_limit = data["rateLimit"]
        if not isinstance(rate_limit, dict) or not isinstance(rate_limit.get("remaining"), int):
            raise IssueSourceError(f"GraphQL rate metadata missing at batch {batches}")
        for alias, (monday, asset) in aliases.items():
            row = data[alias]
            value = row.get("issueCount") if isinstance(row, dict) else None
            if type(value) is not int or not 0 <= value < 1000:
                raise IssueSourceError(f"issue count missing/truncated: {asset} {monday}")
            counts.setdefault(monday, {})[asset] = value
        batches += 1
    if len(counts) != len(weeks) or any(set(counts[monday]) != {a for a, _ in REPOS}
                                       for monday in weeks):
        raise IssueSourceError("incomplete issue count grid")
    canonical = "".join(f"{monday},{counts[monday]['BTCUSDT']},"
                        f"{counts[monday]['ETHUSDT']}\n" for monday in weeks)
    return IssueSource(counts, sha256(canonical.encode("ascii")).hexdigest(),
                       len(weeks) * len(REPOS), batches)
