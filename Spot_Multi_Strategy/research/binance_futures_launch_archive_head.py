"""Price-blind HEAD inventory for the frozen Binance futures launch events.

No ZIP, CSV, bar, price, volume, funding amount or strategy outcome is read.
Archive presence is only a preliminary screen; complete checksum and row-grid
verification must still precede formal outcome calculation.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

import requests


HERE = Path(__file__).parent
BASE = HERE / "preregistrations/2026-11-07_binance_futures_launch_metadata.json"
AMENDMENT = HERE / "preregistrations/2026-11-07_binance_futures_launch_amendment_1.json"
OUTPUT = HERE / "preregistrations/2026-11-08_binance_futures_launch_archive_presence.json"
ROOT_URL = "https://data.binance.vision/data/futures/um"


class PresenceFailure(ValueError):
    pass


def _events(base: dict, amendment: dict) -> list[dict]:
    accepted = {row["article_code"] for row in base["rows"] if row["status"] == "accepted"}
    accepted.update(row["article_code"] for row in amendment["decisions"]
                    if row["new_status"] == "accepted_explicit_usdt_title")
    rows = sorted((row for row in base["rows"] if row["article_code"] in accepted),
                  key=lambda row: (row["first_utc_schedule"], row["article_code"]))
    if len(rows) != base["accepted_count"] + sum(
            row["new_status"] == "accepted_explicit_usdt_title"
            for row in amendment["decisions"]):
        raise PresenceFailure("accepted event count disagreement")
    unique = []
    seen = set()
    for row in rows:
        if row["contract_symbol"] in seen:
            continue  # First announced launch per exact contract symbol only.
        seen.add(row["contract_symbol"])
        unique.append(row)
    return unique


def _urls(symbol: str, launch: datetime) -> dict[str, str]:
    day = launch.date()
    out = {}
    for label, date in (("prior_day", day - timedelta(days=1)),
                        ("launch_day", day), ("next_day", day + timedelta(days=1))):
        date_text = date.isoformat()
        name = f"{symbol}-5m-{date_text}.zip"
        out[label] = f"{ROOT_URL}/daily/klines/{symbol}/5m/{name}"
    months = {launch.strftime("%Y-%m"),
              (launch + timedelta(days=2)).strftime("%Y-%m")}
    for month in sorted(months):
        name = f"{symbol}-fundingRate-{month}.zip"
        out[f"funding_{month}"] = f"{ROOT_URL}/monthly/fundingRate/{symbol}/{name}"
    return out


def _head(url: str) -> int:
    try:
        response = requests.head(url, allow_redirects=True, timeout=(10, 25))
    except requests.RequestException as exc:
        raise PresenceFailure(f"archive HEAD transport error: {url}") from exc
    if response.status_code not in (200, 404):
        raise PresenceFailure(f"archive HEAD {response.status_code}: {url}")
    return response.status_code


def inventory() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("archive presence inventory already exists")
    base_bytes, amendment_bytes = BASE.read_bytes(), AMENDMENT.read_bytes()
    if json.loads(amendment_bytes).get("base_sha256") != sha256(base_bytes).hexdigest():
        raise PresenceFailure("amendment does not match frozen source")
    events = _events(json.loads(base_bytes), json.loads(amendment_bytes))
    tasks = []
    for event in events:
        launch = datetime.fromisoformat(event["first_utc_schedule"])
        if launch.tzinfo is None or launch.utcoffset() != timedelta(0):
            raise PresenceFailure("launch time must be UTC")
        for label, url in _urls(event["contract_symbol"], launch).items():
            tasks.append((event["article_code"], label, url))
    codes: dict[tuple[str, str], int] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_head, url): (code, label) for code, label, url in tasks}
        for future in as_completed(futures):
            code, label = futures[future]
            try:
                codes[(code, label)] = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    rows = []
    for event in events:
        launch = datetime.fromisoformat(event["first_utc_schedule"])
        urls = _urls(event["contract_symbol"], launch)
        status = {label: codes[(event["article_code"], label)] for label in urls}
        rows.append({"article_code": event["article_code"],
                     "contract_symbol": event["contract_symbol"],
                     "launch_utc": event["first_utc_schedule"],
                     "http_status": status})
    result = {
        "source": ROOT_URL,
        "base_manifest_sha256": sha256(base_bytes).hexdigest(),
        "amendment_sha256": sha256(amendment_bytes).hexdigest(),
        "first_announcement_per_symbol": True,
        "event_count": len(events),
        "http_status_counts": dict(sorted(Counter(
            status for row in rows for status in row["http_status"].values()).items())),
        "rows": rows,
        "price_or_return_data_read": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    result = inventory()
    print(json.dumps({"event_count": result["event_count"],
                      "http_status_counts": result["http_status_counts"]},
                     sort_keys=True))
