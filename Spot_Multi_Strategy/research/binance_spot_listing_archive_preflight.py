"""Price-blind Binance spot-listing candidate and archive-presence audit.

This reads only the frozen official announcement catalog's titles/timestamps and
HTTP HEAD metadata. It never downloads a market bar or calculates a return.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import requests
import time


ROOT = Path(__file__).parent
CATALOG = ROOT / "preregistrations/2026-11-07_binance_listing_catalog_metadata.json"
OUTPUT = ROOT / "preregistrations/2026-11-08_binance_spot_listing_archive_presence.json"
TITLE_START = re.compile(r"^Binance (?:Will List|Lists) ", re.IGNORECASE)
SYMBOL = re.compile(r"\(([A-Z0-9]{2,20})\)")
PRODUCT_WORDS = re.compile(
    r"futures|margin|options|bstock|trading pairs|convert|launchpool|perpetual|index|ETF",
    re.IGNORECASE,
)
ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"


def _previous_month(month: str) -> str:
    year, number = map(int, month.split("-"))
    return f"{year - (number == 1):04d}-{12 if number == 1 else number - 1:02d}"


def _head(task: tuple[str, str]) -> dict:
    symbol, month = task
    url = ARCHIVE.format(symbol=symbol, month=month)
    for attempt in range(5):
        try:
            response = requests.head(url, timeout=25, allow_redirects=True)
            if response.status_code in (200, 404):
                break
        except requests.RequestException:
            if attempt == 4:
                raise
        if attempt == 4:
            raise RuntimeError(f"archive metadata HTTP failure: {response.status_code} {url}")
        time.sleep(1.5 * (attempt + 1))
    if response.status_code not in (200, 404):
        raise RuntimeError(f"archive metadata failure: {response.status_code} {url}")
    return {"symbol": symbol, "month": month, "status": response.status_code,
            "content_length": int(response.headers.get("Content-Length", 0))
            if response.status_code == 200 else None, "url": url}


def inventory() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    raw = CATALOG.read_bytes()
    catalog = json.loads(raw)
    rows = []
    for article in catalog["articles"]:
        published = datetime.fromtimestamp(article["release_date_ms"] / 1000, timezone.utc)
        title = article["title"]
        if not (2021 <= published.year <= 2025 and TITLE_START.match(title)
                and title.count("(") == 1 and title.count(")") == 1
                and not PRODUCT_WORDS.search(title)):
            continue
        match = SYMBOL.search(title)
        if match is None:
            continue
        ticker = match.group(1)
        rows.append({"code": article["code"], "title": title,
                     "published_ms": article["release_date_ms"],
                     "ticker": ticker, "symbol": ticker + "USDT",
                     "month": published.strftime("%Y-%m")})
    tasks = sorted({(row["symbol"], month) for row in rows
                    for month in (row["month"], _previous_month(row["month"]))})
    with ThreadPoolExecutor(max_workers=2) as pool:
        heads = list(pool.map(_head, tasks))
    by_key = {(item["symbol"], item["month"]): item for item in heads}
    for row in rows:
        row["event_month"] = by_key[(row["symbol"], row["month"])]
        row["prior_month"] = by_key[(row["symbol"], _previous_month(row["month"]))]
    result = {"source": "Binance official announcement catalog and Binance Vision spot 1h ZIP HEAD",
              "catalog_sha256": sha256(raw).hexdigest(),
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "price_or_return_data_read": False, "candidate_count": len(rows),
              "candidates": rows}
    with OUTPUT.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False, sort_keys=True)
        out.write("\n")
    return result


if __name__ == "__main__":
    report = inventory()
    from collections import Counter
    years = Counter(datetime.fromtimestamp(x["published_ms"] / 1000, timezone.utc).year
                    for x in report["candidates"])
    event_ok = sum(x["event_month"]["status"] == 200 for x in report["candidates"])
    prior_ok = sum(x["prior_month"]["status"] == 200 for x in report["candidates"])
    print({"candidate_count": report["candidate_count"], "years": dict(sorted(years.items())),
           "event_month_zip_200": event_ok, "prior_month_zip_200": prior_ok})
