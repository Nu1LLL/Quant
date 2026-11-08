"""Price-blind source inventory for three-chain DEX activity rotation.

Reads only DEX date grids/validity and opaque SHA-256 digests, plus Binance
Vision .CHECKSUM statements. No candle ZIP, trading signal or return is read.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import time

import requests


ROOT = Path(__file__).parent
OUT_DEX = ROOT / "preregistrations/2026-11-08_three_chain_dex_volume_source_manifest.json"
OUT_PRICE = ROOT / "preregistrations/2026-11-08_bnb_trx_spot_daily_source_manifest.csv"
CHAINS = {"Ethereum": "ETHUSDT", "BSC": "BNBUSDT", "Tron": "TRXUSDT"}
FIRST, LAST = date(2020, 12, 1), date(2025, 12, 31)


class PreflightError(ValueError):
    pass


def _get(url: str) -> bytes:
    last = "unknown"
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=35)
            if response.status_code == 200:
                return response.content
            last = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        if attempt < 4:
            time.sleep(1 + attempt)
    raise PreflightError(f"source metadata fetch failed ({last}): {url}")


def _dex_one(chain: str) -> dict:
    url = (f"https://api.llama.fi/overview/dexs/{chain}"
           "?excludeTotalDataChartBreakdown=true&dataType=dailyVolume")
    body = _get(url)
    try:
        chart = json.loads(body)["totalDataChart"]
        selected = [x for x in chart if isinstance(x, list) and len(x) == 2
                    and FIRST <= datetime.fromtimestamp(x[0], timezone.utc).date() <= LAST]
        dates = [datetime.fromtimestamp(x[0], timezone.utc).date() for x in selected]
        count = (LAST - FIRST).days + 1
        if (len(selected) != count or dates != [FIRST + timedelta(days=i) for i in range(count)]
                or any(not isinstance(x[1], (int, float)) or x[1] <= 0 for x in selected)):
            raise PreflightError(f"{chain} DEX chart lacks positive complete date grid")
    except (ValueError, TypeError, OverflowError, KeyError) as exc:
        raise PreflightError(f"invalid {chain} DEX metadata") from exc
    digest = sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"chain": chain, "token_symbol": CHAINS[chain], "url": url,
            "first": str(FIRST), "last": str(LAST), "days": count,
            "historical_subset_sha256": digest,
            "historical_values_read_for_signal_or_result": False}


def _price_one(task: tuple[str, str]) -> dict:
    symbol, month = task
    name = f"{symbol}-1d-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{name}"
    statement = _get(url + ".CHECKSUM").strip()
    match = re.fullmatch(rb"([0-9a-f]{64})\s+\*?([^\s]+)\s*", statement)
    if match is None or match.group(2).decode("ascii") != name:
        raise PreflightError(f"invalid official checksum statement: {name}")
    return {"symbol": symbol, "month": month, "url": url,
            "official_zip_sha256": match.group(1).decode("ascii")}


def inventory() -> dict:
    if OUT_DEX.exists() or OUT_PRICE.exists():
        raise FileExistsError("source manifest already exists")
    with ThreadPoolExecutor(max_workers=3) as pool:
        dex = sorted(pool.map(_dex_one, CHAINS), key=lambda x: x["chain"])
    months = [(f"{year:04d}-{month:02d}") for year in range(2020, 2027)
              for month in range(1, 13)
              if (year, month) >= (2020, 12) and (year, month) <= (2026, 1)]
    tasks = [(symbol, month) for symbol in ("BNBUSDT", "TRXUSDT") for month in months]
    with ThreadPoolExecutor(max_workers=3) as pool:
        prices = list(pool.map(_price_one, tasks))
    if len(prices) != 124 or len({(x["symbol"], x["month"]) for x in prices}) != 124:
        raise PreflightError("incomplete daily price checksum manifest")
    manifest = {"source": "DefiLlama free DEX volume API, current historical snapshot",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "price_or_strategy_result_read": False, "chains": dex}
    with OUT_DEX.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with OUT_PRICE.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("symbol", "month", "url", "official_zip_sha256"))
        writer.writeheader()
        writer.writerows(prices)
    return {"dex_days_each": dex[0]["days"], "chains": [x["chain"] for x in dex],
            "price_zip_checksums": len(prices)}


if __name__ == "__main__":
    print(json.dumps(inventory(), sort_keys=True))
