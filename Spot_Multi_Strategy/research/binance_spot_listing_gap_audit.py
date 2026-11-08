"""Reproduce the source gap that sealed the registered listing experiment.

Only timestamps in the checksum-verified Binance ZIP are inspected. No prices,
volumes, signal, trade or return are parsed or calculated.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import re
import zipfile

import requests


ROOT = Path(__file__).parent
MANIFEST = ROOT / "preregistrations/2026-11-08_binance_spot_listing_archive_presence.json"
URL = "https://data.binance.vision/data/spot/monthly/klines/BADGERUSDT/1h/BADGERUSDT-1h-2021-03.zip"
EXPECTED_SHA = "a84fa68fa707228d42729b119409f4332470faee2478da8d6b4f453fcb0b6549"
EXPECTED_ARTICLE = "a2aa2ee0e80e4291aa8d3f22f9793948"


def audit() -> dict:
    candidates = json.loads(MANIFEST.read_text(encoding="utf-8"))["candidates"]
    article = next(x for x in candidates if x["ticker"] == "BADGER")
    if (article["code"] != EXPECTED_ARTICLE or article["month"] != "2021-03"
            or article["event_month"]["status"] != 200
            or article["prior_month"]["status"] != 404):
        raise ValueError("frozen candidate metadata changed")
    statement = requests.get(URL + ".CHECKSUM", timeout=30)
    statement.raise_for_status()
    match = re.fullmatch(rb"([0-9a-f]{64})\s+\*?([^\s]+)\s*", statement.content.strip())
    if match is None or match.group(2) != b"BADGERUSDT-1h-2021-03.zip":
        raise ValueError("invalid official checksum statement")
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    digest = sha256(response.content).hexdigest()
    if digest != EXPECTED_SHA or digest != match.group(1).decode("ascii"):
        raise ValueError("archived ZIP changed or checksum mismatched")
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        if archive.namelist() != ["BADGERUSDT-1h-2021-03.csv"]:
            raise ValueError("archive membership changed")
        with archive.open("BADGERUSDT-1h-2021-03.csv") as stream:
            rows = csv.reader(TextIOWrapper(stream, encoding="utf-8"))
            timestamps = [datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc)
                          for row in rows]
    if not timestamps or len(set(timestamps)) != len(timestamps):
        raise ValueError("missing or duplicate archive timestamps")
    t0 = timestamps[0]
    expected = [t0 + timedelta(hours=i) for i in range(97)]
    actual = set(timestamps)
    missing = [t.isoformat() for t in expected if t not in actual]
    return {"symbol": "BADGERUSDT", "article_code": article["code"],
            "published_utc": datetime.fromtimestamp(article["published_ms"] / 1000,
                                                    timezone.utc).isoformat(),
            "archive_sha256": digest, "t0": t0.isoformat(),
            "required_96h_missing_count": len(missing),
            "required_96h_missing_utc": missing,
            "price_or_return_data_read": False}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
