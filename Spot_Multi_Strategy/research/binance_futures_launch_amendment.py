"""Price-blind amendment for old Binance USDT-margined title syntax.

The original source manifest is immutable. This file only validates the
previously excluded explicit-USDT titles against their unchanged official
detail bodies. It never reads a market archive or strategy outcome.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import time

from .binance_futures_launch_preflight import _detail, _prose, SourceFailure


HERE = Path(__file__).parent
BASE = HERE / "preregistrations/2026-11-07_binance_futures_launch_metadata.json"
OUTPUT = HERE / "preregistrations/2026-11-07_binance_futures_launch_amendment_1.json"


def amend() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("source amendment already exists")
    original = BASE.read_bytes()
    base = json.loads(original)
    pending = [row for row in base["rows"]
               if row["status"] == "contract_not_confirmed_in_prose"
               and row["article_title"].startswith(
                   "Binance Futures Will Launch USDT-Margined ")]
    decisions = []
    for index, row in enumerate(pending, start=1):
        if index > 1:
            time.sleep(3.0)
        detail, digest = _detail(row["article_code"])
        if (digest != row["detail_sha256"] or detail.get("id") != row["article_id"]
                or detail.get("publishDate") != row["detail_publish_date_ms"]):
            raise SourceFailure("source detail changed; refuse silent amendment")
        prose = _prose(detail["body"])
        token = row["title_token"]
        prefix = prose[:650]
        confirmed = bool(
            re.search(rf"\b{re.escape(token)}\b", prefix, re.IGNORECASE)
            and re.search(r"USDT.margined", prefix, re.IGNORECASE)
            and re.search(r"USDT as collateral", prefix, re.IGNORECASE)
        )
        decisions.append({"article_code": row["article_code"],
                          "contract_symbol": row["contract_symbol"],
                          "new_status": "accepted_explicit_usdt_title" if confirmed
                                        else "still_excluded_unconfirmed",
                          "detail_sha256": digest})
        if index % 10 == 0 or index == len(pending):
            print(f"amendment details {index}/{len(pending)}", flush=True)
    result = {
        "base_sha256": sha256(original).hexdigest(),
        "reason": "old explicit USDT-margined article titles need not repeat the full ticker in prose",
        "price_or_return_data_read": False,
        "decisions": decisions,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    report = amend()
    print(json.dumps({"checked": len(report["decisions"]), "reinstated": sum(
        row["new_status"] == "accepted_explicit_usdt_title"
        for row in report["decisions"])}, sort_keys=True))
