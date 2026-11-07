"""Price-blind, reproducible futures launch-event source audit.

Reads only official announcement detail metadata and prose. It never requests
market prices, volumes, funding, or a strategy result. The strict title rule
keeps one USDT-margined contract per article and avoids ambiguous multi-asset
or pre-market announcements. The manifest records every inclusion/exclusion.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import html
import json
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).parent
CATALOG = HERE / "preregistrations/2026-11-07_binance_listing_catalog_metadata.json"
OUTPUT = HERE / "preregistrations/2026-11-07_binance_futures_launch_metadata.json"
DETAIL_URL = "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query?articleCode={code}"
TITLE = re.compile(
    r"^Binance Futures Will Launch (?:USDT-Margined|USDⓈ-M|USDⓈ-Margined) "
    r"([A-Z0-9]+) Perpetual Contracts?\b", re.IGNORECASE,
)
UTC_DATE = re.compile(
    r"(20\d\d[-/]\d\d[-/]\d\d)\s+(\d{1,2}:\d\d(?::\d\d)?)"
    r"\s*(AM|PM)?\s*\(UTC\)", re.IGNORECASE,
)
FIRST_YEAR = 2021
LAST_YEAR = 2025
DELAY_SECONDS = 1.0


class SourceFailure(ValueError):
    pass


def _detail(code: str) -> tuple[dict, str]:
    proc = subprocess.run(
        ("curl", "--silent", "--show-error", "--location", "--max-time", "30",
         "--proto", "=https", "--proto-redir", "=https",
         "--write-out", "\nHTTP:%{http_code}", DETAIL_URL.format(code=code)),
        capture_output=True,
    )
    if proc.returncode:
        raise SourceFailure(f"transport failure {code}: {proc.returncode}")
    body, marker, status = proc.stdout.rpartition(b"\nHTTP:")
    if marker != b"\nHTTP:" or status != b"200" or len(body) > 150_000:
        raise SourceFailure(f"detail HTTP/size failure {code}: {status!r}")
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceFailure(f"invalid detail JSON {code}") from exc
    if payload.get("code") != "000000" or not isinstance(payload.get("data"), dict):
        raise SourceFailure(f"detail response failure {code}")
    return payload["data"], sha256(body).hexdigest()


def _prose(body: str) -> str:
    try:
        tree = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceFailure("invalid article body tree") from exc
    words: list[str] = []

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        if node.get("node") == "text" and isinstance(node.get("text"), str):
            words.append(node["text"])
        for child in node.get("child", []):
            walk(child)

    walk(tree)
    return " ".join(html.unescape(" ".join(words)).split())


def _parse_utc(match: re.Match[str]) -> datetime:
    date, clock, meridiem = match.groups()
    date = date.replace("/", "-")
    pattern = "%Y-%m-%d %I:%M" if meridiem else "%Y-%m-%d %H:%M"
    if clock.count(":") == 2:
        pattern += ":%S"
    if meridiem:
        clock += " " + meridiem.upper()
        pattern += " %p"
    return datetime.strptime(date + " " + clock, pattern).replace(tzinfo=timezone.utc)


def _candidate(article: dict) -> tuple[str, str] | None:
    published = datetime.fromtimestamp(article["release_date_ms"] / 1000, timezone.utc)
    if not FIRST_YEAR <= published.year <= LAST_YEAR:
        return None
    title = article["title"]
    match = TITLE.match(title)
    if match is None or "Pre-Market" in title or "TradFi" in title:
        return None
    token = match.group(1).upper()
    if token.endswith(("USDC", "BUSD", "USD1", "USD")):
        return None
    symbol = token if token.endswith("USDT") else token + "USDT"
    return token, symbol


def inventory() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("futures launch source manifest already exists")
    catalog_bytes = CATALOG.read_bytes()
    catalog = json.loads(catalog_bytes)
    if catalog.get("catalog_total") != len(catalog.get("articles", [])):
        raise SourceFailure("incomplete listing catalog")
    candidates = [(a, _candidate(a)) for a in catalog["articles"]]
    candidates = [(a, c) for a, c in candidates if c is not None]
    rows = []
    for index, (article, (token, symbol)) in enumerate(candidates, start=1):
        if index > 1:
            time.sleep(DELAY_SECONDS)
        detail, digest = _detail(article["code"])
        if detail.get("code") != article["code"] or detail.get("id") != article["id"]:
            raise SourceFailure(f"detail identity mismatch {article['code']}")
        if detail.get("publishDate") != article["release_date_ms"]:
            raise SourceFailure(f"publication date mismatch {article['code']}")
        if not isinstance(detail.get("body"), str):
            raise SourceFailure(f"missing article body {article['code']}")
        prose = _prose(detail["body"])
        publication = datetime.fromtimestamp(article["release_date_ms"] / 1000, timezone.utc)
        stamps = []
        for match in UTC_DATE.finditer(prose[:1800]):
            try:
                stamps.append(_parse_utc(match))
            except ValueError:
                pass
        launch = stamps[0] if stamps else None
        status = "accepted"
        if launch is None:
            status = "no_utc_schedule"
        elif not publication <= launch <= publication + timedelta(days=7):
            status = "schedule_outside_publication_window"
        elif symbol not in prose[:1800].upper():
            status = "contract_not_confirmed_in_prose"
        rows.append({
            "article_id": article["id"], "article_code": article["code"],
            "article_title": article["title"], "detail_sha256": digest,
            "published_utc": publication.isoformat(), "title_token": token,
            "contract_symbol": symbol, "first_utc_schedule": launch.isoformat() if launch else None,
            "status": status,
        })
        if index % 25 == 0 or index == len(candidates):
            print(f"details {index}/{len(candidates)}", flush=True)
    result = {
        "source": "Binance official listing catalog and announcement detail API",
        "catalog_sha256": sha256(catalog_bytes).hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "single USDT-margined perpetual launch title; 2021-2025; no pre-market or TradFi title",
        "candidate_count": len(candidates),
        "accepted_count": sum(row["status"] == "accepted" for row in rows),
        "rows": rows,
        "price_or_return_data_read": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    result = inventory()
    print(json.dumps({key: result[key] for key in ("candidate_count", "accepted_count")},
                     sort_keys=True))
