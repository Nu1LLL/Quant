"""Discover a historical Binance USD-M candidate universe without future survival filters.

This reads only the S3 directory listing and HEAD metadata for one fixed day.
It never downloads prices, strategy returns, or raw ZIP archives. Presence of a
historical archive is *not* proof of an exchangeInfo snapshot, liquidity, or
continuous tradability; a later preregistration must audit those separately.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests


S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
PREFIX = "data/futures/um/monthly/klines/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SYMBOL_RE = re.compile(r"[A-Z0-9]+USDT\Z")


class VintageMetadataError(RuntimeError):
    """Historical archive metadata cannot be interpreted without guesswork."""


def list_usdt_directories(session: requests.Session, *, max_pages: int = 20) -> tuple[str, ...]:
    """List all symbol directories; no later file-presence test is made here."""
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    names: set[str] = set()
    token = None
    seen_tokens: set[str] = set()
    for _ in range(max_pages):
        params = {"list-type": "2", "prefix": PREFIX, "delimiter": "/", "max-keys": "1000"}
        if token is not None:
            params["continuation-token"] = token
        response = session.get(S3_LIST_URL, params=params, timeout=(10, 30))
        response.raise_for_status()
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise VintageMetadataError("invalid S3 listing XML") from exc
        if root.findtext("s3:Name", namespaces=NS) != "data.binance.vision":
            raise VintageMetadataError("unexpected bucket name")
        if root.findtext("s3:Prefix", namespaces=NS) != PREFIX:
            raise VintageMetadataError("unexpected S3 prefix")
        for node in root.findall("s3:CommonPrefixes/s3:Prefix", namespaces=NS):
            path = node.text or ""
            if not (path.startswith(PREFIX) and path.endswith("/")):
                raise VintageMetadataError("malformed symbol directory")
            name = path[len(PREFIX):-1]
            if SYMBOL_RE.fullmatch(name):
                names.add(name)
        truncated = root.findtext("s3:IsTruncated", namespaces=NS)
        if truncated == "false":
            return tuple(sorted(names))
        if truncated != "true":
            raise VintageMetadataError("missing S3 truncation state")
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not token or token in seen_tokens:
            raise VintageMetadataError("missing or repeated S3 continuation token")
        seen_tokens.add(token)
    raise VintageMetadataError("S3 listing exceeded max_pages")


def daily_archive_url(symbol: str, trading_day: date) -> str:
    if not SYMBOL_RE.fullmatch(symbol):
        raise ValueError("symbol must be an uppercase USDT contract")
    if not isinstance(trading_day, date):
        raise TypeError("trading_day must be a date")
    name = f"{symbol}-1d-{trading_day.isoformat()}.zip"
    return f"{ARCHIVE_URL}/{quote(symbol)}/1d/{quote(name)}"


def historical_archive_exists(session: requests.Session, symbol: str, trading_day: date) -> bool:
    """404 means absent; transient/authorization/server errors fail closed."""
    url = daily_archive_url(symbol, trading_day)
    response = session.head(url, allow_redirects=True, timeout=(10, 30))
    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False
    raise VintageMetadataError(f"{symbol}: archive HEAD status {response.status_code}")


def discover_vintage(
    session: requests.Session, trading_day: date, *, workers: int = 12
) -> tuple[str, ...]:
    """Select symbols only by evidence from ``trading_day`` itself."""
    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    names = list_usdt_directories(session)
    selected: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(historical_archive_exists, session, name, trading_day): name
            for name in names
        }
        for future in as_completed(pending):
            if future.result():
                selected.add(pending[future])
    return tuple(sorted(selected))


if __name__ == "__main__":
    with requests.Session() as http:
        day = date(2020, 12, 31)
        symbols = discover_vintage(http, day)
    print(json.dumps({"as_of": day.isoformat(), "count": len(symbols), "symbols": symbols}))
