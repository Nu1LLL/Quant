"""Price-blind inventory of Binance's public listing-announcement catalog.

Only article IDs, codes, titles, release timestamps and response hashes are
retained. No article body, price, volume, signal, PnL or account field is read.
Slow sequential requests avoid repeating a prior HTTP-429 batch failure.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).parent
OUTPUT = ROOT / "preregistrations/2026-11-07_binance_listing_catalog_metadata.json"
URL = ("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
       "?type=1&pageNo={page}&pageSize=50&catalogId=48")
PAGE_SIZE = 50
PAGE_DELAY_SECONDS = 3.0


class CatalogFailure(ValueError):
    pass


def _fetch_page(page: int) -> tuple[dict, str]:
    url = URL.format(page=page)
    process = subprocess.run(
        ("curl", "--silent", "--show-error", "--location", "--max-time", "30",
         "--proto", "=https", "--proto-redir", "=https", "--write-out", "\nHTTP:%{http_code}",
         url), capture_output=True,
    )
    if process.returncode:
        raise CatalogFailure(f"catalog transport failure page {page}: {process.returncode}")
    body, marker, code = process.stdout.rpartition(b"\nHTTP:")
    if marker != b"\nHTTP:" or code != b"200":
        raise CatalogFailure(f"catalog HTTP {code.decode('ascii', 'replace')} page {page}")
    if len(body) > 500_000:
        raise CatalogFailure(f"oversized catalog page {page}")
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogFailure(f"invalid catalog JSON page {page}") from exc
    if payload.get("code") != "000000":
        raise CatalogFailure(f"catalog response error page {page}")
    catalogs = payload.get("data", {}).get("catalogs", [])
    if len(catalogs) != 1 or catalogs[0].get("catalogId") != 48:
        raise CatalogFailure(f"unexpected catalog page {page}")
    return catalogs[0], sha256(body).hexdigest()


def inventory() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("price-blind catalog manifest already exists")
    initial, initial_hash = _fetch_page(1)
    total = initial.get("total")
    if type(total) is not int or not 1 <= total <= 100_000:
        raise CatalogFailure("invalid catalog total")
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    articles = []
    hashes = []
    seen_codes, seen_ids = set(), set()
    for page in range(1, pages + 1):
        if page == 1:
            catalog, digest = initial, initial_hash
        else:
            time.sleep(PAGE_DELAY_SECONDS)
            catalog, digest = _fetch_page(page)
        rows = catalog.get("articles")
        expected = PAGE_SIZE if page < pages else total - PAGE_SIZE * (pages - 1)
        if catalog.get("total") != total or not isinstance(rows, list) or len(rows) != expected:
            raise CatalogFailure(f"catalog pagination changed at page {page}")
        hashes.append({"page": page, "sha256": digest, "rows": len(rows)})
        for row in rows:
            article_id, code, title, released = (
                row.get("id"), row.get("code"), row.get("title"), row.get("releaseDate"))
            if (type(article_id) is not int or article_id <= 0 or article_id in seen_ids
                    or not isinstance(code, str) or not re.fullmatch(r"[0-9a-f]{32}", code)
                    or code in seen_codes or not isinstance(title, str) or not title.strip()
                    or len(title) > 500 or type(released) is not int or released <= 0):
                raise CatalogFailure(f"malformed or duplicate catalog article page {page}")
            seen_ids.add(article_id)
            seen_codes.add(code)
            articles.append({"id": article_id, "code": code,
                             "title": title, "release_date_ms": released})
        if page % 10 == 0 or page == pages:
            print(f"catalog pages {page}/{pages}; articles {len(articles)}", flush=True)
    if len(articles) != total:
        raise CatalogFailure("catalog count differs from advertised total")
    dates = [datetime.fromtimestamp(row["release_date_ms"] / 1000, timezone.utc).date()
             for row in articles]
    result = {
        "source": "Binance public catalogId=48 New Cryptocurrency Listing",
        "source_url_template": URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_total": total,
        "page_size": PAGE_SIZE,
        "page_hashes": hashes,
        "first_release_date": str(min(dates)),
        "last_release_date": str(max(dates)),
        "articles": articles,
        "price_or_return_data_read": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


if __name__ == "__main__":
    result = inventory()
    print(json.dumps({key: result[key] for key in
                      ("catalog_total", "first_release_date", "last_release_date")},
                     sort_keys=True))
