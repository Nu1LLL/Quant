"""Synthetic-only tests for historical, non-survivor-filtered universe discovery."""

from datetime import date
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.vintage_universe import (
    VintageMetadataError,
    daily_archive_url,
    discover_vintage,
    list_usdt_directories,
)


def _xml(prefixes, *, truncated=False, token=None):
    rows = "".join(
        f"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/{name}/</Prefix></CommonPrefixes>"
        for name in prefixes
    )
    extra = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        '<Name>data.binance.vision</Name>'
        '<Prefix>data/futures/um/monthly/klines/</Prefix>'
        f'<IsTruncated>{str(truncated).lower()}</IsTruncated>{extra}{rows}'
        '</ListBucketResult>'
    ).encode()


class _Response:
    def __init__(self, status, content=b""):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, pages, head_status):
        self.pages = pages
        self.head_status = head_status
        self.get_params = []
        self.head_urls = []

    def get(self, _url, *, params, timeout):
        self.get_params.append(params)
        return _Response(200, self.pages[len(self.get_params) - 1])

    def head(self, url, *, allow_redirects, timeout):
        self.head_urls.append(url)
        return _Response(self.head_status[url])


class VintageUniverseTests(unittest.TestCase):
    def test_listing_pages_and_only_usdt_symbols(self):
        session = _Session(
            [_xml(["ZRXUSDT", "BTCUSD"], truncated=True, token="next=1"),
             _xml(["BATUSDT", "ZRXUSDT"])], {}
        )
        self.assertEqual(list_usdt_directories(session), ("BATUSDT", "ZRXUSDT"))
        self.assertEqual(session.get_params[1]["continuation-token"], "next=1")

    def test_historical_selection_does_not_query_future_dates(self):
        day = date(2020, 12, 31)
        bat = daily_archive_url("BATUSDT", day)
        zrx = daily_archive_url("ZRXUSDT", day)
        session = _Session([_xml(["BATUSDT", "ZRXUSDT"])], {bat: 200, zrx: 404})
        self.assertEqual(discover_vintage(session, day, workers=2), ("BATUSDT",))
        self.assertEqual(set(session.head_urls), {bat, zrx})
        self.assertTrue(all("2020-12-31" in url for url in session.head_urls))

    def test_ambiguous_http_status_fails_closed(self):
        day = date(2020, 12, 31)
        bat = daily_archive_url("BATUSDT", day)
        session = _Session([_xml(["BATUSDT"])], {bat: 503})
        with self.assertRaises(VintageMetadataError):
            discover_vintage(session, day, workers=1)

    def test_missing_continuation_token_fails_closed(self):
        session = _Session([_xml(["BATUSDT"], truncated=True)], {})
        with self.assertRaises(VintageMetadataError):
            list_usdt_directories(session)


if __name__ == "__main__":
    unittest.main()
