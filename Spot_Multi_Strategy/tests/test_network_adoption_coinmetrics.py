"""Synthetic data-source and publication-lag checks; no market outcomes used."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
import unittest
from urllib.parse import parse_qs, urlencode, urlsplit

from research.network_adoption_coinmetrics import (
    API_URL,
    METRICS,
    NetworkMetricError,
    SYMBOL_TO_ASSET,
    address_growth_score,
    fetch_network_metrics,
)


class FakeResponse:
    status_code = 200

    def __init__(self, payload: object, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responder):
        self.responder = responder
        self.urls: list[str] = []

    def get(self, url, *, timeout, allow_redirects):
        assert timeout == (10, 45)
        assert allow_redirects is False
        self.urls.append(url)
        return self.responder(url)


def metric_row(asset: str, day: date, value: str | None = "10") -> dict[str, str | None]:
    return {
        "asset": asset,
        "time": f"{day.isoformat()}T00:00:00.000000000Z",
        "AdrActCnt": value,
        "AssetEODCompletionTime": str(int(datetime.combine(
            day + timedelta(days=1), datetime.min.time(), timezone.utc
        ).timestamp())),
    }


def asset_from_url(url: str) -> str:
    return parse_qs(urlsplit(url).query)["assets"][0]


class NetworkLoaderTests(unittest.TestCase):
    def test_fixed_universe_metrics_and_null_day_not_forward_filled(self):
        start, end = date(2020, 12, 30), date(2021, 1, 2)

        def respond(url):
            query = parse_qs(urlsplit(url).query)
            self.assertEqual(query["metrics"], [METRICS])
            self.assertEqual(query["frequency"], ["1d"])
            self.assertEqual(query["start_time"], ["2020-12-30"])
            self.assertEqual(query["end_time"], ["2021-01-01"])
            asset = asset_from_url(url)
            return FakeResponse({"data": [
                metric_row(asset, start),
                metric_row(asset, date(2020, 12, 31), None),
                metric_row(asset, date(2021, 1, 1)),
            ]})

        session = FakeSession(respond)
        result = fetch_network_metrics(start, end, opener=session)
        self.assertEqual(set(result), set(SYMBOL_TO_ASSET))
        self.assertEqual(len(session.urls), 7)
        self.assertEqual(set(result["BATUSDT"]), {start, date(2021, 1, 1)})
        self.assertEqual(result["BATUSDT"][start][0], 10.0)

    def test_pagination_and_repeated_day_fail_closed(self):
        start, end = date(2020, 1, 1), date(2020, 1, 3)

        def respond(url):
            parsed = urlsplit(url)
            query = parse_qs(parsed.query)
            asset = query["assets"][0]
            if "next_page_token" in query:
                return FakeResponse({"data": [metric_row(asset, date(2020, 1, 2))]})
            next_query = {key: values[0] for key, values in query.items()}
            next_query["next_page_token"] = "test-token"
            next_url = f"{API_URL}?{urlencode(next_query)}"
            return FakeResponse({"data": [metric_row(asset, start)],
                                 "next_page_url": next_url, "next_page_token": "test-token"})

        result = fetch_network_metrics(start, end, opener=FakeSession(respond))
        self.assertEqual(len(result["BATUSDT"]), 2)

        def duplicate(url):
            parsed = urlsplit(url)
            query = parse_qs(parsed.query)
            asset = query["assets"][0]
            if "next_page_token" in query:
                return FakeResponse({"data": [metric_row(asset, start)]})
            next_query = {key: values[0] for key, values in query.items()}
            next_query["next_page_token"] = "test-token"
            return FakeResponse({"data": [metric_row(asset, start)],
                                 "next_page_url": f"{API_URL}?{urlencode(next_query)}"})

        with self.assertRaisesRegex(NetworkMetricError, "duplicate day"):
            fetch_network_metrics(start, end, opener=FakeSession(duplicate))

    def test_bad_http_schema_utc_and_pagination_are_errors(self):
        start, end = date(2020, 1, 1), date(2020, 1, 2)
        errors = [
            (lambda url: FakeResponse({}, 503), "HTTP 503"),
            (lambda url: FakeResponse({"data": {}}), "schema"),
            (lambda url: FakeResponse({"data": [{**metric_row(asset_from_url(url), start),
                                                  "time": "2020-01-01T01:00:00Z"}]}), "non-midnight"),
            (lambda url: FakeResponse({"data": [metric_row(asset_from_url(url), start)],
                                       "next_page_url": "https://evil.example/v4/timeseries/asset-metrics"}),
             "leaves the fixed"),
            (lambda url: FakeResponse({"data": [metric_row("wrong", start)]}),
             "unexpected asset"),
            (lambda url: FakeResponse({"data": [metric_row(asset_from_url(url), start, "NaN")]}),
             "not a finite"),
            (lambda url: FakeResponse({"data": [{**metric_row(asset_from_url(url), start),
                                                  "AssetEODCompletionTime": "not-time"}]}),
             "invalid AssetEODCompletionTime"),
        ]
        for responder, expected in errors:
            with self.subTest(expected=expected), self.assertRaisesRegex(NetworkMetricError, expected):
                fetch_network_metrics(start, end, opener=FakeSession(responder))

    def test_null_entire_asset_is_not_silently_all_cash(self):
        start, end = date(2020, 1, 1), date(2020, 1, 2)
        session = FakeSession(lambda url: FakeResponse({"data": [
            metric_row(asset_from_url(url), start, None)
        ]}))
        with self.assertRaisesRegex(NetworkMetricError, "no supported metrics"):
            fetch_network_metrics(start, end, opener=session)


class AddressScoreTests(unittest.TestCase):
    def setUp(self):
        self.decision = datetime(2021, 3, 1, 8, tzinfo=timezone.utc)
        friday = date(2021, 2, 26)
        self.points = {}
        for lag in range(34, -1, -1):
            day = friday - timedelta(days=lag)
            value = 10.0 if lag >= 7 else 20.0
            self.points[day] = (value, self.decision - timedelta(hours=1))

    def test_exact_prior_28_and_recent_7(self):
        self.assertAlmostEqual(address_growth_score(self.points, self.decision), math.log(2))

    def test_missing_zero_late_and_friday_cutoff(self):
        friday = date(2021, 2, 26)
        for mutation in (
            lambda p: p.pop(friday - timedelta(days=34)),
            lambda p: p.__setitem__(friday, (0.0, self.decision - timedelta(hours=1))),
            lambda p: p.__setitem__(friday, (20.0, self.decision + timedelta(seconds=1))),
        ):
            with self.subTest(mutation=mutation):
                points = dict(self.points)
                mutation(points)
                self.assertIsNone(address_growth_score(points, self.decision))
        points = dict(self.points)
        points[date(2021, 2, 27)] = (999999.0, self.decision - timedelta(hours=1))
        self.assertAlmostEqual(address_growth_score(points, self.decision), math.log(2))

    def test_only_monday_eight_utc(self):
        with self.assertRaisesRegex(ValueError, "Monday"):
            address_growth_score(self.points, self.decision + timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "timezone-aware UTC"):
            address_growth_score(self.points, self.decision.replace(tzinfo=None))


if __name__ == "__main__":
    unittest.main()
