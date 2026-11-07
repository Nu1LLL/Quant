"""Synthetic OI source, temporal signal and complete report tests."""

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
from types import SimpleNamespace
import csv
import json
import unittest
import zipfile

import pandas as pd

from research.btc_oi_deleveraging_data import (
    FIRST, HEADER, LAST, OiDataError, OiObservation, _parse_day,
)
from research.btc_oi_deleveraging_signal import encode
from research.fed_reserves_crypto_sim import _direction, _validate_observations
from research.fed_reserves_crypto_metrics import evaluate
from research.run_btc_oi_deleveraging import _result_payload
from research.strict_report_json import dumps


def _source(day: pd.Timestamp, *, invalid: int = 0,
            conflict: bool = False) -> tuple[bytes, bytes]:
    label = day.date().isoformat()
    name = f"BTCUSDT-metrics-{label}.zip"
    text = StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(HEADER)
    first = None
    for slot in range(288):
        stamp = (day + pd.Timedelta(minutes=5 * slot)).strftime("%Y-%m-%d %H:%M:%S")
        row = [stamp, "BTCUSDT", "0" if slot < invalid else "100", "0", "1", "1", "1", "1"]
        writer.writerow(row)
        if slot == 0:
            first = row
    writer.writerow(first if not conflict else [*first[:2], "99", *first[3:]])
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name[:-4] + ".csv", text.getvalue())
    body = bio.getvalue()
    check = f"{sha256(body).hexdigest()}  {name}\n".encode("ascii")
    return body, check


class OiStudyTests(unittest.TestCase):
    def test_duplicate_collapse_and_nonpositive_slot_median(self):
        day = FIRST
        zip_body, check = _source(day, invalid=41)
        observation, detail = _parse_day(day, zip_body, check)
        self.assertEqual(observation.value, Decimal("100"))
        self.assertEqual(detail["valid_slots"], 247)
        self.assertEqual(detail["duplicate_identical_rows"], 1)

    def test_conflicting_duplicate_or_too_few_slots_rejected(self):
        body, check = _source(FIRST, conflict=True)
        with self.assertRaisesRegex(OiDataError, "conflicting duplicate"):
            _parse_day(FIRST, body, check)
        body, check = _source(FIRST, invalid=58)
        with self.assertRaisesRegex(OiDataError, "fewer than 231"):
            _parse_day(FIRST, body, check)

    def test_four_week_contraction_known_on_first_monday(self):
        days = pd.date_range(FIRST, LAST, freq="W-WED", tz="UTC")
        target = pd.Timestamp("2020-12-30", tz="UTC")
        def observations(after_target: Decimal):
            return tuple(OiObservation(
                day + pd.Timedelta(days=1), day,
                Decimal("90") if day == target else
                after_target if day > target else Decimal("100"),
            ) for day in days)
        monday = pd.Timestamp("2021-01-04", tz="UTC")
        for future in (Decimal("1"), Decimal("1000")):
            coded, _ = encode(observations(future))
            releases, values = _validate_observations(coded)
            self.assertEqual(_direction(releases, values, monday), 1)
        coded, fingerprint = encode(tuple(OiObservation(
            day + pd.Timedelta(days=1), day, Decimal("100"),
        ) for day in days))
        releases, values = _validate_observations(coded)
        self.assertEqual(_direction(releases, values, monday), 0)
        self.assertEqual(fingerprint["weekly_long_count"], 0)

    def test_complete_result_serialization_with_annual_integer_keys(self):
        ledger = SimpleNamespace(
            trades=(), records=(SimpleNamespace(exposed=True),),
            final_equity=10_100.0, fee_usdt=2.0, slippage_usdt=1.0,
            funding_usdt=0.5, turnover_usdt=200.0,
            conservative_max_drawdown=-0.1,
        )
        report = _result_payload(
            datetime(2026, 11, 8, tzinfo=timezone.utc).isoformat(), "synthetic",
            {"source": {"archives": [{"date": "2021-01-01", "sha256": "a" * 64}]}},
            {"weekly_direction_bits_hex": "00"}, ledger, ledger, ledger,
            {"strategy_execution": {"conservative_max_drawdown_by_year": {2021: -0.1}}},
            None, None,
        )
        parsed = json.loads(dumps(report))
        self.assertEqual(parsed["diagnostics"]["strategy_execution"]
                         ["conservative_max_drawdown_by_year"]["2021"], -0.1)
        self.assertEqual(parsed["account"]["final_usdt"], 10_100.0)

    def test_full_frozen_diagnostics_serialize_from_synthetic_five_year_ledger(self):
        equity = 10_000.0
        records = []
        for index, day in enumerate(pd.date_range("2021-01-01", "2025-12-31", freq="D", tz="UTC")):
            change = 0.001 if index % 2 == 0 else -0.0008
            ending = equity * (1.0 + change)
            records.append(SimpleNamespace(
                day=day.date(), equity_open=equity, equity_next_open=ending,
                return_net=change, conservative_high_equity=max(equity, ending),
                conservative_low_equity=min(equity, ending), exposed=True,
            ))
            equity = ending
        ledger = SimpleNamespace(
            records=tuple(records), final_equity=equity, trades=(),
            conservative_max_drawdown=-0.10,
            conservative_max_drawdown_by_year={year: -0.10 for year in range(2021, 2026)},
            fee_usdt=1.0, slippage_usdt=1.0, funding_usdt=0.0,
            turnover_usdt=100.0, terminal_flat=True,
            capacity_breaches=0, execution_breaches=0, margin_proxy_breaches=0,
        )
        bars = pd.DataFrame({"close": 1.0}, index=pd.date_range(
            "2020-12-01", "2025-12-31 20:00:00", freq="4h", tz="UTC",
        ))
        diagnostics = evaluate(ledger, ledger, bars)
        report = _result_payload("synthetic", "synthetic", {}, {}, ledger,
                                 ledger, ledger, diagnostics, None, None)
        loaded = json.loads(dumps(report))
        self.assertEqual(len(loaded["diagnostics"]["annual_walk_forward"]), 5)
        self.assertEqual(len(loaded["diagnostics"]["monthly_stability"]["returns"]), 60)


if __name__ == "__main__":
    unittest.main()
