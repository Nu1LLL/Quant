"""Synthetic timing and classification checks; no historical IV/price reads."""

from datetime import date
import unittest

from research.btc_exchange_supply_source import SourceFailure
from research.crypto_blackbox.deribit_history_archive import ArchiveWindow
from research.deribit_iv_term_source import _monday_window, _week_from_archive


MONDAY = date(2021, 6, 14)


def _archive(front_iv: float, back_iv: float, *, exclude_one: bool = False,
             bad_iv: bool = False) -> ArchiveWindow:
    first, last = _monday_window(MONDAY)
    rows = []
    for index in range(10):
        front = index < 5
        row = {"trade_id": str(index + 1), "timestamp": first + index * 1000,
               "instrument_name": "BTC-25JUN21-40000-C" if front else
                                  "BTC-27AUG21-40000-P",
               "index_price": 40000, "amount": 1,
               "iv": front_iv if front else back_iv}
        if exclude_one and index == 0:
            row["block_trade_id"] = "B1"
        if bad_iv and index == 0:
            row["iv"] = None
        rows.append(row)
    return ArchiveWindow("BTC", "option", first, last, tuple(rows), "synthetic", 1, 0)


class DeribitIVTermTests(unittest.TestCase):
    def test_front_contango_long_and_inversion_cash(self):
        _, long_target, long_audit = _week_from_archive(MONDAY, _archive(50, 60))
        _, cash_target, cash_audit = _week_from_archive(MONDAY, _archive(70, 60))
        self.assertEqual((long_target, cash_target), (1, 0))
        self.assertEqual((long_audit["front_rows"], long_audit["back_rows"]), (5, 5))
        self.assertFalse(cash_audit["low_density"])

    def test_excluded_trade_forces_cash_without_replacing_week(self):
        _, target, audit = _week_from_archive(MONDAY, _archive(50, 60, exclude_one=True))
        self.assertEqual(target, 0)
        self.assertTrue(audit["low_density"])
        self.assertEqual(audit["front_rows"], 4)

    def test_missing_required_iv_fails_source(self):
        with self.assertRaises(SourceFailure):
            _week_from_archive(MONDAY, _archive(50, 60, bad_iv=True))

    def test_wrong_archive_window_fails(self):
        archive = _archive(50, 60)
        wrong = ArchiveWindow("ETH", "option", archive.start_ms, archive.end_ms,
                              archive.rows, archive.canonical_sha256, 1, 0)
        with self.assertRaises(SourceFailure):
            _week_from_archive(MONDAY, wrong)


if __name__ == "__main__":
    unittest.main()
