"""Synthetic source-integrity tests; no real market outcomes are opened."""

import json
from pathlib import Path
import tempfile
import unittest

from research.crypto_blackbox.binance_spot_flow_prospective import (
    ACTIVATION_MS, GRACE_MS, HOUR_MS, SYMBOLS, FlowCollector, LateTradeError, Trade, _connect,
    parse_trade,
)
from research.crypto_blackbox.binance_spot_flow_coverage import CoverageError, audit


class BinanceSpotFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "flow.sqlite"
        self.db = _connect(self.path)
        self.collector = FlowCollector(self.db, self.path)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    @staticmethod
    def message(*, maker=False, agg_id=7, trade_ms=ACTIVATION_MS + 5000):
        return json.dumps({"stream": "adausdt@aggTrade", "data": {
            "e": "aggTrade", "s": "ADAUSDT", "a": agg_id,
            "T": trade_ms, "E": trade_ms + 100,
            "p": "1.25", "q": "4", "m": maker,
        }})

    def test_aggressor_direction_and_latency(self):
        event = parse_trade(self.message(maker=True), ACTIVATION_MS + 5500)
        self.assertEqual(event.buy_micros, 0)
        self.assertEqual(event.sell_micros, 5_000_000)
        with self.assertRaises(LateTradeError):
            parse_trade(self.message(), ACTIVATION_MS + 20_000)

    def test_late_trade_before_seal_invalidates_its_source_hour(self):
        c = self.collector
        c.connected(ACTIVATION_MS - 600_000)
        received = ACTIVATION_MS + 20_000
        with self.assertRaises(LateTradeError) as caught:
            parse_trade(self.message(trade_ms=ACTIVATION_MS + 5000), received)
        c.invalidate_late(caught.exception.symbol, caught.exception.trade_ms, received)
        c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS)
        status = self.db.execute("SELECT status FROM hourly_flow WHERE symbol='ADAUSDT'").fetchone()[0]
        self.assertEqual(status, "invalid")

    def test_late_trade_after_seal_append_only_incident_removes_coverage(self):
        c = self.collector
        c.connected(ACTIVATION_MS - 600_000)
        for symbol in SYMBOLS:
            c.ingest(Trade(symbol, 7, ACTIVATION_MS + 5000,
                           ACTIVATION_MS + 5100, 5_000_000, 0), ACTIVATION_MS + 5200)
        sealed = ACTIVATION_MS + HOUR_MS + GRACE_MS
        c.seal_due(sealed)
        self.assertEqual(audit(self.path, as_of_ms=sealed)["observed_hours_all_symbols"], 1)
        detected = sealed + 1000
        with self.assertRaises(LateTradeError) as caught:
            parse_trade(self.message(agg_id=8), detected)
        c.invalidate_late(caught.exception.symbol, caught.exception.trade_ms, detected)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM source_incident").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT status FROM hourly_flow WHERE symbol='ADAUSDT'").fetchone()[0],
                         "observed")
        report = audit(self.path, as_of_ms=detected)
        self.assertEqual(report["observed_hours_all_symbols"], 0)
        self.assertEqual(report["postseal_incident_hours"], 1)
        self.assertFalse(report["five_year_independent_oos_proven"])

    def test_continuous_hour_seals_once(self):
        c = self.collector
        c.connected(ACTIVATION_MS - 600_000)
        c.ingest(Trade("ADAUSDT", 7, ACTIVATION_MS + 5000,
                       ACTIVATION_MS + 5100, 5_000_000, 0), ACTIVATION_MS + 5200)
        c.ingest(Trade("ADAUSDT", 8, ACTIVATION_MS + 6000,
                       ACTIVATION_MS + 6100, 0, 2_000_000), ACTIVATION_MS + 6200)
        self.assertEqual(c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS), 12)
        self.assertEqual(c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS), 0)
        row = self.db.execute("""SELECT status,count,buys,sells,buy_micros,
                 sell_micros,first_id,last_id FROM hourly_flow WHERE symbol='ADAUSDT'""").fetchone()
        self.assertEqual(row, ("observed", 2, 1, 1, 5_000_000, 2_000_000, 7, 8))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM hourly_flow").fetchone()[0], 12)

    def test_gap_and_disconnect_fail_closed(self):
        c = self.collector
        c.connected(ACTIVATION_MS - 600_000)
        c.ingest(Trade("ADAUSDT", 7, ACTIVATION_MS + 5000,
                       ACTIVATION_MS + 5100, 5_000_000, 0), ACTIVATION_MS + 5200)
        c.ingest(Trade("ADAUSDT", 9, ACTIVATION_MS + 6000,
                       ACTIVATION_MS + 6100, 5_000_000, 0), ACTIVATION_MS + 6200)
        c.disconnected(ACTIVATION_MS + 30_000)
        c.connected(ACTIVATION_MS + 60_000)
        c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS)
        statuses = dict(self.db.execute(
            "SELECT symbol,status FROM hourly_flow WHERE window_start_ms=?",
            (ACTIVATION_MS,)))
        self.assertEqual(statuses["ADAUSDT"], "invalid")
        self.assertTrue(all(value == "invalid" for symbol, value in statuses.items()
                            if symbol != "ADAUSDT"))

    def test_restart_never_reconstructs_missing_hour_as_live(self):
        c = self.collector
        c.connected(ACTIVATION_MS + 1000)
        c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS)
        self.assertEqual(self.db.execute(
            "SELECT status FROM hourly_flow LIMIT 1").fetchone()[0], "unavailable")
        again = FlowCollector(self.db, self.path)
        again.connected(ACTIVATION_MS + HOUR_MS + 1000)
        again.seal_due(ACTIVATION_MS + 2 * HOUR_MS + GRACE_MS)
        self.assertEqual(self.db.execute(
            "SELECT COUNT(*) FROM hourly_flow").fetchone()[0], 24)

    def test_malformed_boundary_message_invalidates_unsealed_prior_hour(self):
        c = self.collector
        c.connected(ACTIVATION_MS - 600_000)
        c.invalidate_current(ACTIVATION_MS + HOUR_MS + 1000)
        c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS)
        self.assertEqual(self.db.execute(
            "SELECT status FROM hourly_flow WHERE symbol='ADAUSDT'"
        ).fetchone()[0], "invalid")

    def test_coverage_is_read_only_and_never_claims_alpha(self):
        missing = self.path.parent / "missing.sqlite"
        before = audit(missing, as_of_ms=ACTIVATION_MS + HOUR_MS + GRACE_MS)
        self.assertEqual(before["status"], "journal_missing")
        self.assertFalse(missing.exists())
        self.collector.connected(ACTIVATION_MS - 600_000)
        self.collector.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS)
        report = audit(self.path, as_of_ms=ACTIVATION_MS + HOUR_MS + GRACE_MS)
        self.assertEqual(report["observed_hours_all_symbols"], 0)
        self.assertEqual(report["status_counts"]["ADAUSDT"]["empty"], 1)
        self.assertFalse(report["five_year_independent_oos_proven"])

    def test_audit_rejects_gap_even_if_rows_claim_observed(self):
        c = self.collector
        c.connected(ACTIVATION_MS - 600_000)
        c.ingest(Trade("ADAUSDT", 7, ACTIVATION_MS + 5000,
                       ACTIVATION_MS + 5100, 5_000_000, 0), ACTIVATION_MS + 5200)
        c.seal_due(ACTIVATION_MS + HOUR_MS + GRACE_MS)
        second = ACTIVATION_MS + HOUR_MS
        with self.db:
            self.db.execute("""INSERT INTO hourly_flow VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("ADAUSDT", second, second + HOUR_MS + GRACE_MS,
                 "observed", 1, 1, 0, 5_000_000, 0, 9, 9,
                 second + 5000, second + 5000))
        with self.assertRaisesRegex(CoverageError, "ID gap"):
            audit(self.path, as_of_ms=second + HOUR_MS + GRACE_MS)


if __name__ == "__main__":
    unittest.main()
