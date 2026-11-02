"""Synthetic-only one-shot runner plumbing tests; no market source is opened."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Spot_Multi_Strategy.research import run_cme_btc as runner
from Spot_Multi_Strategy.research.cme_btc_cot import COTDataError
from Spot_Multi_Strategy.research.cme_btc_spot import MonthManifest


class CmeBtcRunnerTests(unittest.TestCase):
    def test_source_failure_seals_without_calculating_pnl(self):
        with TemporaryDirectory() as tmp, \
             patch.object(runner, "REPORT_DIR", Path(tmp)), \
             patch.object(runner, "_ancestor", return_value=True), \
             patch.object(runner, "_current_commit", return_value="synthetic"), \
             patch.object(runner, "fetch_cot", side_effect=COTDataError("bad CFTC ZIP")), \
             patch.object(runner, "fetch_bars", return_value=SimpleNamespace(bars=(1,))), \
             patch.object(runner, "simulate") as simulation:
            path = runner.run()
            self.assertEqual(path.name, "DATA_FAILURE.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["pnl_computed"])
            simulation.assert_not_called()

    def test_success_serializes_dictionary_cot_manifest_and_spot_dataclass(self):
        cot = SimpleNamespace(reports=(object(),), manifest={"archive_count": 8})
        spot = SimpleNamespace(
            bars=(object(),),
            manifest=(MonthManifest("2018-12", "https://example.invalid/test.zip",
                                    "a" * 64, 100, 31),),
        )
        trade = SimpleNamespace(side="buy")
        ledger = SimpleNamespace(
            trades=(trade,), records=(SimpleNamespace(exposed=True),),
            final_equity=10_001.0, conservative_max_drawdown=-0.01,
            capacity_pass=True, execution_pass=True,
        )
        with TemporaryDirectory() as tmp, \
             patch.object(runner, "REPORT_DIR", Path(tmp)), \
             patch.object(runner, "_ancestor", return_value=True), \
             patch.object(runner, "_current_commit", return_value="synthetic"), \
             patch.object(runner, "fetch_cot", return_value=cot), \
             patch.object(runner, "fetch_bars", return_value=spot), \
             patch.object(runner, "simulate", return_value=ledger), \
             patch.object(runner, "buy_and_hold", return_value=ledger), \
             patch.object(runner, "evaluate_diagnostics", return_value={"full_gate": False}):
            path = runner.run()
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path.name, "RESULT.json")
            self.assertEqual(payload["sources"]["cftc_tff_years"]["archive_count"], 8)
            self.assertEqual(payload["sources"]["binance_spot_months"][0]["rows"], 31)
            self.assertEqual(payload["code_commit"], "synthetic")
            self.assertFalse(payload["independent_oos"])
            with self.assertRaises(FileExistsError):
                runner.run()


if __name__ == "__main__":
    unittest.main()
