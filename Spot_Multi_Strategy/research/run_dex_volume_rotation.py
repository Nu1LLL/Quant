"""Immutable one-shot writer for the preregistered three-chain DEX study."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from research.dex_volume_rotation_source import load_sources
from research.dex_volume_rotation_sim import (
    FEE, SLIP, equal_weight_buyhold, evaluate, frozen_targets, simulate,
)


RESULT = Path(__file__).parent / "results/three_chain_dex_volume_rotation_2026_11_08.json"
PREREG_COMMIT = "09edb76"


def run_one_shot() -> dict:
    if RESULT.exists():
        raise FileExistsError("formal DEX-rotation result already sealed; no rerun")
    started = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    data_verified = False
    try:
        dex, prices, terminal, audit = load_sources()
        data_verified = True
        targets = frozen_targets(dex)
        primary = simulate(targets, prices, terminal)
        gross = simulate(targets, prices, terminal, fee_rate=0, slippage=0)
        double_slip = simulate(targets, prices, terminal, slippage=2 * SLIP)
        extra_day = simulate(targets, prices, terminal, latency_days=1)
        hold = equal_weight_buyhold(prices, terminal)
        outcome = evaluate(primary, gross, double_slip, extra_day,
                           hold, prices["BTCUSDT"])
        status = ("numeric_pass_only_independent_oos_failed"
                  if outcome["gate"]["all_numeric"] else "evaluated_rejected")
        result = {"experiment_id": "three_chain_dex_volume_rotation_2026_11_08",
                  "status": status, "preregistration_commit": PREREG_COMMIT,
                  "runner_commit": head, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "data_verified": True, "source_audit": audit,
                  "fee_rate_per_fill": FEE, "slippage_per_fill": SLIP, **outcome}
    except Exception as exc:
        result = {"experiment_id": "three_chain_dex_volume_rotation_2026_11_08",
                  "status": "sealed_failed_run", "preregistration_commit": PREREG_COMMIT,
                  "runner_commit": head, "started_utc": started,
                  "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "data_verified": data_verified, "metrics": None,
                  "gate": {"full": False},
                  "failure_reason": f"{type(exc).__name__}: {exc}"}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


if __name__ == "__main__":
    result = run_one_shot()
    score = result.get("full", {})
    print(json.dumps({"status": result["status"], "result": str(RESULT),
                      "gate": result["gate"],
                      "failure_reason": result.get("failure_reason"),
                      "full_metrics": {key: score.get(key) for key in
                                       ("cagr", "sharpe", "max_drawdown",
                                        "profit_factor_daily_dollar")}}, sort_keys=True))
