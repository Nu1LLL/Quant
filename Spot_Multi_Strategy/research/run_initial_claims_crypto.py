"""Run the frozen U.S. initial-claims hypothesis exactly once.

Execute as ``python3 -m research.run_initial_claims_crypto`` from the project
root only after code and tests are committed and pushed. No live orders exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from research.fed_reserves_crypto_archive import _load_and_audit_impl, SYMBOLS
from research.tips_real_yield_metrics import MetricsDataError, evaluate
from research.initial_claims_sim import END, WARMUP, MarginRuinError, SimulationDataError, run_backtest
from research.initial_claims_source import ClaimsSourceError, load_and_audit
from research.wiki_attention_data import ArchiveIntegrityError, _http_fetch


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "Spot_Multi_Strategy" / "reports" / "mini_medallion_initial_claims_crypto_oos"
PREREG_COMMIT = "048b66d"
SEALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight() -> str:
    if any((REPORT_DIR / name).exists() for name in SEALS):
        raise FileExistsError("this historical hypothesis is already sealed")
    for commit in (PREREG_COMMIT,):
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                          cwd=ROOT, check=False, capture_output=True).returncode != 0:
            raise RuntimeError(f"preregistration/amendment {commit} must precede code")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("commit the implementation before reading formal data")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "refs/remotes/origin/main") != head:
        raise RuntimeError("push the implementation before reading formal data")
    return head


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"non-JSON research value: {type(value).__name__}")


def _serialize_payload(payload: dict) -> str:
    """Materialize the entire JSON before opening the irreversible seal."""
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False,
                      default=_json_default) + "\n"


def _write_once(name: str, payload: dict) -> Path:
    serialized = _serialize_payload(payload)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def _base(started: str, commit: str) -> dict:
    return {"experiment": "initial_claims_crypto_2026_11_08",
            "started_utc": started, "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": commit, "prereg_commit": PREREG_COMMIT,
            "evidence_grade": "preregistered_retrospective_current_icsa_temporal_holdout",
            "independent_executable_oos": False, "all_gate_passed": False,
            "historical_icsa_vintage_verified": False,
            "real_orders_placed": False}


def run() -> Path:
    commit = _preflight()
    started = datetime.now(timezone.utc).isoformat()
    try:
        claims = load_and_audit()
        market = _load_and_audit_impl(
            symbols=SYMBOLS, start=WARMUP, terminal=END,
            last_archive_month="2026-09", fetcher=_http_fetch, max_workers=8)
    except (ClaimsSourceError, ArchiveIntegrityError) as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base(started, commit), "status": "data-ineligible",
            "reason": str(exc), "error_type": type(exc).__name__,
            "pnl_computed": False})

    sources = {"icsa": claims.manifest, "binance": market.manifest}
    args = (claims.observations, market.trade_bars, market.mark_bars,
            market.funding_events)
    try:
        baseline = run_backtest(*args)
    except SimulationDataError as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base(started, commit), "status": "data-ineligible", "sources": sources,
            "reason": str(exc), "error_type": type(exc).__name__, "pnl_computed": False})
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base(started, commit), "status": "sealed-margin-failure", "sources": sources,
            "reason": str(exc), "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity, "full_metrics_computed": False})

    schedule = dict(baseline.capacity_schedule)
    paths = {}
    risk_failures = {}
    for name, kwargs in (
        ("cost_25bp", {"slippage_rate": 0.0025}),
        ("latency_1bar", {"latency_bars": 1}),
    ):
        try:
            paths[name] = run_backtest(*args, capacity_schedule=schedule, **kwargs)
        except MarginRuinError as exc:
            paths[name] = None
            risk_failures[name] = str(exc)
        except SimulationDataError as exc:
            return _write_once("DATA_FAILURE.json", {
                **_base(started, commit), "status": "data-ineligible", "sources": sources,
                "reason": str(exc), "error_type": type(exc).__name__,
                "baseline_pnl_computed": True})
    try:
        paths["equal_weight_buy_hold"] = run_backtest(*args, benchmark=True)
    except MarginRuinError as exc:
        paths["equal_weight_buy_hold"] = None
        risk_failures["equal_weight_buy_hold"] = str(exc)
    except SimulationDataError as exc:
        return _write_once("DATA_FAILURE.json", {
            **_base(started, commit), "status": "data-ineligible", "sources": sources,
            "reason": str(exc), "error_type": type(exc).__name__,
            "baseline_pnl_computed": True})
    try:
        diagnostics = evaluate(baseline, paths["equal_weight_buy_hold"],
                               market.trade_bars["BTCUSDT"], paths["cost_25bp"],
                               paths["latency_1bar"])
    except MetricsDataError as exc:
        return _write_once("CODE_FAILURE.json", {
            **_base(started, commit), "status": "sealed-analysis-failure", "sources": sources,
            "reason": str(exc), "error_type": type(exc).__name__,
            "baseline_pnl_computed": True})
    long_weeks = sum(1 for _, active in baseline.capacity_schedule if active)
    cash_weeks = len(baseline.capacity_schedule) - long_weeks
    payload = {
        **_base(started, commit), "status": "sealed-result", "sources": sources,
        "account": {"initial_usdt": 10_000.0, "final_usdt": baseline.final_equity,
                    "fee_usdt": baseline.fee_usdt, "slippage_usdt": baseline.slippage_usdt,
                    "funding_paid_usdt": baseline.funding_usdt,
                    "turnover_usdt": baseline.turnover_usdt,
                    "trade_count": len(baseline.trades),
                    "long_weeks": long_weeks, "cash_weeks": cash_weeks,
                    "capacity_breaches": baseline.capacity_breaches,
                    "minimum_order_skips": baseline.minimum_order_skips,
                    "execution_breaches": baseline.execution_breaches,
                    "terminal_flat": baseline.terminal_flat},
        "stress_final_usdt": {key: value.final_equity if value else None
                              for key, value in paths.items()},
        "stress_risk_failures": risk_failures,
        "diagnostics": diagnostics,
    }
    return _write_once("RESULT.json", payload)


if __name__ == "__main__":
    print(run())
