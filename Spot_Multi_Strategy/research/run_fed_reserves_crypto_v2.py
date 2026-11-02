"""One-shot, source-protocol-only repair of the sealed H.4.1 crypto study.

The trading ledger, Binance source protocol, metrics, assets, and parameters
are imported unchanged from v1. Only the preregistered HTML parser differs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from research import run_fed_reserves_crypto as frozen
from research.fed_reserves_crypto_archive import (
    ArchiveIntegrityError, load_and_audit as load_crypto,
)
from research.fed_reserves_crypto_metrics import evaluate
from research.fed_reserves_crypto_sim import run_backtest
from research.fed_reserves_vintage_data_v2 import (
    VintageDataError, load_and_audit as load_fed,
)
from research.wiki_attention_sim import MarginRuinError, SimulationDataError


REPORT_DIR = frozen.ROOT / "Spot_Multi_Strategy" / "reports" / (
    "mini_medallion_fed_reserves_crypto_v2_oos"
)
PREREG_COMMIT = "478b552"


def _base_payload(started: str, code_commit: str) -> dict:
    return {
        **frozen._base_payload(started, code_commit),
        "prereg_commit": PREREG_COMMIT,
        "source_protocol": "official_h41_table1_continued_dom_v2",
        "previous_sealed_attempt": "3954675",
        "same_economic_hypothesis_as_v1": True,
        "independent_oos": False,
    }


def _write_once(name: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    serialized = json.dumps(payload, ensure_ascii=False, indent=2,
                            allow_nan=False, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def _data_failure(exc: Exception, *, started: str, code_commit: str,
                  sources: dict | None = None,
                  pnl_computed: bool = False) -> Path:
    payload = {
        **_base_payload(started, code_commit),
        "status": "data-ineligible", "reason": str(exc),
        "error_type": type(exc).__name__,
        "pnl_computed": pnl_computed, "pnl_published": False,
        "all_gate_passed": False,
    }
    if sources is not None:
        payload["sources"] = sources
    return _write_once("DATA_FAILURE.json", payload)


def run() -> Path:
    if not frozen._ancestor(PREREG_COMMIT):
        raise RuntimeError("v2 preregistration must precede source access")
    if any((REPORT_DIR / name).exists() for name in (
        "RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json",
    )):
        raise FileExistsError("the one-shot v2 study is already sealed")
    if not frozen._committed_and_pushed():
        raise RuntimeError("v2 study code must be committed, pushed, and worktree clean")
    started = datetime.now(timezone.utc).isoformat()
    code_commit = frozen._current_commit()
    try:
        return _execute_once(started=started, code_commit=code_commit)
    except Exception as exc:
        return _write_once("CODE_FAILURE.json", {
            **_base_payload(started, code_commit),
            "status": "code-failure", "reason": str(exc),
            "error_type": type(exc).__name__, "all_gate_passed": False,
        })


def _execute_once(*, started: str, code_commit: str) -> Path:
    try:
        fed = load_fed()
    except VintageDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit)
    sources = {"fed_h41": fed.manifest}
    try:
        crypto = load_crypto()
    except ArchiveIntegrityError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=sources)
    sources["binance_usdm"] = crypto.manifest

    args = (fed.observations, crypto.trade_bars, crypto.mark_bars,
            crypto.funding_events)
    try:
        primary = run_backtest(*args, slippage_rate=0.001)
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base_payload(started, code_commit),
            "status": "sealed-risk-failure", "reason": str(exc),
            "error_type": type(exc).__name__,
            "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity,
            "stressed_equity_usdt_at_breach": exc.stressed_equity,
            "sources": sources,
            "full_sample_metrics_computed": False,
            "all_gate_passed": False,
        })
    except SimulationDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=sources)

    stress_risk = None
    try:
        stress = run_backtest(*args, slippage_rate=0.0025)
    except MarginRuinError as exc:
        stress, stress_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=sources, pnl_computed=True)

    benchmark_risk = None
    try:
        benchmark = run_backtest(*args, slippage_rate=0.001, benchmark=True)
    except MarginRuinError as exc:
        benchmark, benchmark_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit,
                             sources=sources, pnl_computed=True)

    diagnostics = evaluate(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    return _write_once("RESULT.json", {
        **_base_payload(started, code_commit),
        "status": "sealed-result", "sources": sources,
        "all_gate_passed": False,  # retrospective history is never independent OOS
        "counts": {
            "portfolio_trades": len(primary.trades),
            "exposed_days": sum(bool(row.exposed) for row in primary.records),
            "calendar_days": len(primary.records),
        },
        "account": {
            "initial_usdt": 10_000.0,
            "final_usdt": primary.final_equity,
            "fee_usdt": primary.fee_usdt,
            "slippage_usdt": primary.slippage_usdt,
            "funding_usdt": primary.funding_usdt,
            "turnover_usdt": primary.turnover_usdt,
            "stress_25bp_final_usdt": stress.final_equity if stress is not None else None,
            "stress_25bp_conservative_max_drawdown": (
                stress.conservative_max_drawdown if stress is not None else None
            ),
            "stress_25bp_fees_usdt": stress.fee_usdt if stress is not None else None,
            "stress_25bp_slippage_usdt": (
                stress.slippage_usdt if stress is not None else None
            ),
            "stress_25bp_risk_failure": stress_risk,
            "equal_weight_benchmark_final_usdt": (
                benchmark.final_equity if benchmark is not None else None
            ),
            "equal_weight_benchmark_risk_failure": benchmark_risk,
        },
        "diagnostics": diagnostics,
    })


if __name__ == "__main__":
    print(run())
