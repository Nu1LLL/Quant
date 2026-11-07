"""One-shot BTC open-interest contraction study, frozen before source read."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from research import run_fed_reserves_crypto as frozen
from research.btc_oi_deleveraging_data import OiDataError, load_and_audit as load_oi
from research.btc_oi_deleveraging_signal import encode
from research.fed_reserves_crypto_archive import (
    ArchiveIntegrityError, load_and_audit as load_crypto,
)
from research.fed_reserves_crypto_metrics import evaluate
from research.fed_reserves_crypto_sim import run_backtest
from research.strict_report_json import dumps
from research.wiki_attention_sim import MarginRuinError, SimulationDataError


REPORT_DIR = frozen.ROOT / "Spot_Multi_Strategy" / "reports" / "mini_medallion_btc_oi_deleveraging_oos"
PREREG_COMMIT = "7ec96de"
TERMINALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")
LOCK = ".btc_oi_one_shot.lock"
GRADE = "preregistered_retrospective_binance_oi_current_archive"


def _terminal_exists() -> bool:
    return any((REPORT_DIR / name).exists() for name in TERMINALS)


def _base(started: str, code_commit: str) -> dict:
    return {
        "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_commit": PREREG_COMMIT,
        "code_commit": code_commit,
        "evidence_grade": GRADE,
        "independent_executable_oos": False,
        "full_gate": False,
        "source_attribution": {
            "oi": {
                "credit": "Binance Vision USD-M BTCUSDT daily metrics",
                "field": "sum_open_interest",
                "terms": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
                "modifications": "Identical duplicate rows were collapsed; invalid nonpositive snapshots omitted; each Wednesday's valid snapshots were reduced to a median; four-week contraction became long/cash directions.",
            },
            "exchange": {
                "credit": "Binance Vision USD-M klines, markPriceKlines, fundingRate",
                "terms": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
                "modifications": "Official archives were reduced to derived five-year backtest metrics.",
            },
        },
        "limitations": [
            "Historical ZIPs are current versions; point-in-time file revisions and first publication times are unverified.",
            "The Wednesday D+1 visibility time is an unverified proxy, not a verified historical release timestamp.",
            "Real historical order-book impact, margin tiers and live fills are not fully observed.",
            "Binance Vision data are used for non-commercial personal research under its terms; commercial/live use requires separate rights.",
        ],
    }


def _write_once(name: str, payload: dict) -> Path:
    if name not in TERMINALS or _terminal_exists():
        raise FileExistsError("OI study already sealed or invalid terminal")
    body = dumps(payload)  # Full payload validated before opening a terminal file.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body)
    return path


def _data_failure(exc: Exception, started: str, commit: str,
                  sources: dict | None = None, signal: dict | None = None,
                  pnl_computed: bool = False) -> Path:
    payload = {**_base(started, commit), "status": "data-ineligible",
               "reason": str(exc), "error_type": type(exc).__name__,
               "pnl_computed": pnl_computed, "all_gate_passed": False}
    if sources is not None:
        payload["sources"] = sources
    if signal is not None:
        payload["signal"] = signal
    return _write_once("DATA_FAILURE.json", payload)


def _result_payload(started: str, commit: str, sources: dict, signal: dict,
                    primary: object, stress: object | None, benchmark: object | None,
                    metrics: dict, stress_risk: str | None,
                    benchmark_risk: str | None) -> dict:
    """Pure complete report assembly for synthetic serialization testing."""
    return {
        **_base(started, commit), "status": "sealed-result",
        "sources": sources, "signal": signal,
        "all_gate_passed": False,
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
            "stress_25bp_final_usdt": stress.final_equity if stress else None,
            "stress_25bp_conservative_max_drawdown": stress.conservative_max_drawdown if stress else None,
            "stress_25bp_fee_usdt": stress.fee_usdt if stress else None,
            "stress_25bp_slippage_usdt": stress.slippage_usdt if stress else None,
            "stress_25bp_funding_usdt": stress.funding_usdt if stress else None,
            "stress_25bp_risk_failure": stress_risk,
            "equal_weight_benchmark_final_usdt": benchmark.final_equity if benchmark else None,
            "equal_weight_benchmark_fee_usdt": benchmark.fee_usdt if benchmark else None,
            "equal_weight_benchmark_slippage_usdt": benchmark.slippage_usdt if benchmark else None,
            "equal_weight_benchmark_funding_usdt": benchmark.funding_usdt if benchmark else None,
            "equal_weight_benchmark_turnover_usdt": benchmark.turnover_usdt if benchmark else None,
            "equal_weight_benchmark_risk_failure": benchmark_risk,
        },
        "diagnostics": metrics,
    }


def _execute(started: str, commit: str) -> Path:
    try:
        oi = load_oi()
        coded, signal = encode(oi.observations)
    except OiDataError as exc:
        return _data_failure(exc, started, commit)
    sources = {"binance_btc_oi": oi.manifest}
    try:
        crypto = load_crypto()
    except ArchiveIntegrityError as exc:
        return _data_failure(exc, started, commit, sources, signal)
    sources["binance_usdm_execution"] = crypto.manifest
    args = (coded, crypto.trade_bars, crypto.mark_bars, crypto.funding_events)
    try:
        primary = run_backtest(*args, slippage_rate=0.001)
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base(started, commit), "status": "sealed-risk-failure",
            "reason": str(exc), "error_type": type(exc).__name__,
            "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity,
            "stressed_equity_usdt_at_breach": exc.stressed_equity,
            "sources": sources, "signal": signal,
            "full_sample_metrics_computed": False, "all_gate_passed": False,
        })
    except SimulationDataError as exc:
        return _data_failure(exc, started, commit, sources, signal)
    try:
        stress = run_backtest(*args, slippage_rate=0.0025)
        stress_risk = None
    except MarginRuinError as exc:
        stress, stress_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started, commit, sources, signal, pnl_computed=True)
    try:
        benchmark = run_backtest(*args, slippage_rate=0.001, benchmark=True)
        benchmark_risk = None
    except MarginRuinError as exc:
        benchmark, benchmark_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started, commit, sources, signal, pnl_computed=True)
    metrics = evaluate(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    metrics["evidence_grade"] = GRADE
    metrics["independent_oos"] = False
    metrics["full_gate"] = False
    return _write_once("RESULT.json", _result_payload(
        started, commit, sources, signal, primary, stress, benchmark, metrics,
        stress_risk, benchmark_risk,
    ))


def run() -> Path:
    if not frozen._ancestor(PREREG_COMMIT):
        raise RuntimeError("OI preregistration must precede source access")
    if _terminal_exists():
        raise FileExistsError("OI one-shot study already sealed")
    if not frozen._committed_and_pushed() or frozen._current_commit().startswith(PREREG_COMMIT):
        raise RuntimeError("OI code must be separately committed, pushed, and clean")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock = REPORT_DIR / LOCK
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("OI one-shot lock held; audit before retry") from exc
    os.close(descriptor)
    completed = False
    try:
        if _terminal_exists() or not frozen._committed_and_pushed():
            raise RuntimeError("OI repository guard changed after lock")
        started = datetime.now(timezone.utc).isoformat()
        commit = frozen._current_commit()
        try:
            artifact = _execute(started, commit)
        except FileExistsError:
            raise
        except Exception as exc:
            if _terminal_exists():
                raise
            artifact = _write_once("CODE_FAILURE.json", {
                **_base(started, commit), "status": "code-failure",
                "reason": str(exc), "error_type": type(exc).__name__,
                "all_gate_passed": False,
            })
        if sum((REPORT_DIR / name).exists() for name in TERMINALS) != 1:
            raise RuntimeError("OI one-shot run produced invalid terminal count")
        completed = True
        return artifact
    finally:
        if completed:
            lock.unlink()


if __name__ == "__main__":
    print(run())
