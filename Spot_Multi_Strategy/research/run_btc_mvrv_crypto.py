"""One-shot preregistered BTC MVRV five-crypto retrospective study."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from research import run_fed_reserves_crypto as frozen
from research.btc_mvrv_coinmetrics import MvrvDataError, load_and_audit as load_mvrv
from research.btc_mvrv_direction_adapter import encode
from research.fed_reserves_crypto_archive import (
    ArchiveIntegrityError, load_and_audit as load_crypto,
)
from research.fed_reserves_crypto_metrics import evaluate
from research.fed_reserves_crypto_sim import run_backtest
from research.run_btc_fee_crypto import _json_safe
from research.wiki_attention_sim import MarginRuinError, SimulationDataError


REPORT_DIR = frozen.ROOT / "Spot_Multi_Strategy" / "reports" / "mini_medallion_btc_mvrv_crypto_oos"
PREREG_COMMIT = "46d10cb"
AMENDMENT_COMMIT = "65cc374"
TERMINALS = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")
LOCK = ".mvrv_one_shot.lock"
EVIDENCE_GRADE = "preregistered_retrospective_current_vintage_btc_mvrv"


def _terminal_exists() -> bool:
    return any((REPORT_DIR / item).exists() for item in TERMINALS)


def _base(started: str, code_commit: str) -> dict:
    return {
        "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_commit": PREREG_COMMIT,
        "prereg_amendment_commit": AMENDMENT_COMMIT,
        "code_commit": code_commit,
        "evidence_grade": EVIDENCE_GRADE,
        "independent_executable_oos": False,
        "full_gate": False,
        "source_limitations": [
            "Coin Metrics MVRV is a current historical vintage; original daily publication times and historical as-of versions are unverified.",
            "D+1 visibility is an unverified proxy; Binance ZIPs may have been revised.",
            "Historical order-book impact, margin tiers and live fills are not fully observed.",
            "Coin Metrics and Binance Vision sources are for non-commercial research under their terms; live/commercial use requires separate rights.",
        ],
        "source_attribution": {
            "coin_metrics": {
                "credit": "Coin Metrics Community CapMVRVCur",
                "documentation": "https://docs.coinmetrics.io/network-data/network-data-overview/market/market-capitalization",
                "community_terms": "https://docs.coinmetrics.io/packages/coin-metrics-community-data",
                "modifications": "Daily MVRV was mapped to weekly 365-day median-relative directions.",
            },
            "binance_vision": {
                "credit": "Binance Vision USD-M historical archives",
                "terms": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
                "modifications": "Official bars and funding were transformed into derived backtest metrics.",
            },
        },
    }


def _write_once(name: str, payload: dict) -> Path:
    if name not in TERMINALS or _terminal_exists():
        raise FileExistsError("MVRV study already sealed or invalid terminal")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    body = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body)
    return path


def _data_failure(exc: Exception, started: str, code_commit: str,
                  sources: dict | None = None, signal: dict | None = None,
                  pnl_computed: bool = False) -> Path:
    result = {**_base(started, code_commit), "status": "data-ineligible",
              "reason": str(exc), "error_type": type(exc).__name__,
              "pnl_computed": pnl_computed, "all_gate_passed": False}
    if sources is not None:
        result["sources"] = sources
    if signal is not None:
        result["signal"] = signal
    return _write_once("DATA_FAILURE.json", result)


def _execute(started: str, code_commit: str) -> Path:
    try:
        mvrv = load_mvrv()
        coded, signal = encode(mvrv.observations)
    except MvrvDataError as exc:
        return _data_failure(exc, started, code_commit)
    sources = {"coinmetrics_btc_mvrv": mvrv.manifest}
    try:
        crypto = load_crypto()
    except ArchiveIntegrityError as exc:
        return _data_failure(exc, started, code_commit, sources, signal)
    sources["binance_usdm"] = crypto.manifest
    args = (coded, crypto.trade_bars, crypto.mark_bars, crypto.funding_events)
    try:
        primary = run_backtest(*args, slippage_rate=0.001)
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base(started, code_commit), "status": "sealed-risk-failure",
            "reason": str(exc), "error_type": type(exc).__name__,
            "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity,
            "stressed_equity_usdt_at_breach": exc.stressed_equity,
            "sources": sources, "signal": signal,
            "full_sample_metrics_computed": False, "all_gate_passed": False,
        })
    except SimulationDataError as exc:
        return _data_failure(exc, started, code_commit, sources, signal)
    stress_risk = None
    benchmark_risk = None
    try:
        stress = run_backtest(*args, slippage_rate=0.0025)
    except MarginRuinError as exc:
        stress, stress_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started, code_commit, sources, signal, True)
    try:
        benchmark = run_backtest(*args, slippage_rate=0.001, benchmark=True)
    except MarginRuinError as exc:
        benchmark, benchmark_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(exc, started, code_commit, sources, signal, True)
    metrics = evaluate(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    metrics["evidence_grade"] = EVIDENCE_GRADE
    metrics["independent_oos"] = False
    metrics["full_gate"] = False
    return _write_once("RESULT.json", {
        **_base(started, code_commit), "status": "sealed-result",
        "sources": sources, "signal": signal, "all_gate_passed": False,
        "counts": {"portfolio_trades": len(primary.trades),
                   "exposed_days": sum(bool(row.exposed) for row in primary.records),
                   "calendar_days": len(primary.records)},
        "account": {
            "initial_usdt": 10_000.0,
            "final_usdt": primary.final_equity,
            "fee_usdt": primary.fee_usdt,
            "slippage_usdt": primary.slippage_usdt,
            "funding_usdt": primary.funding_usdt,
            "turnover_usdt": primary.turnover_usdt,
            "stress_25bp_final_usdt": stress.final_equity if stress else None,
            "stress_25bp_risk_failure": stress_risk,
            "equal_weight_benchmark_final_usdt": benchmark.final_equity if benchmark else None,
            "equal_weight_benchmark_risk_failure": benchmark_risk,
        },
        "diagnostics": metrics,
    })


def run() -> Path:
    if not frozen._ancestor(PREREG_COMMIT) or not frozen._ancestor(AMENDMENT_COMMIT):
        raise RuntimeError("MVRV preregistration/amendment missing from HEAD")
    if _terminal_exists():
        raise FileExistsError("MVRV one-shot study already sealed")
    if not frozen._committed_and_pushed() or frozen._current_commit() == AMENDMENT_COMMIT:
        raise RuntimeError("MVRV code must be separately committed, pushed, and clean")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock = REPORT_DIR / LOCK
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("MVRV one-shot lock held; audit before retry") from exc
    os.close(descriptor)
    completed = False
    try:
        if _terminal_exists() or not frozen._committed_and_pushed():
            raise RuntimeError("MVRV study guard changed after lock")
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
        if sum((REPORT_DIR / item).exists() for item in TERMINALS) != 1:
            raise RuntimeError("MVRV study created invalid terminal count")
        completed = True
        return artifact
    finally:
        if completed:
            lock.unlink()


if __name__ == "__main__":
    print(run())
