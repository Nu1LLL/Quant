"""One-shot, preregistered fear-and-greed study on five crypto perpetuals.

This runner has no network activity at import time. Historical FNG values and
exchange archives are first read only after the preregistration and this code
have been separately committed and pushed with a clean worktree. The raw daily
response and exchange bars are never persisted by this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from research import run_fed_reserves_crypto as frozen
from research.fng_crypto_adapter import (
    adapt_fng_observations,
    signal_fingerprint,
)
from research.fng_crypto_archive import (
    FNGIntegrityError,
    load_and_audit as load_fng,
)
from research.fed_reserves_crypto_archive import (
    ArchiveIntegrityError,
    load_and_audit as load_crypto,
)
from research.fed_reserves_crypto_metrics import MetricsDataError, evaluate
from research.fed_reserves_crypto_sim import run_backtest
from research.future_report_json_contract import dumps_report
from research.wiki_attention_sim import MarginRuinError, SimulationDataError


ROOT = frozen.ROOT
REPORT_DIR = ROOT / "Spot_Multi_Strategy" / "reports" / (
    "mini_medallion_fng_crypto_oos"
)
PREREG_COMMIT = "b2db5a6"
EVIDENCE_GRADE = "retrospective_current_vintage_sentiment_snapshot_supported"
SEALED_NAMES = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")
LOCK_NAME = ".fng_crypto_one_shot.lock"
WEEK_COUNT = 261


def _terminal_exists() -> bool:
    return any((REPORT_DIR / name).exists() for name in SEALED_NAMES)


def _write_once(name: str, payload: dict) -> Path:
    """Preflight the whole JSON tree before creating an exclusive terminal."""
    if name not in SEALED_NAMES:
        raise ValueError("only registered FNG terminal artifacts may be written")
    serialized = dumps_report(payload)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if _terminal_exists():
        raise FileExistsError("a one-shot FNG terminal artifact already exists")
    path = REPORT_DIR / name
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def _base_payload(started: str, code_commit: str) -> dict:
    return {
        "started_utc": started,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "prereg_commit": PREREG_COMMIT,
        "evidence_grade": EVIDENCE_GRADE,
        "independent_oos": False,
        "full_gate": False,
        "source_attribution": {
            "alternative_me_fng": {
                "credit": "Alternative.me Crypto Fear & Greed Index",
                "source": "https://alternative.me/crypto/fear-and-greed-index/",
                "api": "https://api.alternative.me/fng/?limit=0",
                "modifications": (
                    "Historical daily observations were reduced to 261 Wednesday "
                    "integers and fixed long, short or cash directions."
                ),
            },
            "binance_vision": {
                "credit": "Binance Vision USD-M historical archives",
                "terms": (
                    "https://github.com/binance/binance-public-data/blob/master/"
                    "TERMS_AND_CONDITIONS.md"
                ),
                "modifications": (
                    "Historical trade and mark bars and funding were transformed "
                    "into derived backtest metrics."
                ),
            },
        },
        "source_warnings": [
            "FNG historical API is today's current vintage, not verified daily as-of vintages.",
            "Dated images and HTTP metadata support, but do not prove, historical values as published.",
            "The FNG index incorporates BTC price and volatility, so it is not price-independent.",
            "The 2024-2025 time holdout is retrospective, not independent forward OOS.",
            "Exchange archives may have been revised; actual order-book impact and historical margin tiers are proxies.",
            "Alternative.me attribution and Binance Vision data terms apply; this study authorizes no live trading.",
        ],
    }


def _data_failure(exc: Exception, *, started: str, code_commit: str,
                  sources: dict | None = None, signal: object | None = None,
                  pnl_computed: bool = False) -> Path:
    payload = {
        **_base_payload(started, code_commit),
        "status": "data-ineligible",
        "reason": str(exc),
        "error_type": type(exc).__name__,
        "pnl_computed": pnl_computed,
        "pnl_published": False,
        "all_gate_passed": False,
    }
    if sources is not None:
        payload["sources"] = sources
    if signal is not None:
        payload["signal"] = signal
    return _write_once("DATA_FAILURE.json", payload)


def run() -> Path:
    """Execute at most once; repository and lock guards precede source access."""
    if not frozen._ancestor(PREREG_COMMIT):
        raise RuntimeError("FNG preregistration must precede source access")
    if _terminal_exists():
        raise FileExistsError("the one-shot FNG study is already sealed")
    if (not frozen._committed_and_pushed() or
            frozen._current_commit().startswith(PREREG_COMMIT)):
        raise RuntimeError("FNG code must be separately committed, pushed, and worktree clean")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = REPORT_DIR / LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("FNG one-shot lock is held; stale locks require manual audit") from exc
    os.close(descriptor)
    completed = False
    try:
        if _terminal_exists():
            completed = True
            raise FileExistsError("the one-shot FNG study was sealed by another process")
        if (not frozen._ancestor(PREREG_COMMIT) or
                not frozen._committed_and_pushed() or
                frozen._current_commit().startswith(PREREG_COMMIT)):
            raise RuntimeError("repository guard changed after FNG lock acquisition")
        started = datetime.now(timezone.utc).isoformat()
        code_commit = frozen._current_commit()
        try:
            result = _execute_once(started=started, code_commit=code_commit)
        except FileExistsError:
            raise  # Conflicting terminal must not be rewritten as CODE_FAILURE.
        except Exception as exc:
            if _terminal_exists():
                raise  # Partial or conflicting terminal requires manual audit.
            result = _write_once("CODE_FAILURE.json", {
                **_base_payload(started, code_commit),
                "status": "code-failure",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "pnl_published": False,
                "all_gate_passed": False,
            })
        if (not isinstance(result, Path) or result.parent != REPORT_DIR or
                result.name not in SEALED_NAMES or not result.is_file() or
                sum((REPORT_DIR / name).exists() for name in SEALED_NAMES) != 1):
            raise RuntimeError("one-shot FNG run did not create exactly one terminal artifact")
        completed = True
        return result
    finally:
        if completed:
            lock_path.unlink()


def _execute_once(*, started: str, code_commit: str) -> Path:
    try:
        fng = load_fng()
    except FNGIntegrityError as exc:
        source = getattr(exc, "manifest", None)
        sources = {"alternative_me_fng": source} if source is not None else None
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
        )
    sources = {"alternative_me_fng": fng.manifest}

    try:
        adapted = adapt_fng_observations(fng.observations)
        signal = signal_fingerprint(fng.observations)
        if len(signal) != WEEK_COUNT:
            raise FNGIntegrityError(f"expected {WEEK_COUNT} Wednesday signal rows")
    except (FNGIntegrityError, SimulationDataError) as exc:
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
        )

    try:
        crypto = load_crypto()
    except ArchiveIntegrityError as exc:
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
            signal=signal,
        )
    sources["binance_usdm"] = crypto.manifest
    args = (
        adapted, crypto.trade_bars, crypto.mark_bars,
        crypto.funding_events,
    )

    try:
        primary = run_backtest(*args, slippage_rate=0.001)
    except MarginRuinError as exc:
        return _write_once("RESULT.json", {
            **_base_payload(started, code_commit),
            "status": "sealed-risk-failure",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "first_breach_utc": exc.timestamp.isoformat(),
            "equity_usdt_at_breach": exc.equity,
            "stressed_equity_usdt_at_breach": exc.stressed_equity,
            "sources": sources,
            "signal": signal,
            "full_sample_metrics_computed": False,
            "all_gate_passed": False,
        })
    except SimulationDataError as exc:
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
            signal=signal,
        )

    stress_risk = None
    try:
        stress = run_backtest(*args, slippage_rate=0.0025)
    except MarginRuinError as exc:
        stress, stress_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
            signal=signal, pnl_computed=True,
        )

    benchmark_risk = None
    try:
        benchmark = run_backtest(*args, slippage_rate=0.001, benchmark=True)
    except MarginRuinError as exc:
        benchmark, benchmark_risk = None, str(exc)
    except SimulationDataError as exc:
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
            signal=signal, pnl_computed=True,
        )

    try:
        diagnostics = evaluate(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    except MetricsDataError as exc:
        return _data_failure(
            exc, started=started, code_commit=code_commit, sources=sources,
            signal=signal, pnl_computed=True,
        )
    diagnostics["evidence_grade"] = EVIDENCE_GRADE
    diagnostics["independent_oos"] = False
    diagnostics["full_gate"] = False
    diagnostics["stress_25bp_survived"] = stress is not None
    diagnostics["benchmark_path_complete"] = benchmark is not None
    diagnostics["numerical_and_stress_gate"] = bool(
        diagnostics["numerical_gate"] and stress is not None
    )
    return _write_once("RESULT.json", {
        **_base_payload(started, code_commit),
        "status": "sealed-result",
        "sources": sources,
        "signal": signal,
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
            "stress_25bp_final_usdt": (
                stress.final_equity if stress is not None else None
            ),
            "stress_25bp_conservative_max_drawdown": (
                stress.conservative_max_drawdown if stress is not None else None
            ),
            "stress_25bp_fee_usdt": (
                stress.fee_usdt if stress is not None else None
            ),
            "stress_25bp_slippage_usdt": (
                stress.slippage_usdt if stress is not None else None
            ),
            "stress_25bp_funding_usdt": (
                stress.funding_usdt if stress is not None else None
            ),
            "stress_25bp_turnover_usdt": (
                stress.turnover_usdt if stress is not None else None
            ),
            "stress_25bp_risk_failure": stress_risk,
            "equal_weight_benchmark_final_usdt": (
                benchmark.final_equity if benchmark is not None else None
            ),
            "equal_weight_benchmark_fee_usdt": (
                benchmark.fee_usdt if benchmark is not None else None
            ),
            "equal_weight_benchmark_slippage_usdt": (
                benchmark.slippage_usdt if benchmark is not None else None
            ),
            "equal_weight_benchmark_funding_usdt": (
                benchmark.funding_usdt if benchmark is not None else None
            ),
            "equal_weight_benchmark_turnover_usdt": (
                benchmark.turnover_usdt if benchmark is not None else None
            ),
            "equal_weight_benchmark_risk_failure": benchmark_risk,
        },
        "diagnostics": diagnostics,
    })


if __name__ == "__main__":
    print(run())
