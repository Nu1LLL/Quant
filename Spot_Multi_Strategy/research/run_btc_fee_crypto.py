"""One-shot BTC-fee long/cash study on five USD-M crypto perpetuals.

The Coin Metrics loader, long/cash adapter, and frozen exchange ledger have
separate responsibilities. This module orchestrates them only after the
preregistration and code have been separately pushed with a clean worktree.
It does not fetch data at import time or persist raw market/fee observations.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from research import run_fed_reserves_crypto as frozen
from research.btc_fee_coinmetrics_data import (
    BtcFeeDataError, load_and_audit as load_fee,
)
from research.btc_fee_long_cash_adapter import adapt_long_cash_observations
from research.fed_reserves_crypto_archive import (
    ArchiveIntegrityError, load_and_audit as load_crypto,
)
from research.fed_reserves_crypto_metrics import evaluate
from research.fed_reserves_crypto_sim import run_backtest
from research.wiki_attention_sim import MarginRuinError, SimulationDataError


ROOT = frozen.ROOT
REPORT_DIR = ROOT / "Spot_Multi_Strategy" / "reports" / (
    "mini_medallion_btc_fee_crypto_oos"
)
PREREG_COMMIT = "70ee332"
EVIDENCE_GRADE = "retrospective_current_vintage_btc_fee"
SEALED_NAMES = ("RESULT.json", "DATA_FAILURE.json", "CODE_FAILURE.json")
LOCK_NAME = ".btc_fee_one_shot.lock"
WEEK_COUNT = 271
WARMUP_WEEKS = 4


def _terminal_exists() -> bool:
    return any((REPORT_DIR / name).exists() for name in SEALED_NAMES)


def _json_safe(value: object) -> object:
    """Accept only explicit JSON-compatible report types; never stringify numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite numeric report value")
        return number
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite numeric report value")
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("report object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"unsupported report value type: {type(value).__name__}")


def _write_once(name: str, payload: dict) -> Path:
    if name not in SEALED_NAMES:
        raise ValueError("only preregistered terminal artifacts may be written")
    serialized = json.dumps(
        _json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if _terminal_exists():
        raise FileExistsError("a one-shot BTC fee terminal artifact already exists")
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
        "signal_publication_proxy": {
            "rule": "Wednesday D 00:00 UTC observation assumed visible D+1 day 00:00 UTC; Monday decision imposes four more days of lag",
            "actual_first_publication_time_verified": False,
            "historical_as_of_vintages_verified": False,
        },
        "source_attribution": {
            "coin_metrics": {
                "credit": "Coin Metrics Community BTC FeeTotNtv",
                "license": "CC BY-NC 4.0",
                "api_docs": "https://github.com/coinmetrics/docs-website/blob/master/api.md",
                "license_reference": "https://github.com/coinmetrics/data/blob/master/README.md",
                "modifications": "Daily native-unit BTC fees were aggregated into 28-day rolling sums and long/cash weekly directions.",
            },
            "binance_vision": {
                "credit": "Binance Vision USD-M historical archives",
                "license": "CC BY-NC-SA 4.0",
                "terms": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
                "modifications": "Historical bars and funding were transformed into derived backtest metrics.",
                "derived_report_metrics_license": "CC BY-NC-SA 4.0; non-commercial research only",
            },
        },
        "source_warnings": [
            "Coin Metrics historical fees are a current vintage, not verified historical as-of values.",
            "The D+1 visibility timestamp is an unverified research proxy, not an observed publication timestamp.",
            "Coin Metrics Community API data is for non-commercial use; commercial or live use needs appropriate permission (https://github.com/coinmetrics/docs-website/blob/master/api.md).",
            "Binance Vision datasets permit personal non-production backtesting only under their non-commercial terms; live trading requires separate authorization (https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md).",
            "BTC fees may reflect congestion or supply shocks rather than profitable demand and may follow prices.",
            "Binance public ZIPs may be revised; historical margin tiers and lot limits are proxies.",
            "Actual funding rows up to 100ms after nominal settlement are mapped to that hour.",
            "The 2021–2025 retrospective holdout is not an independent prospective OOS test.",
        ],
    }


def _data_failure(exc: Exception, *, started: str, code_commit: str,
                  sources: dict | None = None, signal: dict | None = None,
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


def _observation_value(observation: object) -> Decimal:
    if isinstance(observation, Mapping):
        if "value" not in observation:
            raise BtcFeeDataError("weekly observation lacks value")
        raw = observation["value"]
    else:
        try:
            raw = getattr(observation, "value")
        except AttributeError as exc:
            raise BtcFeeDataError("weekly observation lacks value") from exc
    if isinstance(raw, bool):
        raise BtcFeeDataError("weekly observation value is not numeric")
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BtcFeeDataError("weekly observation value is not numeric") from exc
    if not value.is_finite() or value < 0:
        raise BtcFeeDataError("weekly observation value is nonfinite or negative")
    return value


def _direction_fingerprint(observations: Sequence[object],
                           adapted: Sequence[object]) -> dict:
    """Pack 271 preregistered weekly long/cash flags, earliest week first.

    The first four Wednesdays lack a four-week comparator and are explicitly
    encoded as zero. Each later bit is one iff R_i > R_(i-4). Bits are packed
    most-significant-bit first; the unused final low bit is padded with zero.
    """
    if len(observations) != WEEK_COUNT or len(adapted) != WEEK_COUNT:
        raise BtcFeeDataError(f"expected {WEEK_COUNT} raw and adapted weekly observations")
    raw = tuple(_observation_value(item) for item in observations)
    encoded = tuple(_observation_value(item) for item in adapted)
    if any(value != 0 for value in encoded[:WARMUP_WEEKS]):
        raise BtcFeeDataError("adapted warmup values must be zero")
    flags = []
    for index in range(WARMUP_WEEKS, WEEK_COUNT):
        encoded_delta = encoded[index] - encoded[index - WARMUP_WEEKS]
        raw_increase = raw[index] > raw[index - WARMUP_WEEKS]
        if encoded_delta not in (0, 1) or bool(encoded_delta) != raw_increase:
            raise BtcFeeDataError("long/cash adapter disagrees with raw BTC fee direction")
        flags.append("1" if encoded_delta else "0")
    bits = "0" * WARMUP_WEEKS + "".join(flags)
    padding = (-len(bits)) % 8
    packed = int(bits + "0" * padding, 2).to_bytes(
        (len(bits) + padding) // 8, byteorder="big",
    ).hex()
    return {
        "weekly_direction_count": WEEK_COUNT,
        "weekly_direction_long_count": bits.count("1"),
        "weekly_direction_bits_hex": packed,
        "weekly_direction_bit_order": "chronological earliest Wednesday first; MSB-first; final low bit zero-padded",
        "weekly_direction_warmup_zero_bits": WARMUP_WEEKS,
        "weekly_direction_padding_bits": padding,
    }


def run() -> Path:
    """Execute at most once; never fetch before all repository guards pass."""
    if not frozen._ancestor(PREREG_COMMIT):
        raise RuntimeError("BTC fee preregistration must precede source access")
    if _terminal_exists():
        raise FileExistsError("the one-shot BTC fee study is already sealed")
    if (not frozen._committed_and_pushed() or
            frozen._current_commit().startswith(PREREG_COMMIT)):
        raise RuntimeError("study code must be separately committed, pushed, and worktree clean")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = REPORT_DIR / LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError("BTC fee one-shot lock is held; stale locks require manual audit") from exc
    os.close(descriptor)
    completed = False
    try:
        if _terminal_exists():
            completed = True
            raise FileExistsError("the one-shot BTC fee study was sealed by another process")
        if (not frozen._ancestor(PREREG_COMMIT) or
                not frozen._committed_and_pushed() or
                frozen._current_commit().startswith(PREREG_COMMIT)):
            raise RuntimeError("repository guard changed after BTC fee lock acquisition")
        started = datetime.now(timezone.utc).isoformat()
        code_commit = frozen._current_commit()
        try:
            result = _execute_once(started=started, code_commit=code_commit)
        except FileExistsError:
            raise  # Never rewrite a conflicting terminal as CODE_FAILURE.
        except Exception as exc:
            if _terminal_exists():
                raise  # A partial/conflicting terminal must fail closed.
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
            raise RuntimeError("one-shot run did not create exactly one terminal artifact")
        completed = True
        return result
    finally:
        if completed:
            lock_path.unlink()


def _execute_once(*, started: str, code_commit: str) -> Path:
    try:
        fee = load_fee()
    except BtcFeeDataError as exc:
        return _data_failure(exc, started=started, code_commit=code_commit)
    sources = {"coinmetrics_btc_fee": fee.manifest}
    try:
        adapted = adapt_long_cash_observations(fee.observations)
        signal = _direction_fingerprint(fee.observations, adapted)
    except (BtcFeeDataError, SimulationDataError) as exc:
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

    diagnostics = evaluate(primary, benchmark, crypto.trade_bars["BTCUSDT"])
    diagnostics["evidence_grade"] = EVIDENCE_GRADE
    diagnostics["independent_oos"] = False
    diagnostics["full_gate"] = False
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
