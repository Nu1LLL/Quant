"""Frozen diagnostics for the five-coin Wikipedia-attention experiment.

The simulator supplies audited daily ledgers and the ordered 4h stress-path
drawdowns. This module neither downloads observations nor changes signals.
The experiment was registered after its historical sample; a time holdout is
therefore descriptive, never independent out-of-sample evidence.
"""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


START_DAY = date(2021, 1, 1)
END_DAY_EXCLUSIVE = date(2026, 1, 1)
YEARS = tuple(range(2021, 2026))
ANNUAL_DAYS = 365.25
MC_PATHS = 10_000
MC_BLOCK_DAYS = 21
MC_SEED = 20261102
REGIMES = ("bull", "bear", "sideways")


class MetricsDataError(ValueError):
    """Required ledger or BTC market-state information is invalid."""


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MetricsDataError(f"invalid {label}") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise MetricsDataError(f"invalid {label}")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MetricsDataError(f"invalid {label}")
    return value


def _checked_records(ledger: object, label: str) -> tuple[object, ...]:
    try:
        rows = tuple(ledger.records)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise MetricsDataError(f"missing {label} daily ledger") from exc
    expected_days = (END_DAY_EXCLUSIVE - START_DAY).days
    if len(rows) != expected_days:
        raise MetricsDataError(f"{label} daily grid has {len(rows)} instead of {expected_days} days")
    previous_end: float | None = None
    for offset, row in enumerate(rows):
        day = getattr(row, "day", None)
        if type(day) is not date or day != START_DAY + timedelta(days=offset):
            raise MetricsDataError(f"{label} daily date gap or duplicate at offset {offset}")
        opening = _finite(getattr(row, "equity_open", None), f"{label} opening equity", positive=True)
        ending = _finite(getattr(row, "equity_next_open", None), f"{label} ending equity", positive=True)
        ret = _finite(getattr(row, "return_net", None), f"{label} daily return")
        high = _finite(getattr(row, "conservative_high_equity", None),
                       f"{label} 4h stressed high", positive=True)
        low = _finite(getattr(row, "conservative_low_equity", None),
                      f"{label} 4h stressed low", positive=True)
        if high < low or not math.isclose(ending / opening - 1.0, ret,
                                         rel_tol=1e-10, abs_tol=1e-12):
            raise MetricsDataError(f"inconsistent {label} daily record at {day}")
        if previous_end is not None and not math.isclose(opening, previous_end,
                                                        rel_tol=1e-10, abs_tol=1e-7):
            raise MetricsDataError(f"discontinuous {label} equity at {day}")
        if type(getattr(row, "exposed", None)) is not bool:
            raise MetricsDataError(f"invalid {label} exposure at {day}")
        if offset == 0 and not math.isclose(opening, 10_000.0, rel_tol=0, abs_tol=1e-7):
            raise MetricsDataError(f"{label} initial equity is not 10,000 USDT")
        previous_end = ending
    final = _finite(getattr(ledger, "final_equity", None), f"{label} final equity",
                    positive=True)
    if not math.isclose(final, previous_end, rel_tol=1e-10, abs_tol=1e-7):
        raise MetricsDataError(f"{label} final equity differs from daily ledger")
    return rows


def _ledger_metadata(ledger: object, label: str) -> dict[str, Any]:
    """Do not silently treat omitted costs, execution checks, or stress as zero."""
    by_year = getattr(ledger, "conservative_max_drawdown_by_year", None)
    if not isinstance(by_year, dict) or set(by_year) != set(YEARS):
        raise MetricsDataError(f"missing {label} exact annual 4h stress drawdowns")
    stressed = _finite(getattr(ledger, "conservative_max_drawdown", None),
                       f"{label} exact 4h stress drawdown")
    annual = {year: _finite(by_year[year], f"{label} {year} exact 4h stress drawdown")
              for year in YEARS}
    if not -1.0 <= stressed <= 0.0 or any(not -1.0 <= dd <= 0.0 for dd in annual.values()):
        raise MetricsDataError(f"invalid {label} 4h stress drawdown range")
    if stressed > min(annual.values()) + 1e-10:
        raise MetricsDataError(f"{label} global stress drawdown is milder than an annual fold")
    fee = _finite(getattr(ledger, "fee_usdt", None), f"{label} fees")
    slippage = _finite(getattr(ledger, "slippage_usdt", None), f"{label} slippage")
    funding = _finite(getattr(ledger, "funding_usdt", None), f"{label} funding")
    turnover = _finite(getattr(ledger, "turnover_usdt", None), f"{label} turnover")
    if fee < 0 or slippage < 0 or turnover < 0:
        raise MetricsDataError(f"invalid {label} cost or turnover sign")
    trades = getattr(ledger, "trades", None)
    if not isinstance(trades, tuple):
        raise MetricsDataError(f"invalid {label} trade sequence")
    terminal_flat = getattr(ledger, "terminal_flat", None)
    if type(terminal_flat) is not bool:
        raise MetricsDataError(f"invalid {label} terminal-flat flag")
    return {
        "conservative_max_drawdown": stressed,
        "conservative_max_drawdown_by_year": annual,
        "fee_usdt": fee,
        "slippage_usdt": slippage,
        "funding_usdt": funding,
        "turnover_usdt": turnover,
        "trades": len(trades),
        "terminal_flat": terminal_flat,
        "capacity_breaches": _nonnegative_int(getattr(ledger, "capacity_breaches", None),
                                               f"{label} capacity breaches"),
        "execution_breaches": _nonnegative_int(getattr(ledger, "execution_breaches", None),
                                                f"{label} execution breaches"),
        "margin_proxy_breaches": _nonnegative_int(getattr(ledger, "margin_proxy_breaches", None),
                                                   f"{label} margin proxy breaches"),
    }


def _returns_metrics(values: Sequence[float], *, starting_equity: float = 1.0) -> dict[str, Any]:
    """Daily-return metrics; undefined variance or loss denominator never passes."""
    if not values:
        raise MetricsDataError("empty daily returns")
    start = _finite(starting_equity, "starting equity", positive=True)
    equity = peak = start
    worst = gains = losses = mean = m2 = 0.0
    positive = negative = zero = 0
    for count, raw in enumerate(values, start=1):
        ret = _finite(raw, "daily return")
        if ret <= -1.0:
            raise MetricsDataError("daily return implies bankrupt account")
        equity *= 1.0 + ret
        if not math.isfinite(equity) or equity <= 0:
            raise MetricsDataError("invalid compounded account equity")
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
        delta = ret - mean
        mean += delta / count
        m2 += delta * (ret - mean)
        if ret > 0:
            gains += ret
            positive += 1
        elif ret < 0:
            losses -= ret
            negative += 1
        else:
            zero += 1
    variance = max(0.0, m2 / (len(values) - 1)) if len(values) > 1 else 0.0
    std = math.sqrt(variance)
    sharpe = math.sqrt(ANNUAL_DAYS) * mean / std if std > 0 else None
    pf = gains / losses if losses > 0 else None
    multiple = equity / start
    return {
        "days": len(values), "starting_equity": start, "ending_equity": equity,
        "ending_multiple": multiple, "return_net": multiple - 1.0,
        "cagr": multiple ** (ANNUAL_DAYS / len(values)) - 1.0,
        "sharpe": sharpe, "sharpe_status": "defined" if sharpe is not None else "zero_sample_variance",
        "sample_std": std, "max_drawdown_daily": worst,
        "profit_factor_daily": pf,
        "profit_factor_status": "defined" if pf is not None else "no_negative_days",
        "positive_days": positive, "negative_days": negative, "zero_days": zero,
    }


def _summary(rows: Sequence[object], *, stressed_drawdown: float | None = None) -> dict[str, Any]:
    result = _returns_metrics([float(row.return_net) for row in rows],
                              starting_equity=float(rows[0].equity_open))
    ending = float(rows[-1].equity_next_open)
    if not math.isclose(result["ending_equity"], ending, rel_tol=1e-9, abs_tol=1e-6):
        raise MetricsDataError("daily returns do not compound to ledger final equity")
    result["ending_equity"] = ending
    result["ending_multiple"] = ending / result["starting_equity"]
    result["return_net"] = result["ending_multiple"] - 1.0
    result["cagr"] = result["ending_multiple"] ** (ANNUAL_DAYS / len(rows)) - 1.0
    result["exposure_days"] = sum(bool(row.exposed) for row in rows)
    result["cash_days"] = len(rows) - result["exposure_days"]
    if stressed_drawdown is not None:
        result["conservative_max_drawdown"] = stressed_drawdown
        result["conservative_drawdown_basis"] = "ordered_4h_peak_then_trough"
    return result


def _annual_walk_forward(rows: Sequence[object], annual_drawdowns: dict[int, float]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for year in YEARS:
        subset = [row for row in rows if row.day.year == year]
        if len(subset) != (date(year + 1, 1, 1) - date(year, 1, 1)).days:
            raise MetricsDataError(f"incomplete annual fold {year}")
        summary = _summary(subset, stressed_drawdown=annual_drawdowns[year])
        passed = (summary["return_net"] > 0 and
                  summary["profit_factor_daily"] is not None and
                  summary["profit_factor_daily"] > 1.0 and
                  summary["conservative_max_drawdown"] >= -0.20)
        folds.append({
            "year": year, "days": len(subset), "net_return": summary["return_net"],
            "profit_factor_daily": summary["profit_factor_daily"],
            "conservative_max_drawdown": summary["conservative_max_drawdown"],
            "passed": bool(passed),
        })
    return folds


def _monthly_stability(rows: Sequence[object]) -> dict[str, Any]:
    endpoints: dict[str, tuple[float, float]] = {}
    for row in rows:
        month = row.day.strftime("%Y-%m")
        if month not in endpoints:
            endpoints[month] = (float(row.equity_open), float(row.equity_next_open))
        else:
            endpoints[month] = (endpoints[month][0], float(row.equity_next_open))
    if len(endpoints) != 60:
        raise MetricsDataError("monthly grid does not contain 60 months")
    returns = {month: end / start - 1.0 for month, (start, end) in sorted(endpoints.items())}
    positive = sum(value > 0 for value in returns.values())
    worst_month = min(returns, key=returns.get)
    streak = longest = 0
    for value in returns.values():
        streak = streak + 1 if value < 0 else 0
        longest = max(longest, streak)
    ratio = positive / 60
    return {
        "complete_months": 60, "positive_months": positive,
        "positive_month_ratio": ratio, "worst_month": worst_month,
        "worst_month_return": returns[worst_month],
        "longest_consecutive_losing_months": longest,
        "returns": returns,
        "passed": ratio >= 0.70 and returns[worst_month] >= -0.10,
    }


def _btc_daily_closes(bars: object) -> dict[date, float]:
    if not isinstance(bars, pd.DataFrame) or not isinstance(bars.index, pd.DatetimeIndex):
        raise MetricsDataError("BTC trading bars must be a UTC 4h DataFrame")
    if bars.index.tz is None or str(bars.index.tz) != "UTC" or "close" not in bars:
        raise MetricsDataError("BTC trading bars lack UTC index or close column")
    expected = pd.date_range("2020-12-01", "2026-01-01", freq="4h",
                             inclusive="left", tz="UTC")
    selected = bars.loc[(bars.index >= expected[0]) & (bars.index <= expected[-1])]
    if not selected.index.equals(expected):
        raise MetricsDataError("BTC prior 30-day close source has a 4h gap or duplicate")
    values = pd.to_numeric(selected["close"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise MetricsDataError("BTC close contains an invalid value")
    return {stamp.date(): float(value) for stamp, value in zip(expected[5::6], values[5::6])}


def _regime_report(rows: Sequence[object], closes: dict[date, float]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[object]] = {name: [] for name in REGIMES}
    for row in rows:
        previous = row.day - timedelta(days=1)
        first = row.day - timedelta(days=31)
        if previous not in closes or first not in closes:
            raise MetricsDataError(f"missing prior 30 completed UTC BTC closes for {row.day}")
        ratio = closes[previous] / closes[first]
        label = "bull" if ratio > 1.10 else "bear" if ratio < 0.90 else "sideways"
        buckets[label].append(row)
    report: dict[str, dict[str, Any]] = {}
    for name, subset in buckets.items():
        if not subset:
            report[name] = {"days": 0, "return_net": None, "sharpe": None,
                            "max_drawdown_daily": None, "profit_factor_daily": None,
                            "exposure_days": 0, "cash_days": 0, "net_pnl_usdt": 0.0}
            continue
        metrics = _returns_metrics([float(row.return_net) for row in subset])
        active = sum(bool(row.exposed) for row in subset)
        report[name] = {
            "days": len(subset), "return_net": metrics["return_net"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_daily": metrics["max_drawdown_daily"],
            "profit_factor_daily": metrics["profit_factor_daily"],
            "exposure_days": active, "cash_days": len(subset) - active,
            "net_pnl_usdt": sum(float(row.equity_next_open) - float(row.equity_open)
                                 for row in subset),
        }
    return report


def _lower_percentile(values: np.ndarray, probability: float = 0.05) -> float | None:
    if not np.all(np.isfinite(values)):
        return None
    rank = int(math.floor(probability * (len(values) - 1)))
    return float(np.sort(values)[rank])


def block_monte_carlo(daily_returns: Sequence[float], *, paths: int = MC_PATHS,
                      block_days: int = MC_BLOCK_DAYS, seed: int = MC_SEED) -> dict[str, Any]:
    """21-day circular-block bootstrap; its drawdown is daily-close only."""
    if paths <= 0 or block_days <= 0 or not daily_returns:
        raise ValueError("invalid Monte Carlo dimensions")
    values = np.asarray(tuple(daily_returns), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= -1):
        raise MetricsDataError("invalid Monte Carlo daily returns")
    n = len(values)
    block_count = math.ceil(n / block_days)
    offsets = np.arange(block_days, dtype=np.int64)
    rng = np.random.default_rng(seed)
    cagrs = np.empty(paths, dtype=np.float64)
    sharpes = np.full(paths, np.nan, dtype=np.float64)
    drawdowns = np.empty(paths, dtype=np.float64)
    pfs = np.full(paths, np.nan, dtype=np.float64)
    joint_pass = 0
    for base in range(0, paths, 64):
        batch = min(64, paths - base)
        heads = rng.integers(0, n, size=(batch, block_count), dtype=np.int64)
        indices = ((heads[:, :, None] + offsets) % n).reshape(batch, -1)[:, :n]
        sample = values[indices]
        factors = 1.0 + sample
        ending = np.prod(factors, axis=1)
        cagr = np.power(ending, ANNUAL_DAYS / n) - 1.0
        mean = sample.mean(axis=1)
        std = sample.std(axis=1, ddof=1) if n > 1 else np.zeros(batch)
        std[np.all(sample == sample[:, :1], axis=1)] = 0.0
        sharpe = np.full(batch, np.nan)
        np.divide(math.sqrt(ANNUAL_DAYS) * mean, std, out=sharpe, where=std > 0)
        gains = np.maximum(sample, 0.0).sum(axis=1)
        losses = -np.minimum(sample, 0.0).sum(axis=1)
        pf = np.full(batch, np.nan)
        np.divide(gains, losses, out=pf, where=losses > 0)
        equity = np.cumprod(factors, axis=1)
        peaks = np.maximum.accumulate(np.concatenate((np.ones((batch, 1)), equity), axis=1),
                                      axis=1)
        drawdown = np.min(equity / peaks[:, 1:] - 1.0, axis=1)
        cagrs[base:base + batch] = cagr
        sharpes[base:base + batch] = sharpe
        drawdowns[base:base + batch] = drawdown
        pfs[base:base + batch] = pf
        joint_pass += int(np.count_nonzero((cagr >= 0.40) & (sharpe > 1.5)
                                          & (drawdown >= -0.20) & (pf > 1.3)))
    p05_cagr = _lower_percentile(cagrs)
    probability = joint_pass / paths
    defined_mc_metrics = bool(np.all(np.isfinite(sharpes)) and np.all(np.isfinite(pfs)))
    return {
        "paths": paths, "block_days": block_days, "seed": seed,
        "p05_cagr": p05_cagr, "p05_sharpe": _lower_percentile(sharpes),
        "p05_max_drawdown_daily": _lower_percentile(drawdowns),
        "p05_profit_factor_daily": _lower_percentile(pfs),
        "joint_numeric_gate_paths": joint_pass,
        "joint_numeric_gate_probability": probability,
        "passed": defined_mc_metrics and probability >= 0.05
            and p05_cagr is not None and p05_cagr > 0,
        "drawdown_basis": "daily_close_only_not_main_4h_stress",
    }


def _affected_windows(rows: Sequence[object]) -> dict[str, dict[str, Any]]:
    intervals = {
        "wikimedia_2021_2022_traffic_loss": (date(2021, 6, 4), date(2022, 1, 27)),
        "wikimedia_2025_source_change": (date(2025, 4, 1), date(2025, 12, 31)),
    }
    report: dict[str, dict[str, Any]] = {}
    for name, (start, end) in intervals.items():
        subset = [row for row in rows if start <= row.day <= end]
        if len(subset) != (end - start).days + 1:
            raise MetricsDataError(f"incomplete Wikimedia affected window {name}")
        report[name] = {
            "start": start.isoformat(), "end_inclusive": end.isoformat(),
            "days": len(subset),
            "net_return": float(subset[-1].equity_next_open) / float(subset[0].equity_open) - 1.0,
            "exposure_days": sum(bool(row.exposed) for row in subset),
            "excluded_from_analysis": False,
        }
    return report


def evaluate(ledger: object, benchmark_ledger: object, btc_trade_bars: object) -> dict[str, Any]:
    """Evaluate fixed 2021–2025 retrospective Gate; never promote OOS evidence."""
    rows = _checked_records(ledger, "strategy")
    metadata = _ledger_metadata(ledger, "strategy")
    strategy = _summary(rows, stressed_drawdown=metadata["conservative_max_drawdown"])
    benchmark = benchmark_metadata = None
    if benchmark_ledger is not None:
        benchmark_rows = _checked_records(benchmark_ledger, "benchmark")
        benchmark_metadata = _ledger_metadata(benchmark_ledger, "benchmark")
        benchmark = _summary(benchmark_rows,
                             stressed_drawdown=benchmark_metadata["conservative_max_drawdown"])
    folds = _annual_walk_forward(rows, metadata["conservative_max_drawdown_by_year"])
    fold_passes = sum(item["passed"] for item in folds)
    monthly = _monthly_stability(rows)
    regimes = _regime_report(rows, _btc_daily_closes(btc_trade_bars))
    holdout_rows = [row for row in rows if row.day.year >= 2024]
    holdout = _summary(holdout_rows)
    holdout["period"] = "2024-01-01/2025-12-31"
    holdout["independent_oos"] = False
    holdout["drawdown_basis"] = "daily_close_only; exact 4h path not supplied for two-year slice"
    mc = block_monte_carlo([float(row.return_net) for row in rows])
    checks = {
        "history_at_least_five_complete_years": len(rows) == 1826,
        "five_complete_annual_folds": len(folds) == 5,
        "cagr_at_least_40pct": strategy["cagr"] >= 0.40,
        "sharpe_above_1_5": strategy["sharpe"] is not None and strategy["sharpe"] > 1.5,
        "conservative_4h_drawdown_at_least_minus_20pct":
            strategy["conservative_max_drawdown"] >= -0.20,
        "daily_profit_factor_above_1_3": strategy["profit_factor_daily"] is not None
            and strategy["profit_factor_daily"] > 1.3,
        "annual_walk_forward_at_least_4_of_5": fold_passes >= 4,
        "monthly_stability": monthly["passed"],
        "all_three_regimes_present": all(regimes[name]["days"] > 0 for name in REGIMES),
        "monte_carlo": mc["passed"],
        "capacity": metadata["capacity_breaches"] == 0,
        "minimum_order_execution": metadata["execution_breaches"] == 0,
        "margin_proxy": metadata["margin_proxy_breaches"] == 0,
        "terminal_flat": metadata["terminal_flat"],
    }
    return {
        "strategy": strategy, "benchmark": benchmark,
        "strategy_execution": metadata, "benchmark_execution": benchmark_metadata,
        "annual_walk_forward": folds, "annual_walk_forward_passes": fold_passes,
        "historical_time_holdout_2024_2025": holdout,
        "monthly_stability": monthly, "regimes": regimes,
        "wikimedia_affected_windows": _affected_windows(rows),
        "monte_carlo": mc, "gate_checks": checks,
        "numerical_gate": all(checks.values()),
        "benchmark_path_complete": benchmark is not None,
        "independent_oos": False, "full_gate": False,
        "evidence_grade": "retrospective_hypothesis_holdout",
    }
