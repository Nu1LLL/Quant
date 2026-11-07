"""Frozen, fail-closed diagnostics for the preregistered weekend study."""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.weekend_reentry_sim import END, INITIAL, START, WeekendLedger


ANNUAL_DAYS = 365.25
YEARS = tuple(range(2021, 2027))
REGIMES = ("up_high_vol", "up_low_vol", "down_high_vol", "down_low_vol")
MC_PATHS = 5_000
MC_BLOCK = 21
MC_SEED = 20261108


class MetricsDataError(ValueError):
    """A ledger or market-state input violates the frozen analysis contract."""


def _finite(value: object, label: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricsDataError(f"invalid {label}") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise MetricsDataError(f"invalid {label}")
    return result


def _records(ledger: WeekendLedger, label: str) -> tuple:
    rows = ledger.records
    n = (END.date() - START.date()).days
    if len(rows) != n or not math.isclose(rows[0].equity_open, INITIAL, abs_tol=1e-7):
        raise MetricsDataError(f"{label}: incomplete or incorrect initial daily ledger")
    for i, row in enumerate(rows):
        if type(row.day) is not date or row.day != START.date() + timedelta(days=i):
            raise MetricsDataError(f"{label}: daily date gap or duplicate")
        opening = _finite(row.equity_open, "opening equity", True)
        ending = _finite(row.equity_next_open, "ending equity", True)
        high = _finite(row.conservative_high_equity, "high equity", True)
        low = _finite(row.conservative_low_equity, "low equity", True)
        if high < max(low, opening, ending) - 1e-6 or low > min(opening, ending) + 1e-6:
            raise MetricsDataError(f"{label}: inconsistent adverse 4h daily extrema")
        if not math.isclose(ending / opening - 1, _finite(row.return_net, "return"),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise MetricsDataError(f"{label}: inconsistent daily return")
        if i and not math.isclose(opening, rows[i - 1].equity_next_open,
                                  rel_tol=1e-10, abs_tol=1e-6):
            raise MetricsDataError(f"{label}: equity discontinuity")
    if not math.isclose(rows[-1].equity_next_open, ledger.final_equity,
                        rel_tol=1e-10, abs_tol=1e-6):
        raise MetricsDataError(f"{label}: final equity mismatch")
    if set(ledger.conservative_max_drawdown_by_year) != set(YEARS):
        raise MetricsDataError(f"{label}: missing annual 4h drawdowns")
    if len(ledger.capacity_schedule) != 300 or len(set(ledger.capacity_schedule)) != 300:
        # There are 300 Mondays from 2021-01-04 through 2026-09-28.
        raise MetricsDataError(f"{label}: incomplete weekly decision schedule")
    return rows


def _drawdown(rows: Sequence) -> float:
    peak = _finite(rows[0].equity_open, "drawdown opening", True)
    worst = 0.0
    for row in rows:
        peak = max(peak, row.conservative_high_equity)
        worst = min(worst, row.conservative_low_equity / peak - 1)
    return worst


def _summary(rows: Sequence, *, exact_dd: float | None = None) -> dict[str, Any]:
    if not rows:
        raise MetricsDataError("empty metric window")
    rets = np.array([row.return_net for row in rows], dtype=float)
    pnl = np.array([row.equity_next_open - row.equity_open for row in rows], dtype=float)
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    losses = -float(np.minimum(pnl, 0).sum())
    start, end = rows[0].equity_open, rows[-1].equity_next_open
    dd = _drawdown(rows)
    if exact_dd is not None:
        if not -1 <= exact_dd <= 0:
            raise MetricsDataError("invalid ordered 4h drawdown")
        # A daily high can occur after its daily low. Placing that high first
        # produces the conservative (larger) drawdown, never a favorable one.
        dd = min(dd, exact_dd)
    return {
        "days": len(rows), "initial_usdt": start, "final_usdt": end,
        "net_return": end / start - 1,
        "cagr": (end / start) ** (ANNUAL_DAYS / len(rows)) - 1,
        "sharpe": float(math.sqrt(ANNUAL_DAYS) * np.mean(rets) / std) if std > 0 else None,
        "profit_factor_daily_dollar": float(np.maximum(pnl, 0).sum() / losses)
        if losses > 0 else None,
        "conservative_4h_max_drawdown": dd,
        "exposure_days": sum(bool(row.exposed) for row in rows),
    }


def _monthly(rows: Sequence) -> dict[str, Any]:
    months: dict[str, list[float]] = {}
    for row in rows:
        key = row.day.strftime("%Y-%m")
        if key not in months:
            months[key] = [row.equity_open, row.equity_next_open]
        else:
            months[key][1] = row.equity_next_open
    if len(months) != 69:
        raise MetricsDataError("expected 69 full months")
    returns = {month: end / start - 1 for month, (start, end) in months.items()}
    positive = sum(value > 0 for value in returns.values())
    streak = longest = 0
    for value in returns.values():
        streak = streak + 1 if value <= 0 else 0
        longest = max(longest, streak)
    return {"returns": returns, "positive_months": positive,
            "positive_ratio": positive / 69,
            "longest_nonpositive_streak": longest,
            "passed": positive / 69 >= 0.60 and longest <= 3}


def _btc_closes(bars: pd.DataFrame) -> dict[date, float]:
    start = pd.Timestamp("2020-12-01", tz="UTC")
    full_grid = pd.date_range(start, END, freq="4h", tz="UTC")
    expected = full_grid[:-1]
    if (not isinstance(bars, pd.DataFrame) or not bars.index.equals(pd.date_range(
            start, END, freq="4h", tz="UTC")) or "close" not in bars):
        raise MetricsDataError("BTC trading-bar grid missing")
    closes = pd.to_numeric(bars.loc[expected, "close"], errors="coerce").to_numpy(float)
    if not np.isfinite(closes).all() or np.any(closes <= 0):
        raise MetricsDataError("invalid BTC prior close")
    return {stamp.date(): float(value) for stamp, value in zip(expected[5::6], closes[5::6], strict=True)}


def _regimes(rows: Sequence, bars: pd.DataFrame) -> dict[str, dict[str, Any]]:
    closes = _btc_closes(bars)
    buckets: dict[str, list] = {name: [] for name in REGIMES}
    unscored = 0
    for offset, row in enumerate(rows):
        if offset < 30:
            unscored += 1
            continue
        prior = row.day - timedelta(days=1)
        first = row.day - timedelta(days=31)
        if first not in closes:
            raise MetricsDataError("missing BTC regime warmup")
        prices = np.array([closes[first + timedelta(days=j)] for j in range(31)])
        returns = prices[1:] / prices[:-1] - 1
        up = prices[-1] / prices[0] - 1 > 0
        high = np.std(returns, ddof=1) * math.sqrt(365) >= 0.80
        label = ("up" if up else "down") + ("_high_vol" if high else "_low_vol")
        buckets[label].append(row)
    report = {}
    for label, subset in buckets.items():
        pnl = sum(row.equity_next_open - row.equity_open for row in subset)
        report[label] = {"days": len(subset), "net_pnl_usdt": pnl,
                         "passed": len(subset) >= 90 and pnl > 0}
    report["unscored_first_days"] = {"days": unscored}
    return report


def _mc(rows: Sequence) -> dict[str, Any]:
    returns = np.array([row.return_net for row in rows], dtype=float)
    highs = np.array([row.conservative_high_equity / row.equity_open for row in rows])
    lows = np.array([row.conservative_low_equity / row.equity_open for row in rows])
    n = len(rows)
    rng = np.random.default_rng(MC_SEED)
    outputs = {name: np.full(MC_PATHS, np.nan) for name in ("cagr", "sharpe", "drawdown", "pf")}
    block_count = math.ceil(n / MC_BLOCK)
    offsets = np.arange(MC_BLOCK)
    for base in range(0, MC_PATHS, 32):
        batch = min(32, MC_PATHS - base)
        heads = rng.integers(0, n, size=(batch, block_count), dtype=np.int64)
        idx = ((heads[:, :, None] + offsets) % n).reshape(batch, -1)[:, :n]
        sample = returns[idx]
        factors = 1 + sample
        equity = np.cumprod(factors, axis=1)
        opening = np.concatenate((np.ones((batch, 1)), equity[:, :-1]), axis=1)
        pnl = opening * sample
        gain = np.maximum(pnl, 0).sum(axis=1)
        loss = -np.minimum(pnl, 0).sum(axis=1)
        std = sample.std(axis=1, ddof=1)
        path_high = opening * highs[idx]
        path_low = opening * lows[idx]
        peak = np.maximum.accumulate(np.maximum(path_high, opening), axis=1)
        peak = np.maximum(1.0, peak)
        outputs["cagr"][base:base + batch] = equity[:, -1] ** (ANNUAL_DAYS / n) - 1
        outputs["drawdown"][base:base + batch] = np.min(path_low / peak - 1, axis=1)
        np.divide(math.sqrt(ANNUAL_DAYS) * sample.mean(axis=1), std,
                  out=outputs["sharpe"][base:base + batch], where=std > 0)
        np.divide(gain, loss, out=outputs["pf"][base:base + batch], where=loss > 0)
    valid = np.logical_and.reduce([np.isfinite(value) for value in outputs.values()])
    passes = valid & (outputs["cagr"] >= 0.4) & (outputs["sharpe"] > 1.5) & (
        outputs["drawdown"] >= -0.2) & (outputs["pf"] > 1.3)
    quantiles = {name: {str(q): float(np.quantile(value[np.isfinite(value)], q))
                        if np.isfinite(value).any() else None for q in (0.05, 0.5, 0.95)}
                 for name, value in outputs.items()}
    return {"paths": MC_PATHS, "block_days": MC_BLOCK, "seed": MC_SEED,
            "joint_passes": int(passes.sum()), "joint_pass_probability": float(passes.mean()),
            "quantiles": quantiles, "passed": float(passes.mean()) >= 0.80}


def evaluate(primary: WeekendLedger, benchmark: WeekendLedger | None,
             btc_bars: pd.DataFrame, stress: WeekendLedger | None,
             latency: WeekendLedger | None) -> dict[str, Any]:
    rows = _records(primary, "primary")
    for label, item in (("benchmark", benchmark), ("cost_stress", stress),
                        ("latency_stress", latency)):
        if item is not None:
            _records(item, label)
    full = _summary(rows, exact_dd=primary.conservative_max_drawdown)
    holdout_rows = tuple(row for row in rows if row.day >= date(2024, 1, 1))
    holdout = _summary(holdout_rows)
    annual = []
    for year in YEARS:
        subset = tuple(row for row in rows if row.day.year == year)
        expected = (date(year + 1, 1, 1) if year < 2026 else END.date()) - date(year, 1, 1)
        if len(subset) != expected.days:
            raise MetricsDataError(f"incomplete annual fold {year}")
        item = _summary(subset, exact_dd=primary.conservative_max_drawdown_by_year[year])
        item["year"] = year
        item["passed"] = (item["net_return"] > 0 and item["sharpe"] is not None
                          and item["sharpe"] > 0 and item["profit_factor_daily_dollar"] is not None
                          and item["profit_factor_daily_dollar"] > 1
                          and item["conservative_4h_max_drawdown"] >= -0.2)
        annual.append(item)
    monthly = _monthly(rows)
    regimes = _regimes(rows, btc_bars)
    mc = _mc(rows)
    def numeric(window: dict[str, Any]) -> bool:
        return (window["cagr"] >= 0.40 and window["sharpe"] is not None
                and window["sharpe"] > 1.5 and window["conservative_4h_max_drawdown"] >= -0.2
                and window["profit_factor_daily_dollar"] is not None
                and window["profit_factor_daily_dollar"] > 1.3)
    checks = {"full_window": numeric(full), "time_holdout": numeric(holdout),
              "all_annual_folds": all(item["passed"] for item in annual),
              "monthly_stability": monthly["passed"],
              "all_market_regimes": all(regimes[name]["passed"] for name in REGIMES),
              "monte_carlo": mc["passed"],
              "capacity": primary.capacity_breaches == 0,
              "execution": primary.execution_breaches == 0 and primary.terminal_flat,
              "stress_cost": stress is not None and stress.final_equity > INITIAL
              and stress.terminal_flat and stress.execution_breaches == 0,
              "stress_latency": latency is not None and latency.final_equity > INITIAL
              and latency.terminal_flat and latency.execution_breaches == 0}
    return {"full": full, "time_holdout_2024_2026": holdout,
            "annual_walk_forward": annual, "monthly": monthly,
            "market_regimes": regimes, "monte_carlo": mc,
            "benchmark": _summary(benchmark.records) if benchmark else None,
            "checks": checks, "numerical_gate": all(checks.values()),
            "independent_executable_oos": False, "all_gate_passed": False}
