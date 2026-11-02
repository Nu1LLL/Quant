"""Pure diagnostics for the preregistered BTC difficulty-recovery experiment.

The ledgers and bars supplied here have already been audited and simulated.
This module never reads a source, changes a trade, or selects a parameter.  A
daily record dated D covers [D 00:00, D+1 00:00), including any order at the
latter open.  The stressed high is deliberately processed before the stressed
low, the preregistered adverse peak/trough ordering.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


START_DAY = date(2018, 1, 1)
END_DAY_EXCLUSIVE = date(2026, 10, 1)
COMPLETE_YEARS = tuple(range(2018, 2026))
ANNUAL_DAYS = 365.25
MC_PATHS = 10_000
MC_BLOCK_DAYS = 21
MC_SEED = 20261102
REGIME_NAMES = ("bull", "bear", "sideways")


class MetricsDataError(ValueError):
    """A supplied ledger or BTC close series cannot support the fixed test."""


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MetricsDataError(f"invalid {label}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise MetricsDataError(f"invalid {label}")
    return number


def _checked_records(ledger: object, label: str) -> tuple[object, ...]:
    try:
        records = tuple(ledger.records)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise MetricsDataError(f"missing {label} daily records") from exc
    expected = (END_DAY_EXCLUSIVE - START_DAY).days
    if len(records) != expected:
        raise MetricsDataError(f"{label} daily grid has {len(records)} rather than {expected} days")
    previous_end: float | None = None
    for offset, row in enumerate(records):
        day = getattr(row, "day", None)
        if type(day) is not date or day != START_DAY + timedelta(days=offset):
            raise MetricsDataError(f"{label} daily date gap or duplicate at offset {offset}")
        opening = _finite(getattr(row, "equity_open", None), f"{label} opening equity", positive=True)
        ending = _finite(getattr(row, "equity_next_open", None), f"{label} ending equity", positive=True)
        ret = _finite(getattr(row, "return_net", None), f"{label} daily return")
        high = _finite(getattr(row, "conservative_high_equity", None),
                       f"{label} stressed high equity", positive=True)
        low = _finite(getattr(row, "conservative_low_equity", None),
                      f"{label} stressed low equity", positive=True)
        if (not math.isclose(ending / opening - 1.0, ret, rel_tol=1e-10, abs_tol=1e-12)
                or high < low):
            raise MetricsDataError(f"inconsistent {label} daily record {day}")
        if previous_end is not None and not math.isclose(
                opening, previous_end, rel_tol=1e-10, abs_tol=1e-7):
            raise MetricsDataError(f"discontinuous {label} equity at {day}")
        if type(getattr(row, "exposed", None)) is not bool:
            raise MetricsDataError(f"invalid {label} exposure at {day}")
        previous_end = ending
        if offset == 0 and not math.isclose(opening, 10_000.0, rel_tol=0.0, abs_tol=1e-7):
            raise MetricsDataError(f"{label} initial equity is not 10,000 USDT")
    stated_final = _finite(getattr(ledger, "final_equity", None),
                           f"{label} final equity", positive=True)
    if not math.isclose(stated_final, previous_end, rel_tol=1e-10, abs_tol=1e-7):
        raise MetricsDataError(f"{label} final equity disagrees with daily ledger")
    return records


def _returns_metrics(values: Sequence[float], *, starting_equity: float = 1.0) -> dict[str, Any]:
    """365.25-day CAGR, sample Sharpe, daily PF and close-to-close drawdown."""
    if not values:
        raise ValueError("daily returns are empty")
    start = _finite(starting_equity, "starting equity", positive=True)
    equity = start
    peak = start
    drawdown = 0.0
    mean = m2 = gains = losses = 0.0
    positive_days = negative_days = zero_days = 0
    for count, raw in enumerate(values, start=1):
        ret = _finite(raw, "daily return")
        if ret <= -1.0:
            raise MetricsDataError("daily return implies nonpositive account equity")
        equity *= 1.0 + ret
        if not math.isfinite(equity) or equity <= 0:
            raise MetricsDataError("invalid compounded account equity")
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
        delta = ret - mean
        mean += delta / count
        m2 += delta * (ret - mean)
        if ret > 0:
            gains += ret
            positive_days += 1
        elif ret < 0:
            losses -= ret
            negative_days += 1
        else:
            zero_days += 1
    n = len(values)
    variance = max(0.0, m2 / (n - 1)) if n > 1 else 0.0
    std = math.sqrt(variance)
    sharpe = math.sqrt(ANNUAL_DAYS) * mean / std if std > 0 else None
    pf = gains / losses if losses > 0 else None
    multiple = equity / start
    return {
        "days": n,
        "ending_multiple": multiple,
        "return_net": multiple - 1.0,
        "cagr": multiple ** (ANNUAL_DAYS / n) - 1.0,
        "sharpe": sharpe,
        "sharpe_status": "defined" if sharpe is not None else "zero_sample_variance",
        "sample_std": std,
        "max_drawdown_daily": drawdown,
        "profit_factor_daily": pf,
        "profit_factor_status": "defined" if pf is not None else "no_negative_days",
        "positive_days": positive_days,
        "negative_days": negative_days,
        "zero_days": zero_days,
    }


def _stressed_drawdown(rows: Sequence[object]) -> float:
    """Reset the peak at the first row's open; assume each high precedes low."""
    if not rows:
        raise ValueError("stressed drawdown needs daily rows")
    peak = float(rows[0].equity_open)
    worst = 0.0
    for row in rows:
        opening = float(row.equity_open)
        high = float(row.conservative_high_equity)
        low = float(row.conservative_low_equity)
        ending = float(row.equity_next_open)
        peak = max(peak, opening, high)
        worst = min(worst, low / peak - 1.0, ending / peak - 1.0)
        peak = max(peak, ending)
    return worst


def _ledger_summary(rows: Sequence[object]) -> dict[str, Any]:
    opening = float(rows[0].equity_open)
    ending = float(rows[-1].equity_next_open)
    result = _returns_metrics([float(row.return_net) for row in rows],
                              starting_equity=opening)
    multiple = ending / opening
    result["starting_equity"] = opening
    result["ending_equity"] = ending
    result["ending_multiple"] = multiple
    result["return_net"] = multiple - 1.0
    result["cagr"] = multiple ** (ANNUAL_DAYS / len(rows)) - 1.0
    result["conservative_max_drawdown"] = _stressed_drawdown(rows)
    result["exposure_days"] = sum(bool(row.exposed) for row in rows)
    result["cash_days"] = len(rows) - result["exposure_days"]
    return result


def _annual_walk_forward(rows: Sequence[object]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for year in COMPLETE_YEARS:
        subset = [row for row in rows if row.day.year == year]
        expected = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(subset) != expected:
            raise MetricsDataError(f"incomplete annual fold {year}")
        summary = _ledger_summary(subset)
        passed = (summary["return_net"] > 0 and
                  summary["profit_factor_daily"] is not None and
                  summary["profit_factor_daily"] > 1.0 and
                  summary["conservative_max_drawdown"] >= -0.20)
        folds.append({"year": year, "net_return": summary["return_net"],
                      "profit_factor_daily": summary["profit_factor_daily"],
                      "conservative_max_drawdown": summary["conservative_max_drawdown"],
                      "passed": bool(passed)})
    partial_rows = [row for row in rows if row.day.year == 2026]
    if len(partial_rows) != (END_DAY_EXCLUSIVE - date(2026, 1, 1)).days:
        raise MetricsDataError("incomplete partial 2026 period")
    partial = _ledger_summary(partial_rows)
    partial["period"] = "2026-01-01/2026-09-30"
    return folds, partial


def _monthly_stability(rows: Sequence[object]) -> dict[str, Any]:
    months: dict[str, tuple[float, float]] = {}
    for row in rows:
        key = row.day.strftime("%Y-%m")
        if key not in months:
            months[key] = (float(row.equity_open), float(row.equity_next_open))
        else:
            months[key] = (months[key][0], float(row.equity_next_open))
    monthly_returns = {key: ending / opening - 1.0
                       for key, (opening, ending) in sorted(months.items())}
    positive = sum(ret > 0 for ret in monthly_returns.values())
    worst_month = min(monthly_returns, key=monthly_returns.get)
    streak = longest = 0
    for ret in monthly_returns.values():
        streak = streak + 1 if ret < 0 else 0
        longest = max(longest, streak)
    ratio = positive / len(monthly_returns)
    return {
        "complete_months": len(monthly_returns),
        "positive_months": positive,
        "positive_month_ratio": ratio,
        "worst_month": worst_month,
        "worst_month_return": monthly_returns[worst_month],
        "longest_consecutive_losing_months": longest,
        "returns": monthly_returns,
        "passed": ratio >= 0.70 and monthly_returns[worst_month] >= -0.10,
    }


def _btc_closes(bars: object) -> dict[date, float]:
    source: Iterable[object] = getattr(bars, "bars", bars)  # type: ignore[assignment]
    closes: dict[date, float] = {}
    try:
        iterator = iter(source)
    except TypeError as exc:
        raise MetricsDataError("BTC bars are not iterable") from exc
    for bar in iterator:
        stamp = getattr(bar, "open_time", None)
        if (not isinstance(stamp, datetime) or stamp.tzinfo is None or
                stamp.utcoffset() != timedelta(0)):
            raise MetricsDataError("BTC bar lacks UTC open time")
        day = stamp.date()
        if day in closes:
            raise MetricsDataError(f"duplicate BTC close date {day}")
        closes[day] = _finite(getattr(bar, "close", None), "BTC close", positive=True)
    return closes


def _regimes(rows: Sequence[object], btc_closes: Mapping[date, float]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[object]] = {name: [] for name in REGIME_NAMES}
    for row in rows:
        day = row.day
        previous = day - timedelta(days=1)
        first = day - timedelta(days=31)
        if previous not in btc_closes or first not in btc_closes:
            raise MetricsDataError(f"missing prior 30-day BTC close window for {day}")
        ratio = btc_closes[previous] / btc_closes[first]
        label = "bull" if ratio > 1.10 else "bear" if ratio < 0.90 else "sideways"
        buckets[label].append(row)
    out: dict[str, dict[str, Any]] = {}
    for name, subset in buckets.items():
        if not subset:
            out[name] = {"days": 0, "exposure_days": 0, "cash_days": 0,
                         "return_net": None, "sharpe": None,
                         "max_drawdown_daily": None, "profit_factor_daily": None}
            continue
        metrics = _returns_metrics([float(row.return_net) for row in subset])
        active = sum(bool(row.exposed) for row in subset)
        out[name] = {
            "days": len(subset), "exposure_days": active,
            "cash_days": len(subset) - active,
            "return_net": metrics["return_net"],
            "sharpe": metrics["sharpe"],
            "max_drawdown_daily": metrics["max_drawdown_daily"],
            "profit_factor_daily": metrics["profit_factor_daily"],
        }
    return out


def _lower_percentile(values: np.ndarray, probability: float = 0.05) -> float | None:
    rank = int(math.floor(probability * (len(values) - 1)))
    value = float(np.sort(values)[rank])
    return value if math.isfinite(value) else None


def block_monte_carlo(
    daily_returns: Sequence[float], *, paths: int = MC_PATHS,
    block_days: int = MC_BLOCK_DAYS, seed: int = MC_SEED,
) -> dict[str, Any]:
    """Fixed-length circular-block bootstrap; daily close drawdown only.

    The public evaluator always calls this with the frozen 10,000/21/20261102
    specification.  The optional arguments allow small deterministic unit tests.
    The 5th percentile is the lower order statistic (zero-based rank floor of
    0.05 * (paths - 1)), avoiding interpolation across simulated paths.
    """
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
    sharpes = np.empty(paths, dtype=np.float64)
    drawdowns = np.empty(paths, dtype=np.float64)
    pfs = np.empty(paths, dtype=np.float64)
    joint_pass = 0
    batch_size = 64
    for base in range(0, paths, batch_size):
        batch = min(batch_size, paths - base)
        heads = rng.integers(0, n, size=(batch, block_count), dtype=np.int64)
        indices = ((heads[:, :, None] + offsets) % n).reshape(batch, -1)[:, :n]
        sample = values[indices]
        factors = 1.0 + sample
        ending = np.prod(factors, axis=1)
        cagr = np.power(ending, ANNUAL_DAYS / n) - 1.0
        mean = sample.mean(axis=1)
        std = sample.std(axis=1, ddof=1) if n > 1 else np.zeros(batch)
        sharpe = np.full(batch, -np.inf)
        np.divide(math.sqrt(ANNUAL_DAYS) * mean, std, out=sharpe, where=std > 0)
        gains = np.maximum(sample, 0.0).sum(axis=1)
        losses = -np.minimum(sample, 0.0).sum(axis=1)
        pf = np.full(batch, -np.inf)
        np.divide(gains, losses, out=pf, where=losses > 0)
        equity = np.cumprod(factors, axis=1)
        peaks = np.maximum.accumulate(
            np.concatenate((np.ones((batch, 1)), equity), axis=1), axis=1)
        drawdown = np.min(equity / peaks[:, 1:] - 1.0, axis=1)
        cagrs[base:base + batch] = cagr
        sharpes[base:base + batch] = sharpe
        drawdowns[base:base + batch] = drawdown
        pfs[base:base + batch] = pf
        joint_pass += int(np.count_nonzero((cagr >= 0.40) & (sharpe > 1.5)
                                          & (drawdown >= -0.20) & (pf > 1.3)))
    p05_cagr = _lower_percentile(cagrs)
    assert p05_cagr is not None  # Valid finite return factors imply finite CAGR.
    probability = joint_pass / paths
    return {
        "paths": paths,
        "block_days": block_days,
        "seed": seed,
        "p05_cagr": p05_cagr,
        "p05_sharpe": _lower_percentile(sharpes),
        "p05_max_drawdown_daily": _lower_percentile(drawdowns),
        "p05_profit_factor_daily": _lower_percentile(pfs),
        "joint_numeric_gate_paths": joint_pass,
        "joint_numeric_gate_probability": probability,
        "passed": probability >= 0.05 and p05_cagr > 0,
        "drawdown_basis": "daily_close_only",
    }


def evaluate_diagnostics(
    strategy_ledger: object, bars: object, benchmark_ledger: object,
) -> dict[str, Any]:
    """Evaluate the exact retrospective calendar and fixed numeric Gates.

    ``bars`` may be the audited BarsResult or its ordered ``bars`` tuple.  The
    benchmark ledger must already implement the preregistered same-cost spot
    buy-and-hold account; this pure function does not resimulate its fills.
    Full Gate is false by construction because registration occurred in 2026.
    """
    rows = _checked_records(strategy_ledger, "strategy")
    benchmark_rows = _checked_records(benchmark_ledger, "benchmark")
    strategy = _ledger_summary(rows)
    benchmark = _ledger_summary(benchmark_rows)
    stated_drawdown = _finite(getattr(strategy_ledger, "conservative_max_drawdown", None),
                              "strategy stated drawdown")
    if not math.isclose(stated_drawdown, strategy["conservative_max_drawdown"],
                        rel_tol=1e-10, abs_tol=1e-10):
        raise MetricsDataError("strategy conservative drawdown disagrees with daily stresses")
    benchmark_stated_drawdown = _finite(
        getattr(benchmark_ledger, "conservative_max_drawdown", None),
        "benchmark stated drawdown")
    if not math.isclose(benchmark_stated_drawdown, benchmark["conservative_max_drawdown"],
                        rel_tol=1e-10, abs_tol=1e-10):
        raise MetricsDataError("benchmark conservative drawdown disagrees with daily stresses")
    folds, partial = _annual_walk_forward(rows)
    months = _monthly_stability(rows)
    regimes = _regimes(rows, _btc_closes(bars))
    mc = block_monte_carlo([float(row.return_net) for row in rows])
    fold_passes = sum(item["passed"] for item in folds)
    checks = {
        "history_at_least_five_years": len(rows) >= 5 * 365,
        "eight_complete_annual_folds": len(folds) == 8,
        "cagr_at_least_40pct": strategy["cagr"] >= 0.40,
        "sharpe_above_1_5": strategy["sharpe"] is not None and strategy["sharpe"] > 1.5,
        "conservative_drawdown_at_least_minus_20pct": strategy["conservative_max_drawdown"] >= -0.20,
        "daily_profit_factor_above_1_3": (strategy["profit_factor_daily"] is not None
                                           and strategy["profit_factor_daily"] > 1.3),
        "annual_walk_forward_at_least_7_of_8": fold_passes >= 7,
        "monthly_stability": months["passed"],
        "all_three_regimes_present": all(regimes[name]["days"] > 0 for name in REGIME_NAMES),
        "monte_carlo": mc["passed"],
        "capacity": bool(getattr(strategy_ledger, "capacity_pass", False)),
    }
    numerical_gate = all(checks.values())
    return {
        "strategy": strategy,
        "benchmark": benchmark,
        "annual_walk_forward": folds,
        "annual_walk_forward_passes": fold_passes,
        "partial_2026": partial,
        "monthly_stability": months,
        "regimes": regimes,
        "monte_carlo": mc,
        "gate_checks": checks,
        "numerical_gate": numerical_gate,
        "independent_oos": False,
        "full_gate": False,
        "evidence_grade": "retrospective_hypothesis_holdout",
    }
