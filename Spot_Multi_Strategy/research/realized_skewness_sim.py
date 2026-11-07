"""Frozen daily realized-skewness cross-sectional signal and account Gate."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import hashlib
import math

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .weekday_seasonality_data import SYMBOLS, WeekdayTape
from .weekday_seasonality_sim import (
    BASE_DATE, END, END_DAY, FOLDS, INITIAL_EQUITY, N_DAYS, SCENARIOS,
    START, START_DAY, WeekdayStrategyFailure, _Account, _daily_factors,
    _regimes,
)


class SkewnessStrategyFailure(ValueError):
    pass


def _scores(tape: WeekdayTape, day_index: int) -> np.ndarray:
    """Read exactly the preceding UTC day's 288 completed five-minute returns."""
    first = (day_index - 1) * 288
    last = day_index * 288
    if first < 1 or last > len(tape.close):
        raise SkewnessStrategyFailure("preceding 288-bar close window incomplete")
    returns = np.log(tape.close[first:last] / tape.close[first - 1:last - 1])
    if returns.shape != (288, len(SYMBOLS)) or not np.all(np.isfinite(returns)):
        raise SkewnessStrategyFailure("preceding five-minute returns invalid")
    sumsq = np.sum(returns * returns, axis=0)
    if np.any(sumsq <= 0) or not np.all(np.isfinite(sumsq)):
        raise SkewnessStrategyFailure("zero/nonfinite realized-variance denominator")
    realized_skew = np.sqrt(288.0) * np.sum(returns ** 3, axis=0) / (sumsq ** 1.5)
    if not np.all(np.isfinite(realized_skew)):
        raise SkewnessStrategyFailure("nonfinite realized-skewness score")
    return realized_skew


def _ranking(tape: WeekdayTape, day_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    scores = _scores(tape, day_index)
    ranked = sorted(range(len(SYMBOLS)), key=lambda col: (float(scores[col]), SYMBOLS[col]))
    return tuple(ranked[:2]), tuple(ranked[-2:])


def run_simulation(tape: WeekdayTape) -> tuple[dict[str, _Account], dict[int, str], str]:
    if (len(tape.opening) != END_DAY * 288
            or any(array.shape != tape.opening.shape for array in (
                tape.high, tape.low, tape.close, tape.quote_volume))
            or tape.opening.shape[1] != len(SYMBOLS)):
        raise SkewnessStrategyFailure("tape shape or registered end changed")
    daily_prices = tape.opening[::288, :]
    daily_returns = np.log(daily_prices[1:] / daily_prices[:-1])
    if not np.all(np.isfinite(daily_returns)):
        raise SkewnessStrategyFailure("nonfinite daily source return")
    regimes = _regimes(daily_prices, daily_returns)
    accounts = {name: _Account(name, *settings) for name, settings in SCENARIOS.items()}
    schedule_hash = hashlib.sha256()
    for day_index in range(START_DAY, END_DAY):
        day = BASE_DATE + timedelta(days=day_index)
        if day == END - timedelta(days=1):
            for account in accounts.values():
                account.flat_day(day)
            schedule_hash.update(f"{day.isoformat()}:terminal_flat\n".encode())
            continue
        longs, shorts = _ranking(tape, day_index)
        selected = longs + shorts
        baseline_notional = accounts["base_1x"].equity * .25
        capacity = all(float(np.median(tape.quote_volume[
            (day_index - 1) * 288:day_index * 288, coin])) >= 50 * baseline_notional
                       for coin in selected)
        if capacity:
            for account in accounts.values():
                account.execute(tape, day_index, longs, shorts, regimes[day_index])
            schedule_hash.update((f"{day.isoformat()}:"
                                  + ",".join(SYMBOLS[i] for i in selected) + "\n").encode())
        else:
            for account in accounts.values():
                account.flat_day(day)
            schedule_hash.update(f"{day.isoformat()}:capacity_cash\n".encode())
    if any(len(account.records) != N_DAYS for account in accounts.values()):
        raise SkewnessStrategyFailure("daily record grid incomplete")
    return accounts, regimes, schedule_hash.hexdigest()


def evaluate(tape: WeekdayTape) -> dict:
    accounts, states, schedule_hash = run_simulation(tape)
    factors = {name: _daily_factors(account.records) for name, account in accounts.items()}
    metrics = {name: _metrics_from_factors(factors[name]) for name in SCENARIOS}
    for name, account in accounts.items():
        if (not math.isclose(metrics[name]["ending_equity"], account.equity,
                             rel_tol=1e-8, abs_tol=1e-5)
                or not math.isclose(metrics[name]["max_drawdown"], account.max_drawdown,
                                    rel_tol=1e-8, abs_tol=1e-8)):
            raise SkewnessStrategyFailure(f"{name}: compressed path mismatch")
    baseline = accounts["base_1x"]
    base_factors = factors["base_1x"]
    folds = {}
    for index, (first, last) in enumerate(FOLDS, start=1):
        a, b = (first - START).days, (last - START).days
        fold = _metrics_from_factors(base_factors[a:b])
        count = sum(first <= trade.entry_day < last for trade in baseline.trades)
        folds[f"year_{index}"] = {
            "start": str(first), "end_exclusive": str(last),
            "days": b - a, "trades": count,
            "net_return": fold["ending_equity"] / INITIAL_EQUITY - 1,
            "daily_sharpe": fold["sharpe"],
            "pass": count >= 1 and fold["ending_equity"] > INITIAL_EQUITY
            and fold["sharpe"] is not None and fold["sharpe"] > 0,
        }
    tail = _metrics_from_factors(base_factors[(FOLDS[-1][1] - START).days:])
    terminal = {"days": 133, "net_return": tail["ending_equity"] / INITIAL_EQUITY - 1,
                "pass": tail["ending_equity"] > INITIAL_EQUITY}

    by_month = defaultdict(list)
    for record in baseline.records:
        by_month[record.date.strftime("%Y-%m")].append(record)
    monthly = {month: group[-1].equity_end / group[0].equity_start - 1
               for month, group in sorted(by_month.items())}
    if len(monthly) != 65:
        raise SkewnessStrategyFailure("monthly grid changed")
    positive_months = sum(value > 0 for value in monthly.values())
    losing = longest_losing = 0
    for value in monthly.values():
        losing = losing + 1 if value < 0 else 0
        longest_losing = max(longest_losing, losing)
    monthly_summary = {
        "positive": positive_months,
        "negative": sum(value < 0 for value in monthly.values()),
        "flat": sum(value == 0 for value in monthly.values()),
        "longest_losing_streak": longest_losing,
        "worst_month": min(monthly, key=monthly.get),
        "worst_month_return": min(monthly.values()),
    }
    regimes = {name: {"days": 0, "trades": 0, "net_pnl": 0.0}
               for name in ("bull_high", "bull_low", "bear_high", "bear_low")}
    for state in states.values():
        regimes[state]["days"] += 1
    for trade in baseline.trades:
        regimes[trade.regime]["trades"] += 1
        regimes[trade.regime]["net_pnl"] += trade.net_pnl
    mc = monte_carlo_compressed(base_factors, paths=5000, block_days=21,
                                seed=20261113)
    primary = metrics["base_1x"]
    gate = {
        "primary_metrics": (primary["cagr"] >= .40
                            and primary["sharpe"] is not None and primary["sharpe"] > 1.5
                            and primary["max_drawdown"] <= .20
                            and primary["profit_factor"] is not None
                            and primary["profit_factor"] > 1.3),
        "minimum_trade_days": len(baseline.trades) >= 365,
        "five_annual_walk_forward_folds": all(item["pass"] for item in folds.values()),
        "terminal_133_days": terminal["pass"],
        "monthly_stability": positive_months >= 39 and longest_losing <= 3,
        "four_market_states": all(item["days"] >= 90 and item["net_pnl"] > 0
                                  for item in regimes.values()),
        "monte_carlo": mc["joint_pass_fraction"] >= .80,
        "doubled_slippage_profitable": metrics["double_slippage_1x"]["ending_equity"] > INITIAL_EQUITY,
        "extra_bar_latency_profitable": metrics["extra_bar_latency_1x"]["ending_equity"] > INITIAL_EQUITY,
        "no_funding_credit_profitable": metrics["no_funding_credit_1x"]["ending_equity"] > INITIAL_EQUITY,
        "independent_executable_oos": False,
    }
    gate["all_numeric"] = all(value for key, value in gate.items()
                              if key != "independent_executable_oos")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_oos"]
    pnl = {}
    for name, account in accounts.items():
        price = math.fsum(trade.gross_price_pnl for trade in account.trades)
        fees = math.fsum(trade.fee_pnl for trade in account.trades)
        funding = math.fsum(trade.funding_pnl for trade in account.trades)
        net = account.equity - INITIAL_EQUITY
        if not math.isclose(net, price + fees + funding, rel_tol=1e-8, abs_tol=1e-5):
            raise SkewnessStrategyFailure(f"{name}: trade PnL does not reconcile")
        pnl[name] = {"net": net, "executed_price_pnl": price,
                     "taker_fee_pnl": fees, "funding_pnl": funding}
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_band": .40 <= primary["cagr"] <= .60,
        "annual_folds": folds, "terminal_holdout": terminal,
        "monthly_returns": monthly, "monthly_summary": monthly_summary,
        "market_states": regimes, "monte_carlo": mc,
        "traded_days": len(baseline.trades),
        "cash_days": N_DAYS - len(baseline.trades),
        "schedule_sha256": schedule_hash,
        "actual_fill_participation_above_2pct": sum(
            trade.high_participation_fills for trade in baseline.trades),
        "actual_fill_zero_quote_volume": sum(
            trade.zero_volume_fills for trade in baseline.trades),
        "pnl_accounting": pnl,
        "gate": gate,
        "evidence_grade": "post-publication historical temporal OOS; no archived executable bid/ask or prospective fills",
    }
