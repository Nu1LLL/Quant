"""Preregistered weekly conditional-downside-beta cross-sectional replay."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import hashlib
import math

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .btc_eth_pairs_sim import DayRecord
from .weekday_seasonality_data import SYMBOLS, WeekdayTape
from .weekday_seasonality_sim import BASE_DATE, END_DAY, INITIAL_EQUITY, SCENARIOS, _regimes
from .weekly_illiquidity_sim import _WeeklyAccount


START = date(2021, 7, 9)
END = date(2026, 10, 1)
START_DAY = (START - BASE_DATE).days
N_DAYS = (END - START).days
WEEKS = 272
LOOKBACK_WEEKS = 26
FOLDS = tuple((date(2021 + i, 7, 9), date(2022 + i, 7, 9)) for i in range(5))


class DownsideBetaFailure(ValueError):
    pass


def _weekly_returns(tape: WeekdayTape, day_index: int) -> np.ndarray:
    """26 weekly log returns ending at the last *completed* Thursday bar."""
    boundaries = (day_index - 7 * np.arange(LOOKBACK_WEEKS + 1)) * 288 - 1
    if boundaries[-1] < 0 or boundaries[0] >= len(tape.close):
        raise DownsideBetaFailure("26-week completed-bar history unavailable")
    prices = tape.close[boundaries, :]
    if prices.shape != (LOOKBACK_WEEKS + 1, len(SYMBOLS)) or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise DownsideBetaFailure("invalid completed weekly close")
    returns = np.log(prices[:-1] / prices[1:])
    if returns.shape != (LOOKBACK_WEEKS, len(SYMBOLS)) or not np.all(np.isfinite(returns)):
        raise DownsideBetaFailure("nonfinite 26-week returns")
    return returns


def _rank(tape: WeekdayTape, day_index: int
          ) -> tuple[tuple[int, int], tuple[int, int]] | None:
    weekly = _weekly_returns(tape, day_index)
    market = weekly[:, SYMBOLS.index("BTCUSDT")]
    if np.count_nonzero(market < 0) < 5 or np.count_nonzero(market > 0) < 5:
        return None
    design = np.column_stack((np.ones(LOOKBACK_WEEKS), market,
                              market * (market > 0)))
    if np.linalg.matrix_rank(design) != 3:
        raise DownsideBetaFailure("downside-beta OLS design rank deficient")
    coefficients = np.linalg.lstsq(design, weekly, rcond=None)[0]
    downside = coefficients[1, :]
    if downside.shape != (len(SYMBOLS),) or not np.all(np.isfinite(downside)):
        raise DownsideBetaFailure("nonfinite downside beta")
    order = sorted(range(len(SYMBOLS)), key=lambda col: (float(downside[col]), SYMBOLS[col]))
    return tuple(order[-2:]), tuple(order[:2])


def _daily_factors(records) -> np.ndarray:
    if len(records) != N_DAYS:
        raise DownsideBetaFailure("formal daily record count changed")
    factors = np.empty((N_DAYS, 4), dtype=np.float64)
    prior_equity = INITIAL_EQUITY
    for j, row in enumerate(records):
        if (row.date != START + timedelta(days=j)
                or not math.isclose(row.equity_start, prior_equity,
                                    rel_tol=1e-10, abs_tol=1e-6)):
            raise DownsideBetaFailure("daily account path discontinuity")
        factors[j] = (row.equity_end / row.equity_start,
                      row.max_regular_factor, row.min_adverse_factor,
                      row.max_intraday_drawdown)
        prior_equity = row.equity_end
    return factors


def run_simulation(tape: WeekdayTape):
    if (len(tape.opening) != END_DAY * 288
            or any(array.shape != tape.opening.shape for array in
                   (tape.high, tape.low, tape.close, tape.quote_volume))
            or tape.opening.shape[1] != len(SYMBOLS)):
        raise DownsideBetaFailure("tape shape or registered source end changed")
    daily_prices = tape.opening[::288, :]
    daily_returns = np.log(daily_prices[1:] / daily_prices[:-1])
    if not np.all(np.isfinite(daily_returns)):
        raise DownsideBetaFailure("nonfinite daily source return")
    states = _regimes(daily_prices, daily_returns)
    accounts = {name: _WeeklyAccount(name, *settings) for name, settings in SCENARIOS.items()}
    schedule_hash = hashlib.sha256()
    signal_cash_weeks = capacity_cash_weeks = 0
    for week in range(WEEKS):
        day_index = START_DAY + 7 * week
        day = BASE_DATE + timedelta(days=day_index)
        selection = _rank(tape, day_index)
        if selection is None:
            signal_cash_weeks += 1
            for account in accounts.values():
                account.flat_week(day_index)
            schedule_hash.update(f"{day.isoformat()}:insufficient_market_down_weeks_cash\n".encode())
            continue
        longs, shorts = selection
        selected = longs + shorts
        base_notional = accounts["base_1x"].equity * .25
        capacity = all(float(np.median(tape.quote_volume[
            (day_index - 1) * 288:day_index * 288, col])) >= 50 * base_notional
                       for col in selected)
        if not capacity:
            capacity_cash_weeks += 1
            for account in accounts.values():
                account.flat_week(day_index)
            schedule_hash.update(f"{day.isoformat()}:capacity_cash\n".encode())
            continue
        for account in accounts.values():
            account.trade_week(tape, day_index, longs, shorts)
        schedule_hash.update((f"{day.isoformat()}:"
                              + ",".join(SYMBOLS[i] for i in selected) + "\n").encode())
    for offset in range(N_DAYS - WEEKS * 7):
        day = START + timedelta(days=WEEKS * 7 + offset)
        for account in accounts.values():
            account.records.append(DayRecord(day, account.equity, account.equity,
                                             1.0, 1.0, 0.0))
        schedule_hash.update(f"{day.isoformat()}:terminal_cash\n".encode())
    if any(len(account.records) != N_DAYS for account in accounts.values()):
        raise DownsideBetaFailure("daily grid incomplete")
    return accounts, states, schedule_hash.hexdigest(), signal_cash_weeks, capacity_cash_weeks


def evaluate(tape: WeekdayTape) -> dict:
    accounts, states, schedule_hash, signal_cash_weeks, capacity_cash_weeks = run_simulation(tape)
    factors = {name: _daily_factors(account.records) for name, account in accounts.items()}
    metrics = {name: _metrics_from_factors(factors[name]) for name in SCENARIOS}
    for name, account in accounts.items():
        if (not math.isclose(metrics[name]["ending_equity"], account.equity,
                             rel_tol=1e-8, abs_tol=1e-5)
                or not math.isclose(metrics[name]["max_drawdown"], account.max_drawdown,
                                    rel_tol=1e-8, abs_tol=1e-8)):
            raise DownsideBetaFailure(f"{name}: compressed path mismatch")
    baseline = accounts["base_1x"]
    base_factors = factors["base_1x"]
    folds = {}
    for index, (first, last) in enumerate(FOLDS, start=1):
        a, b = (first - START).days, (last - START).days
        fold = _metrics_from_factors(base_factors[a:b])
        count = sum(first <= row.date < last and row.equity_end != row.equity_start
                    for row in baseline.records)
        folds[f"year_{index}"] = {
            "start": str(first), "end_exclusive": str(last), "days": b - a,
            "traded_days": count, "net_return": fold["ending_equity"] / INITIAL_EQUITY - 1,
            "daily_sharpe": fold["sharpe"],
            "pass": count >= 1 and fold["ending_equity"] > INITIAL_EQUITY
            and fold["sharpe"] is not None and fold["sharpe"] > 0,
        }
    tail = _metrics_from_factors(base_factors[(FOLDS[-1][1] - START).days:])
    terminal = {"days": 84, "net_return": tail["ending_equity"] / INITIAL_EQUITY - 1,
                "pass": tail["ending_equity"] > INITIAL_EQUITY}
    by_month = defaultdict(list)
    for row in baseline.records:
        by_month[row.date.strftime("%Y-%m")].append(row)
    monthly = {month: group[-1].equity_end / group[0].equity_start - 1
               for month, group in sorted(by_month.items())}
    if len(monthly) != 63:
        raise DownsideBetaFailure("monthly grid changed")
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
    regimes = {name: {"days": 0, "traded_days": 0, "net_pnl": 0.0}
               for name in ("bull_high", "bull_low", "bear_high", "bear_low")}
    for j, row in enumerate(baseline.records):
        state = states[START_DAY + j]
        pnl = row.equity_end - row.equity_start
        regimes[state]["days"] += 1
        regimes[state]["traded_days"] += pnl != 0
        regimes[state]["net_pnl"] += pnl
    mc = monte_carlo_compressed(base_factors, paths=5000, block_days=21,
                                seed=20261115)
    primary = metrics["base_1x"]
    gate = {
        "primary_metrics": (primary["cagr"] >= .40
                            and primary["sharpe"] is not None and primary["sharpe"] > 1.5
                            and primary["max_drawdown"] <= .20
                            and primary["profit_factor"] is not None
                            and primary["profit_factor"] > 1.3),
        "minimum_trade_weeks": len(baseline.trades) >= 150,
        "five_annual_walk_forward_folds": all(item["pass"] for item in folds.values()),
        "terminal_84_days": terminal["pass"],
        "monthly_stability": positive_months >= 38 and longest_losing <= 3,
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
        price = math.fsum(trade.executed_price_pnl for trade in account.trades)
        fees = math.fsum(trade.fee_pnl for trade in account.trades)
        funding = math.fsum(trade.funding_pnl for trade in account.trades)
        net = account.equity - INITIAL_EQUITY
        if not math.isclose(net, price + fees + funding, rel_tol=1e-8, abs_tol=1e-5):
            raise DownsideBetaFailure(f"{name}: PnL does not reconcile")
        pnl[name] = {"net": net, "executed_price_pnl": price,
                     "taker_fee_pnl": fees, "funding_pnl": funding}
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_band": .40 <= primary["cagr"] <= .60,
        "annual_folds": folds, "terminal_holdout": terminal,
        "monthly_returns": monthly, "monthly_summary": monthly_summary,
        "market_states": regimes, "monte_carlo": mc,
        "traded_weeks": len(baseline.trades),
        "signal_cash_weeks": signal_cash_weeks,
        "capacity_cash_weeks": capacity_cash_weeks,
        "terminal_cash_days": N_DAYS - WEEKS * 7,
        "schedule_sha256": schedule_hash,
        "actual_fill_participation_above_2pct": sum(
            trade.high_participation_fills for trade in baseline.trades),
        "actual_fill_zero_quote_volume": sum(
            trade.zero_volume_fills for trade in baseline.trades),
        "pnl_accounting": pnl,
        "gate": gate,
        "evidence_grade": "post-publication historical temporal OOS; no archived executable bid/ask or prospective fills",
    }
