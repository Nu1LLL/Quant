"""Preregistered daily high-minus-low BTC-residual-volatility transfer test."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import hashlib
import math

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .weekday_seasonality_data import SYMBOLS, WeekdayTape
from .weekday_seasonality_sim import (
    BASE_DATE, END_DAY, INITIAL_EQUITY, SCENARIOS, _Account, _regimes,
)


START = date(2021, 7, 9)
END = date(2026, 10, 1)
START_DAY = (START - BASE_DATE).days
N_DAYS = (END - START).days
LOOKBACK_DAYS = 182
BTC = SYMBOLS.index("BTCUSDT")
ALT_COLS = tuple(i for i, symbol in enumerate(SYMBOLS) if symbol != "BTCUSDT")
FOLDS = tuple((date(2021 + i, 7, 9), date(2022 + i, 7, 9)) for i in range(5))


class IvolFailure(ValueError):
    pass


def _completed_returns(tape: WeekdayTape, day_index: int) -> np.ndarray:
    """182 daily returns ending at the *last completed* 23:55 five-minute bar."""
    boundaries = (day_index - np.arange(LOOKBACK_DAYS + 1)) * 288 - 1
    if boundaries[-1] < 0 or boundaries[0] >= len(tape.close):
        raise IvolFailure("completed 182-day source history unavailable")
    prices = tape.close[boundaries, :]
    if (prices.shape != (LOOKBACK_DAYS + 1, len(SYMBOLS))
            or not np.all(np.isfinite(prices)) or np.any(prices <= 0)):
        raise IvolFailure("invalid completed daily close")
    returns = np.log(prices[:-1] / prices[1:])
    if returns.shape != (LOOKBACK_DAYS, len(SYMBOLS)) or not np.all(np.isfinite(returns)):
        raise IvolFailure("nonfinite completed daily returns")
    return returns


def _rank(returns: np.ndarray, eligible: tuple[int, ...]
          ) -> tuple[tuple[int, int], tuple[int, int]]:
    if (returns.shape != (LOOKBACK_DAYS, len(SYMBOLS))
            or not np.all(np.isfinite(returns))
            or len(eligible) < 4 or len(set(eligible)) != len(eligible)
            or any(col not in ALT_COLS for col in eligible)):
        raise IvolFailure("invalid residual-volatility ranking input")
    market = returns[:, BTC]
    design = np.column_stack((np.ones(LOOKBACK_DAYS), market))
    if np.linalg.matrix_rank(design) != 2:
        raise IvolFailure("BTC market regression is rank deficient")
    selected = returns[:, eligible]
    coefficients = np.linalg.lstsq(design, selected, rcond=None)[0]
    residuals = selected - design @ coefficients
    scores = np.sqrt(np.sum(residuals * residuals, axis=0) / (LOOKBACK_DAYS - 2))
    if len(scores) != len(eligible) or not np.all(np.isfinite(scores)):
        raise IvolFailure("nonfinite residual-volatility score")
    order = sorted(range(len(eligible)), key=lambda k: (float(scores[k]), SYMBOLS[eligible[k]]))
    return tuple(eligible[k] for k in order[-2:]), tuple(eligible[k] for k in order[:2])


def _daily_factors(records) -> np.ndarray:
    if len(records) != N_DAYS:
        raise IvolFailure("formal daily record count changed")
    factors = np.empty((N_DAYS, 4), dtype=np.float64)
    prior_equity = INITIAL_EQUITY
    for j, row in enumerate(records):
        if (row.date != START + timedelta(days=j)
                or not math.isclose(row.equity_start, prior_equity,
                                    rel_tol=1e-10, abs_tol=1e-6)):
            raise IvolFailure("daily account path discontinuity")
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
        raise IvolFailure("tape shape or registered source end changed")
    daily_prices = tape.opening[::288, :]
    daily_returns = np.log(daily_prices[1:] / daily_prices[:-1])
    if not np.all(np.isfinite(daily_returns)):
        raise IvolFailure("nonfinite daily source return")
    states = _regimes(daily_prices, daily_returns)
    accounts = {name: _Account(name, *settings) for name, settings in SCENARIOS.items()}
    schedule_hash = hashlib.sha256()
    capacity_cash_days = 0
    eligible_days_by_coin = {symbol: 0 for symbol in SYMBOLS if symbol != "BTCUSDT"}
    for day_index in range(START_DAY, END_DAY):
        day = BASE_DATE + timedelta(days=day_index)
        if day == END - timedelta(days=1):
            for account in accounts.values():
                account.flat_day(day)
            schedule_hash.update(f"{day.isoformat()}:terminal_cash\n".encode())
            continue
        notional = accounts["base_1x"].equity * .25
        eligible = tuple(col for col in ALT_COLS if float(np.median(tape.quote_volume[
            (day_index - 1) * 288:day_index * 288, col])) >= 50 * notional)
        for col in eligible:
            eligible_days_by_coin[SYMBOLS[col]] += 1
        if len(eligible) < 4:
            capacity_cash_days += 1
            for account in accounts.values():
                account.flat_day(day)
            schedule_hash.update(f"{day.isoformat()}:capacity_cash\n".encode())
            continue
        longs, shorts = _rank(_completed_returns(tape, day_index), eligible)
        for account in accounts.values():
            account.execute(tape, day_index, longs, shorts, states[day_index])
        schedule_hash.update((f"{day.isoformat()}:"
                              + ",".join(SYMBOLS[i] for i in longs + shorts) + "\n").encode())
    if any(len(account.records) != N_DAYS for account in accounts.values()):
        raise IvolFailure("daily account grid incomplete")
    return accounts, states, schedule_hash.hexdigest(), capacity_cash_days, eligible_days_by_coin


def evaluate(tape: WeekdayTape) -> dict:
    accounts, states, schedule_hash, capacity_cash_days, eligible_days = run_simulation(tape)
    factors = {name: _daily_factors(account.records) for name, account in accounts.items()}
    metrics = {name: _metrics_from_factors(factors[name]) for name in SCENARIOS}
    for name, account in accounts.items():
        if (not math.isclose(metrics[name]["ending_equity"], account.equity,
                             rel_tol=1e-8, abs_tol=1e-5)
                or not math.isclose(metrics[name]["max_drawdown"], account.max_drawdown,
                                    rel_tol=1e-8, abs_tol=1e-8)):
            raise IvolFailure(f"{name}: compressed account mismatch")
    baseline = accounts["base_1x"]
    base_factors = factors["base_1x"]
    folds = {}
    for index, (first, last) in enumerate(FOLDS, start=1):
        a, b = (first - START).days, (last - START).days
        fold = _metrics_from_factors(base_factors[a:b])
        count = sum(first <= trade.entry_day < last for trade in baseline.trades)
        folds[f"year_{index}"] = {
            "start": str(first), "end_exclusive": str(last), "days": b - a,
            "trades": count, "net_return": fold["ending_equity"] / INITIAL_EQUITY - 1,
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
        raise IvolFailure("monthly account grid changed")
    positive = sum(value > 0 for value in monthly.values())
    losing = longest_losing = 0
    for value in monthly.values():
        losing = losing + 1 if value < 0 else 0
        longest_losing = max(longest_losing, losing)
    monthly_summary = {
        "positive": positive, "negative": sum(value < 0 for value in monthly.values()),
        "flat": sum(value == 0 for value in monthly.values()),
        "longest_losing_streak": longest_losing,
        "worst_month": min(monthly, key=monthly.get),
        "worst_month_return": min(monthly.values()),
    }
    regimes = {state: {"days": 0, "trades": 0, "net_pnl": 0.0}
               for state in ("bull_high", "bull_low", "bear_high", "bear_low")}
    for j in range(len(baseline.records)):
        regimes[states[START_DAY + j]]["days"] += 1
    for trade in baseline.trades:
        regimes[trade.regime]["trades"] += 1
        regimes[trade.regime]["net_pnl"] += trade.net_pnl
    mc = monte_carlo_compressed(base_factors, paths=5000, block_days=21,
                                seed=20261116)
    primary = metrics["base_1x"]
    gate = {
        "primary_metrics": (primary["cagr"] >= .40
                            and primary["sharpe"] is not None and primary["sharpe"] > 1.5
                            and primary["max_drawdown"] <= .20
                            and primary["profit_factor"] is not None
                            and primary["profit_factor"] > 1.3),
        "minimum_trade_days": len(baseline.trades) >= 1000,
        "five_annual_walk_forward_folds": all(item["pass"] for item in folds.values()),
        "terminal_84_days": terminal["pass"],
        "monthly_stability": positive >= 38 and longest_losing <= 3,
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
            raise IvolFailure(f"{name}: trade PnL does not reconcile")
        pnl[name] = {"net": net, "executed_price_pnl": price,
                     "taker_fee_pnl": fees, "funding_pnl": funding}
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_band": .40 <= primary["cagr"] <= .60,
        "annual_folds": folds, "terminal_holdout": terminal,
        "monthly_returns": monthly, "monthly_summary": monthly_summary,
        "market_states": regimes, "monte_carlo": mc,
        "traded_days": len(baseline.trades),
        "capacity_cash_days": capacity_cash_days,
        "cash_days": N_DAYS - len(baseline.trades),
        "eligible_days_by_coin": eligible_days,
        "schedule_sha256": schedule_hash,
        "actual_fill_participation_above_2pct": sum(
            trade.high_participation_fills for trade in baseline.trades),
        "actual_fill_zero_quote_volume": sum(
            trade.zero_volume_fills for trade in baseline.trades),
        "pnl_accounting": pnl,
        "gate": gate,
        "evidence_grade": "post-publication historical transfer; reused program history and no archived executable BBO/fills",
    }
