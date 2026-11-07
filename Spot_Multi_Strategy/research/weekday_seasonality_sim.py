"""Frozen one-day cross-sectional weekday-seasonality account replay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import math

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .btc_eth_pairs_sim import DayRecord
from .weekday_seasonality_data import BASE_MS, SYMBOLS, WeekdayTape


START = date(2021, 5, 21)
END = date(2026, 10, 1)
BASE_DATE = date(2021, 1, 1)
START_DAY = (START - BASE_DATE).days
END_DAY = (END - BASE_DATE).days
N_DAYS = (END - START).days
INITIAL_EQUITY = 10_000.0
FEE = 0.0005
FOLDS = tuple((date(2021 + i, 5, 21), date(2022 + i, 5, 21)) for i in range(5))
SCENARIOS = {
    "base_1x": (0.0005, 1, False),
    "double_slippage_1x": (0.0010, 1, False),
    "extra_bar_latency_1x": (0.0005, 2, False),
    "no_funding_credit_1x": (0.0005, 1, True),
}


class WeekdayStrategyFailure(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WeekdayTrade:
    entry_day: date
    long_symbols: tuple[str, str]
    short_symbols: tuple[str, str]
    regime: str
    net_pnl: float
    gross_price_pnl: float
    fee_pnl: float
    funding_pnl: float
    high_participation_fills: int
    zero_volume_fills: int


@dataclass
class _Account:
    name: str
    slippage: float
    delay_bars: int
    no_funding_credit: bool
    equity: float = INITIAL_EQUITY
    peak: float = INITIAL_EQUITY
    max_drawdown: float = 0.0
    records: list[DayRecord] = field(default_factory=list)
    trades: list[WeekdayTrade] = field(default_factory=list)

    def flat_day(self, day: date) -> None:
        self.records.append(DayRecord(day, self.equity, self.equity, 1.0, 1.0, 0.0))

    def execute(self, tape: WeekdayTape, day_index: int, longs: tuple[int, int],
                shorts: tuple[int, int], regime: str) -> None:
        if len(set(longs + shorts)) != 4 or self.equity <= 0:
            raise WeekdayStrategyFailure("duplicate leg or nonpositive starting equity")
        start_equity = self.equity
        base = day_index * 288
        entry = base + self.delay_bars
        exit_bar = base + 289
        if not 0 <= entry < exit_bar < len(tape.opening):
            raise WeekdayStrategyFailure("trade exceeds registered source window")
        cols = np.array(longs + shorts, dtype=np.int64)
        signs = np.array((1.0, 1.0, -1.0, -1.0), dtype=np.float64)
        opening = tape.opening
        entry_mid = opening[entry, cols]
        exit_mid = opening[exit_bar, cols]
        entry_fill = entry_mid * (1.0 + signs * self.slippage)
        exit_fill = exit_mid * (1.0 - signs * self.slippage)
        quantity = signs * (start_equity * .25) / entry_fill
        entry_fee = FEE * float(np.sum(np.abs(quantity * entry_fill)))
        exit_fee = FEE * float(np.sum(np.abs(quantity * exit_fill)))
        price_pnl = float(np.sum(quantity * (exit_fill - entry_fill)))

        funding_steps = np.zeros(exit_bar - entry, dtype=np.float64)
        for bar in range(entry, exit_bar):
            for coin, rate in tape.funding.get(bar, ()):
                matches = np.flatnonzero(cols == coin)
                if len(matches) != 1:
                    continue
                leg = int(matches[0])
                payment = -float(quantity[leg]) * float(opening[bar, coin]) * rate
                if self.no_funding_credit:
                    payment = min(payment, 0.0)
                funding_steps[bar - entry] += payment
        funding_cumulative = np.cumsum(funding_steps)
        funding_pnl = float(funding_cumulative[-1])
        cash_after_entry = start_equity - entry_fee
        held_close = tape.close[entry:exit_bar, :][:, cols]
        held_low = tape.low[entry:exit_bar, :][:, cols]
        held_high = tape.high[entry:exit_bar, :][:, cols]
        adverse_prices = np.where(signs > 0, held_low, held_high)
        regular = (cash_after_entry
                   + np.sum((held_close - entry_fill) * quantity, axis=1)
                   + funding_cumulative)
        adverse = (cash_after_entry
                   + np.sum((adverse_prices - entry_fill) * quantity, axis=1)
                   + funding_cumulative)
        entry_mark = cash_after_entry + float(np.sum((entry_mid - entry_fill) * quantity))
        final = start_equity + price_pnl - entry_fee - exit_fee + funding_pnl
        regular_path = np.concatenate(([start_equity, entry_mark], regular, [final]))
        adverse_path = np.concatenate(([start_equity, entry_mark], adverse, [final]))
        if (not np.all(np.isfinite(regular_path)) or not np.all(np.isfinite(adverse_path))
                or np.min(adverse_path) <= 0):
            raise WeekdayStrategyFailure(f"{self.name}: nonpositive or nonfinite intrabar equity")
        prior_peak = self.peak
        global_regular_peaks = np.maximum.accumulate(
            np.maximum(regular_path, prior_peak))
        global_drawdown = float(np.max(1.0 - adverse_path / global_regular_peaks))
        daily_regular_peaks = np.maximum.accumulate(regular_path)
        daily_drawdown = float(np.max(1.0 - adverse_path / daily_regular_peaks))
        self.peak = max(prior_peak, float(np.max(regular_path)))
        self.max_drawdown = max(self.max_drawdown, global_drawdown)
        self.equity = final

        count_high, count_zero = 0, 0
        for bar, prices in ((entry, entry_fill), (exit_bar, exit_fill)):
            for k, coin in enumerate(cols):
                volume = float(tape.quote_volume[bar, coin])
                notional = abs(float(quantity[k] * prices[k]))
                if volume <= 0:
                    count_zero += 1
                elif notional / volume > .02:
                    count_high += 1
        day = BASE_DATE + timedelta(days=day_index)
        self.records.append(DayRecord(
            day, start_equity, final,
            float(np.max(regular_path)) / start_equity,
            float(np.min(adverse_path)) / start_equity,
            daily_drawdown,
        ))
        self.trades.append(WeekdayTrade(
            day, tuple(SYMBOLS[i] for i in longs),
            tuple(SYMBOLS[i] for i in shorts), regime,
            final - start_equity, price_pnl, -entry_fee - exit_fee,
            funding_pnl, count_high, count_zero,
        ))


def _ranking(daily_returns: np.ndarray, day_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    lookbacks = np.array([day_index - 7 * week for week in range(1, 21)], dtype=np.int64)
    if lookbacks[-1] < 0:
        raise WeekdayStrategyFailure("20-week signal history incomplete")
    scores = np.mean(daily_returns[lookbacks, :], axis=0)
    if not np.all(np.isfinite(scores)):
        raise WeekdayStrategyFailure("nonfinite weekday signal")
    ranked = sorted(range(len(SYMBOLS)), key=lambda col: (float(scores[col]), SYMBOLS[col]))
    return tuple(ranked[-2:]), tuple(ranked[:2])


def _regimes(daily_prices: np.ndarray, daily_returns: np.ndarray) -> dict[int, str]:
    btc = SYMBOLS.index("BTCUSDT")
    vol = np.full(len(daily_prices), np.nan, dtype=np.float64)
    for end in range(30, len(daily_prices)):
        vol[end] = float(np.std(daily_returns[end - 30:end, btc], ddof=1))
    states = {}
    for day_index in range(START_DAY, END_DAY):
        trailing_return = math.log(float(daily_prices[day_index, btc]
                                         / daily_prices[day_index - 30, btc]))
        threshold = float(np.median(vol[day_index - 89:day_index + 1]))
        current = float(vol[day_index])
        if not all(math.isfinite(x) for x in (trailing_return, threshold, current)):
            raise WeekdayStrategyFailure("regime history incomplete or nonfinite")
        states[day_index] = ("bull" if trailing_return > 0 else "bear") + "_" + (
            "high" if current > threshold else "low")
    return states


def _daily_factors(records: list[DayRecord]) -> np.ndarray:
    if len(records) != N_DAYS:
        raise WeekdayStrategyFailure("formal daily record count changed")
    factors = np.empty((N_DAYS, 4), dtype=np.float64)
    previous_equity = INITIAL_EQUITY
    for j, record in enumerate(records):
        if (record.date != START + timedelta(days=j)
                or not math.isclose(record.equity_start, previous_equity,
                                    rel_tol=1e-10, abs_tol=1e-6)):
            raise WeekdayStrategyFailure("daily account path discontinuity")
        factors[j] = (record.equity_end / record.equity_start,
                      record.max_regular_factor,
                      record.min_adverse_factor,
                      record.max_intraday_drawdown)
        previous_equity = record.equity_end
    return factors


def run_simulation(tape: WeekdayTape) -> tuple[dict[str, _Account], dict[int, str], str]:
    if (len(tape.opening) != END_DAY * 288
            or any(array.shape != tape.opening.shape for array in (
                tape.high, tape.low, tape.close, tape.quote_volume))
            or tape.opening.shape[1] != len(SYMBOLS)):
        raise WeekdayStrategyFailure("tape shape or registered end changed")
    daily_prices = tape.opening[::288, :]
    daily_returns = np.log(daily_prices[1:] / daily_prices[:-1])
    if not np.all(np.isfinite(daily_returns)):
        raise WeekdayStrategyFailure("nonfinite daily source return")
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
        longs, shorts = _ranking(daily_returns, day_index)
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
        raise WeekdayStrategyFailure("daily record grid incomplete")
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
            raise WeekdayStrategyFailure(f"{name}: compressed replay mismatch")
    baseline = accounts["base_1x"]
    base_factors = factors["base_1x"]
    folds = {}
    for index, (first, last) in enumerate(FOLDS, start=1):
        a, b = (first - START).days, (last - START).days
        fold = _metrics_from_factors(base_factors[a:b])
        count = sum(first <= trade.entry_day < last for trade in baseline.trades)
        folds[f"year_{index}"] = {"start": str(first), "end_exclusive": str(last),
                                  "days": b - a, "trades": count,
                                  "net_return": fold["ending_equity"] / INITIAL_EQUITY - 1,
                                  "daily_sharpe": fold["sharpe"],
                                  "pass": count >= 1 and fold["ending_equity"] > INITIAL_EQUITY
                                  and fold["sharpe"] is not None and fold["sharpe"] > 0}
    tail = _metrics_from_factors(base_factors[(FOLDS[-1][1] - START).days:])
    terminal = {"days": 133, "net_return": tail["ending_equity"] / INITIAL_EQUITY - 1,
                "pass": tail["ending_equity"] > INITIAL_EQUITY}

    by_month: dict[str, list[DayRecord]] = defaultdict(list)
    for record in baseline.records:
        by_month[record.date.strftime("%Y-%m")].append(record)
    monthly = {month: group[-1].equity_end / group[0].equity_start - 1
               for month, group in sorted(by_month.items())}
    if len(monthly) != 65:
        raise WeekdayStrategyFailure("monthly grid changed")
    positive_months = sum(value > 0 for value in monthly.values())
    longest_losing, losing = 0, 0
    for value in monthly.values():
        losing = losing + 1 if value < 0 else 0
        longest_losing = max(longest_losing, losing)
    monthly_summary = {"positive": positive_months,
                       "negative": sum(value < 0 for value in monthly.values()),
                       "flat": sum(value == 0 for value in monthly.values()),
                       "longest_losing_streak": longest_losing,
                       "worst_month": min(monthly, key=monthly.get),
                       "worst_month_return": min(monthly.values())}

    regimes = {name: {"days": 0, "trades": 0, "net_pnl": 0.0}
               for name in ("bull_high", "bull_low", "bear_high", "bear_low")}
    for state in states.values():
        regimes[state]["days"] += 1
    for trade in baseline.trades:
        regimes[trade.regime]["trades"] += 1
        regimes[trade.regime]["net_pnl"] += trade.net_pnl
    mc = monte_carlo_compressed(base_factors, paths=5000, block_days=21,
                                seed=20261112)
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
            raise WeekdayStrategyFailure(f"{name}: trade PnL does not reconcile")
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
        "actual_fill_zero_quote_volume": sum(trade.zero_volume_fills for trade in baseline.trades),
        "pnl_accounting": pnl,
        "gate": gate,
        "evidence_grade": "post-publication historical temporal OOS; no archived executable bid/ask or prospective fills",
    }
