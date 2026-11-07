"""Frozen weekly Amihud-illiquidity test; no parameter selection from returns."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import math

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .btc_eth_pairs_sim import DayRecord
from .weekday_seasonality_data import SYMBOLS, WeekdayTape
from .weekday_seasonality_sim import (
    BASE_DATE, END, END_DAY, FOLDS, INITIAL_EQUITY, N_DAYS, SCENARIOS,
    START, START_DAY, FEE, _daily_factors, _regimes,
)


WEEKS = 279
LOOKBACK_DAYS = 28


class IlliquidityFailure(ValueError):
    pass


def _score(tape: WeekdayTape, day_index: int) -> np.ndarray:
    """The 28 completed daily Amihud ratios, never the current UTC day."""
    first_day = day_index - LOOKBACK_DAYS
    if first_day < 1 or day_index * 288 > len(tape.close):
        raise IlliquidityFailure("28-day signal history unavailable")
    indices = np.arange(first_day, day_index, dtype=np.int64)
    last = tape.close[(indices + 1) * 288 - 1, :]
    previous = tape.close[indices * 288 - 1, :]
    quote = tape.quote_volume[first_day * 288:day_index * 288, :]
    daily_quote = quote.reshape(LOOKBACK_DAYS, 288, len(SYMBOLS)).sum(axis=1)
    if (last.shape != (LOOKBACK_DAYS, len(SYMBOLS))
            or previous.shape != last.shape or quote.shape != (LOOKBACK_DAYS * 288, len(SYMBOLS))
            or not np.all(np.isfinite(last)) or not np.all(np.isfinite(previous))
            or not np.all(np.isfinite(daily_quote)) or np.any(daily_quote <= 0)):
        raise IlliquidityFailure("invalid Amihud price or positive quote-volume denominator")
    ratios = np.abs(np.log(last / previous)) / daily_quote
    scores = ratios.mean(axis=0)
    if not np.all(np.isfinite(scores)):
        raise IlliquidityFailure("nonfinite Amihud score")
    return scores


def _rank(tape: WeekdayTape, day_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    scores = _score(tape, day_index)
    order = sorted(range(len(SYMBOLS)), key=lambda col: (float(scores[col]), SYMBOLS[col]))
    return tuple(order[-2:]), tuple(order[:2])


@dataclass(frozen=True, slots=True)
class WeekTrade:
    entry_day: object
    long_symbols: tuple[str, str]
    short_symbols: tuple[str, str]
    net_pnl: float
    executed_price_pnl: float
    fee_pnl: float
    funding_pnl: float
    high_participation_fills: int
    zero_volume_fills: int


@dataclass
class _WeeklyAccount:
    name: str
    slippage: float
    delay_bars: int
    no_funding_credit: bool
    equity: float = INITIAL_EQUITY
    peak: float = INITIAL_EQUITY
    max_drawdown: float = 0.0
    records: list[DayRecord] = field(default_factory=list)
    trades: list[WeekTrade] = field(default_factory=list)

    def flat_week(self, day_index: int) -> None:
        for offset in range(7):
            day = BASE_DATE + timedelta(days=day_index + offset)
            self.records.append(DayRecord(day, self.equity, self.equity, 1.0, 1.0, 0.0))

    def trade_week(self, tape: WeekdayTape, day_index: int,
                   longs: tuple[int, int], shorts: tuple[int, int]) -> None:
        if len(set(longs + shorts)) != 4 or self.equity <= 0:
            raise IlliquidityFailure("invalid weekly legs/equity")
        first = day_index * 288
        entry = first + self.delay_bars
        end = first + 7 * 288 + 1
        if not 0 <= entry < end < len(tape.opening):
            raise IlliquidityFailure("weekly trade exceeds source window")
        cols = np.array(longs + shorts, dtype=np.int64)
        signs = np.array((1.0, 1.0, -1.0, -1.0))
        initial = self.equity
        entry_mid = tape.opening[entry, cols]
        exit_mid = tape.opening[end, cols]
        entry_fill = entry_mid * (1 + signs * self.slippage)
        exit_fill = exit_mid * (1 - signs * self.slippage)
        quantity = signs * (initial * .25) / entry_fill
        entry_fee = FEE * float(np.sum(np.abs(quantity * entry_fill)))
        exit_fee = FEE * float(np.sum(np.abs(quantity * exit_fill)))
        price_pnl = float(np.sum(quantity * (exit_fill - entry_fill)))
        if not all(np.all(np.isfinite(value)) for value in
                   (entry_mid, exit_mid, entry_fill, exit_fill, quantity)):
            raise IlliquidityFailure("nonfinite weekly fill")

        funding_by_bar = np.zeros(end - entry, dtype=np.float64)
        for bar in range(entry, end):
            for col, rate in tape.funding.get(bar, ()):
                loc = np.flatnonzero(cols == col)
                if len(loc) != 1:
                    continue
                payment = -float(quantity[int(loc[0])]) * float(tape.opening[bar, col]) * rate
                if self.no_funding_credit:
                    payment = min(payment, 0.0)
                funding_by_bar[bar - entry] += payment
        funding_at_bar = np.cumsum(funding_by_bar)
        total_funding = float(funding_at_bar[-1])
        cumulative_before = np.concatenate(([0.0], funding_at_bar))

        def mark(prices: np.ndarray, funding: np.ndarray) -> np.ndarray:
            return (initial - entry_fee
                    + np.sum((prices - entry_fill) * quantity, axis=-1)
                    + funding)

        last_daily_equity = initial
        for offset in range(7):
            left = first + offset * 288 + (self.delay_bars if offset == 0 else 1)
            right = first + (offset + 1) * 288 + 1
            if not entry <= left < right <= end:
                raise IlliquidityFailure("weekly daily-bar partition invalid")
            within = slice(left, right)
            funding_after = funding_at_bar[within.start - entry:within.stop - entry]
            held_close = tape.close[within, :][:, cols]
            held_adverse = np.where(signs > 0,
                                    tape.low[within, :][:, cols],
                                    tape.high[within, :][:, cols])
            regular_marks = mark(held_close, funding_after)
            adverse_marks = mark(held_adverse, funding_after)
            next_boundary = right
            if next_boundary == end:
                daily_end = initial + price_pnl - entry_fee - exit_fee + total_funding
            else:
                funding_to_boundary = cumulative_before[next_boundary - entry]
                daily_end = float(mark(tape.opening[next_boundary, cols],
                                       funding_to_boundary))
            if offset == 0:
                entry_mark = float(mark(entry_mid, 0.0))
                regular = np.concatenate(([last_daily_equity, entry_mark],
                                          regular_marks, [daily_end]))
                adverse = np.concatenate(([last_daily_equity, entry_mark],
                                          adverse_marks, [daily_end]))
            else:
                regular = np.concatenate(([last_daily_equity], regular_marks, [daily_end]))
                adverse = np.concatenate(([last_daily_equity], adverse_marks, [daily_end]))
            if (not np.all(np.isfinite(regular)) or not np.all(np.isfinite(adverse))
                    or float(np.min(adverse)) <= 0):
                raise IlliquidityFailure(f"{self.name}: nonpositive/nonfinite intraweek equity")
            prior_peak = self.peak
            global_peaks = np.maximum.accumulate(np.maximum(regular, prior_peak))
            daily_peaks = np.maximum.accumulate(regular)
            self.max_drawdown = max(self.max_drawdown,
                                    float(np.max(1 - adverse / global_peaks)))
            intra_drawdown = float(np.max(1 - adverse / daily_peaks))
            self.peak = max(self.peak, float(np.max(regular)))
            day = BASE_DATE + timedelta(days=day_index + offset)
            self.records.append(DayRecord(day, last_daily_equity, daily_end,
                                          float(np.max(regular)) / last_daily_equity,
                                          float(np.min(adverse)) / last_daily_equity,
                                          intra_drawdown))
            last_daily_equity = daily_end

        self.equity = last_daily_equity
        high_participation = zero_volume = 0
        for bar, fills in ((entry, entry_fill), (end, exit_fill)):
            for k, col in enumerate(cols):
                volume = float(tape.quote_volume[bar, col])
                notional = abs(float(quantity[k] * fills[k]))
                if volume <= 0:
                    zero_volume += 1
                elif notional / volume > .02:
                    high_participation += 1
        self.trades.append(WeekTrade(BASE_DATE + timedelta(days=day_index),
                                     tuple(SYMBOLS[i] for i in longs),
                                     tuple(SYMBOLS[i] for i in shorts),
                                     self.equity - initial, price_pnl,
                                     -entry_fee - exit_fee, total_funding,
                                     high_participation, zero_volume))


def run_simulation(tape: WeekdayTape) -> tuple[dict[str, _WeeklyAccount], dict[int, str], str]:
    if (len(tape.opening) != END_DAY * 288
            or any(array.shape != tape.opening.shape for array in
                   (tape.high, tape.low, tape.close, tape.quote_volume))
            or tape.opening.shape[1] != len(SYMBOLS)):
        raise IlliquidityFailure("tape shape/registered source end changed")
    daily_prices = tape.opening[::288, :]
    daily_returns = np.log(daily_prices[1:] / daily_prices[:-1])
    if not np.all(np.isfinite(daily_returns)):
        raise IlliquidityFailure("nonfinite daily source return")
    states = _regimes(daily_prices, daily_returns)
    accounts = {name: _WeeklyAccount(name, *settings) for name, settings in SCENARIOS.items()}
    schedule_hash = hashlib.sha256()
    for week in range(WEEKS):
        day_index = START_DAY + week * 7
        longs, shorts = _rank(tape, day_index)
        selected = longs + shorts
        base_notional = accounts["base_1x"].equity * .25
        capacity = all(float(np.median(tape.quote_volume[
            (day_index - 1) * 288:day_index * 288, col])) >= 50 * base_notional
                       for col in selected)
        day = BASE_DATE + timedelta(days=day_index)
        if capacity:
            for account in accounts.values():
                account.trade_week(tape, day_index, longs, shorts)
            schedule_hash.update((f"{day.isoformat()}:"
                                  + ",".join(SYMBOLS[i] for i in selected) + "\n").encode())
        else:
            for account in accounts.values():
                account.flat_week(day_index)
            schedule_hash.update(f"{day.isoformat()}:capacity_cash\n".encode())
    for offset in range(N_DAYS - WEEKS * 7):
        day = START + timedelta(days=WEEKS * 7 + offset)
        for account in accounts.values():
            account.records.append(DayRecord(day, account.equity, account.equity,
                                             1.0, 1.0, 0.0))
        schedule_hash.update(f"{day.isoformat()}:terminal_cash\n".encode())
    if any(len(account.records) != N_DAYS for account in accounts.values()):
        raise IlliquidityFailure("daily grid incomplete")
    return accounts, states, schedule_hash.hexdigest()


def evaluate(tape: WeekdayTape) -> dict:
    accounts, states, schedule_hash = run_simulation(tape)
    factors = {name: _daily_factors(account.records) for name, account in accounts.items()}
    metrics = {name: _metrics_from_factors(factors[name]) for name in SCENARIOS}
    for name, account in accounts.items():
        if (not math.isclose(metrics[name]["ending_equity"], account.equity,
                             rel_tol=1e-8, abs_tol=1e-5)
                or not math.isclose(metrics[name]["max_drawdown"], account.max_drawdown,
                                    rel_tol=1e-8, abs_tol=1e-8)):
            raise IlliquidityFailure(f"{name}: compressed path mismatch")
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
    terminal = {"days": 133, "net_return": tail["ending_equity"] / INITIAL_EQUITY - 1,
                "pass": tail["ending_equity"] > INITIAL_EQUITY}

    by_month = defaultdict(list)
    for row in baseline.records:
        by_month[row.date.strftime("%Y-%m")].append(row)
    monthly = {month: group[-1].equity_end / group[0].equity_start - 1
               for month, group in sorted(by_month.items())}
    if len(monthly) != 65:
        raise IlliquidityFailure("monthly grid changed")
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
        regimes[state]["days"] += 1
        pnl = row.equity_end - row.equity_start
        regimes[state]["net_pnl"] += pnl
        regimes[state]["traded_days"] += pnl != 0
    mc = monte_carlo_compressed(base_factors, paths=5000, block_days=21,
                                seed=20261114)
    primary = metrics["base_1x"]
    gate = {
        "primary_metrics": (primary["cagr"] >= .40
                            and primary["sharpe"] is not None and primary["sharpe"] > 1.5
                            and primary["max_drawdown"] <= .20
                            and primary["profit_factor"] is not None
                            and primary["profit_factor"] > 1.3),
        "minimum_trade_weeks": len(baseline.trades) >= 150,
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
        price = math.fsum(trade.executed_price_pnl for trade in account.trades)
        fees = math.fsum(trade.fee_pnl for trade in account.trades)
        funding = math.fsum(trade.funding_pnl for trade in account.trades)
        net = account.equity - INITIAL_EQUITY
        if not math.isclose(net, price + fees + funding, rel_tol=1e-8, abs_tol=1e-5):
            raise IlliquidityFailure(f"{name}: trade PnL does not reconcile")
        pnl[name] = {"net": net, "executed_price_pnl": price,
                     "taker_fee_pnl": fees, "funding_pnl": funding}
    return {
        "metrics": metrics,
        "formal_cagr_within_40_to_60_percent_band": .40 <= primary["cagr"] <= .60,
        "annual_folds": folds, "terminal_holdout": terminal,
        "monthly_returns": monthly, "monthly_summary": monthly_summary,
        "market_states": regimes, "monte_carlo": mc,
        "traded_weeks": len(baseline.trades),
        "cash_weeks": WEEKS - len(baseline.trades),
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
