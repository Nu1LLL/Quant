"""One fixed post-launch short hypothesis; no parameter search or rerun path."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import math
from statistics import median

import numpy as np

from .binance_new_perpetual_data import EventTape, LaunchTape
from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed


INITIAL = 10_000.0
START = date(2021, 1, 1)
END = date(2026, 1, 1)
FEE = 0.0005
BASE_SLIPPAGE = 0.0015
NOTIONAL_FRACTION = 0.25
CAPACITY_MULTIPLE = 50.0
STOP_MULTIPLE = 1.10
MC_SEED = 20261118


class LaunchStrategyFailure(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Observation:
    time: datetime
    regular: float
    adverse: float


@dataclass(frozen=True, slots=True)
class Trade:
    article_code: str
    symbol: str
    original_entry: datetime
    fill_entry: datetime
    fill_exit: datetime
    scheduled_exit: datetime
    stopped: bool
    regime: str
    starting_equity: float
    ending_equity: float
    gross_price_pnl: float
    fee_pnl: float
    funding_pnl: float
    entry_participation: float | None
    exit_participation: float | None
    observations: tuple[Observation, ...]

    @property
    def net_pnl(self) -> float:
        return self.ending_equity - self.starting_equity


def _regime(tape: LaunchTape, entry: datetime) -> str:
    prior = entry.date() - timedelta(days=1)
    dates = [prior - timedelta(days=j) for j in range(30, -1, -1)]
    try:
        closes = np.array([tape.btc_close[day] for day in dates], dtype=float)
    except KeyError as exc:
        raise LaunchStrategyFailure("missing previous 31 BTC closes") from exc
    if len(closes) != 31 or not np.all(np.isfinite(closes)) or np.any(closes <= 0):
        raise LaunchStrategyFailure("invalid BTC regime source")
    returns = np.diff(np.log(closes))
    momentum = float(np.sum(returns))
    volatility = float(np.std(returns, ddof=1) * math.sqrt(365.0))
    if not math.isfinite(volatility):
        raise LaunchStrategyFailure("nonfinite BTC regime volatility")
    return ("bull" if momentum > 0 else "bear") + ("_high" if volatility >= .80 else "_low")


def _participation(notional: float, volume: float) -> float | None:
    return notional / volume if volume > 0 else None


def _execute(event: EventTape, equity: float, regime: str, *,
             slippage: float, latency_bars: int) -> Trade:
    if not math.isfinite(equity) or equity <= 0 or latency_bars not in (0, 1):
        raise LaunchStrategyFailure("invalid account or latency")
    by_time = {bar.time: bar for bar in event.bars}
    entry_time = event.entry + timedelta(minutes=5 * latency_bars)
    try:
        entry_bar, scheduled_exit_bar = by_time[entry_time], by_time[event.exit]
    except KeyError as exc:
        raise LaunchStrategyFailure("entry/exit bar missing") from exc
    notional = NOTIONAL_FRACTION * equity
    entry_fill = entry_bar.opening * (1.0 - slippage)
    quantity = notional / entry_fill
    entry_fee = FEE * notional
    stop_price = STOP_MULTIPLE * entry_fill
    funding_by_time = defaultdict(list)
    for record in event.funding:
        if record.time not in by_time:
            raise LaunchStrategyFailure("funding event is not on the 5m grid")
        funding_by_time[record.time].append(record.rate)
    funding_debit = 0.0
    observations = [Observation(entry_time, equity - entry_fee +
                                quantity * (entry_fill - entry_bar.opening),
                                equity - entry_fee + quantity * (entry_fill - entry_bar.opening))]
    trigger_high = None
    exit_time = event.exit
    for bar in event.bars:
        if not entry_time <= bar.time < event.exit:
            continue
        for rate in funding_by_time.get(bar.time, ()):
            if rate < 0:
                funding_debit += quantity * max(entry_fill, bar.high) * -rate
        regular = equity - entry_fee + quantity * (entry_fill - bar.close) - funding_debit
        adverse = equity - entry_fee + quantity * (entry_fill - bar.high) - funding_debit
        # End-of-bar marks belong to the bar's UTC calendar date, including 23:55.
        marked_at = bar.time + timedelta(minutes=5) - timedelta(microseconds=1)
        observations.append(Observation(marked_at, regular, adverse))
        if bar.high >= stop_price:
            trigger_high = bar.high
            exit_time = bar.time + timedelta(minutes=5)
            break
    if exit_time not in by_time:
        raise LaunchStrategyFailure("stop exit bar missing")
    exit_bar = by_time[exit_time]
    exit_mid = (max(exit_bar.opening, trigger_high, stop_price)
                if trigger_high is not None else scheduled_exit_bar.opening)
    exit_fill = exit_mid * (1.0 + slippage)
    exit_fee = FEE * quantity * exit_fill
    gross = quantity * (entry_fill - exit_fill)
    final = equity + gross - entry_fee - exit_fee - funding_debit
    observations.append(Observation(exit_time, final, final))
    if (not all(math.isfinite(x) for point in observations
                for x in (point.regular, point.adverse))
            or min(point.adverse for point in observations) <= 0
            or final <= 0):
        raise LaunchStrategyFailure("nonpositive or nonfinite account mark")
    if not math.isclose(final - equity, gross - entry_fee - exit_fee - funding_debit,
                        rel_tol=1e-10, abs_tol=1e-8):
        raise LaunchStrategyFailure("trade PnL does not reconcile")
    if any(b.time <= a.time for a, b in zip(observations, observations[1:])):
        raise LaunchStrategyFailure("nonmonotone trade observations")
    return Trade(event.article_code, event.symbol, event.entry, entry_time, exit_time,
                 event.exit, trigger_high is not None, regime, equity, final, gross,
                 -entry_fee - exit_fee, -funding_debit,
                 _participation(notional, entry_bar.quote_volume),
                 _participation(quantity * exit_fill, exit_bar.quote_volume),
                 tuple(observations))


def _select_and_trade(tape: LaunchTape) -> tuple[list[Trade], list[EventTape], dict]:
    if len(tape.events) != 215:
        raise LaunchStrategyFailure("registered event count changed")
    events = sorted(tape.events, key=lambda e: (e.effective, e.symbol))
    equity = INITIAL
    slot_end = datetime.min.replace(tzinfo=timezone.utc)
    selected, trades = [], []
    skips = {"capacity": 0, "overlap": 0}
    schedule = sha256()
    for event in events:
        if event.entry < slot_end:
            skips["overlap"] += 1
            schedule.update(f"{event.article_code}:overlap\n".encode())
            continue
        if len(event.bars) != 289 or event.bars[0].time != event.effective:
            raise LaunchStrategyFailure("event 5m grid changed")
        past_volume = [bar.quote_volume for bar in event.bars[:6]]
        if median(past_volume) < CAPACITY_MULTIPLE * NOTIONAL_FRACTION * equity:
            skips["capacity"] += 1
            schedule.update(f"{event.article_code}:capacity\n".encode())
            continue
        regime = _regime(tape, event.entry)
        trade = _execute(event, equity, regime, slippage=BASE_SLIPPAGE, latency_bars=0)
        selected.append(event)
        trades.append(trade)
        equity = trade.ending_equity
        slot_end = event.exit
        schedule.update(f"{event.article_code}:trade\n".encode())
    if len(selected) + sum(skips.values()) != len(events):
        raise LaunchStrategyFailure("event selection count mismatch")
    skips["schedule_sha256"] = schedule.hexdigest()
    return trades, selected, skips


def _daily_factors(trades: list[Trade]) -> tuple[np.ndarray, dict[date, float], float]:
    by_day = defaultdict(list)
    for trade in trades:
        for point in trade.observations:
            by_day[point.time.date()].append(point)
    factors = np.empty(((END - START).days, 4), dtype=float)
    daily_end = {}
    equity = INITIAL
    global_peak = INITIAL
    direct_dd = 0.0
    for index in range((END - START).days):
        day = START + timedelta(days=index)
        start = equity
        regular_peak = adverse_min = equity
        intraday_peak = equity
        intraday_dd = 0.0
        for point in sorted(by_day[day], key=lambda p: p.time):
            direct_dd = max(direct_dd, 1.0 - point.adverse / global_peak)
            intraday_dd = max(intraday_dd, 1.0 - point.adverse / intraday_peak)
            regular_peak = max(regular_peak, point.regular)
            global_peak = max(global_peak, point.regular)
            intraday_peak = max(intraday_peak, point.regular)
            adverse_min = min(adverse_min, point.adverse)
            equity = point.regular
        factors[index] = (equity / start, regular_peak / start,
                          adverse_min / start, intraday_dd)
        daily_end[day] = equity
    if trades and not math.isclose(equity, trades[-1].ending_equity,
                                   rel_tol=1e-9, abs_tol=1e-5):
        raise LaunchStrategyFailure("terminal equity differs from trade ledger")
    if not np.all(np.isfinite(factors)) or np.any(factors[:, :3] <= 0):
        raise LaunchStrategyFailure("invalid daily factor path")
    return factors, daily_end, direct_dd


def _scenario(tape: LaunchTape, selected: list[EventTape], *,
              slippage: float, latency_bars: int) -> tuple[list[Trade], np.ndarray, dict, float]:
    equity = INITIAL
    trades = []
    for event in selected:
        regime = _regime(tape, event.entry)
        trade = _execute(event, equity, regime, slippage=slippage,
                         latency_bars=latency_bars)
        trades.append(trade)
        equity = trade.ending_equity
    factors, daily_end, direct_dd = _daily_factors(trades)
    return trades, factors, daily_end, direct_dd


def evaluate(tape: LaunchTape) -> dict:
    baseline_trades, selected, selection = _select_and_trade(tape)
    baseline_factors, baseline_daily, baseline_direct_dd = _daily_factors(baseline_trades)
    scenarios = {"base": (baseline_trades, baseline_factors, baseline_daily,
                           baseline_direct_dd)}
    scenarios["double_slippage"] = _scenario(tape, selected, slippage=.0030,
                                              latency_bars=0)
    scenarios["extra_bar_latency"] = _scenario(tape, selected, slippage=.0015,
                                                latency_bars=1)
    metrics = {}
    reconciliation = {}
    for name, (trades, factors, _, direct_dd) in scenarios.items():
        metric = _metrics_from_factors(factors)
        if (not math.isclose(metric["max_drawdown"], direct_dd,
                             rel_tol=1e-8, abs_tol=1e-8)
                or (trades and not math.isclose(metric["ending_equity"],
                                                trades[-1].ending_equity,
                                                rel_tol=1e-8, abs_tol=1e-5))):
            raise LaunchStrategyFailure(f"{name}: compressed metrics disagree with ledger")
        price = math.fsum(trade.gross_price_pnl for trade in trades)
        fees = math.fsum(trade.fee_pnl for trade in trades)
        funding = math.fsum(trade.funding_pnl for trade in trades)
        if not math.isclose(metric["ending_equity"] - INITIAL, price + fees + funding,
                            rel_tol=1e-8, abs_tol=1e-5):
            raise LaunchStrategyFailure(f"{name}: PnL components do not reconcile")
        metrics[name] = metric
        reconciliation[name] = {"gross_price_pnl": price, "fees": fees,
                                "funding": funding,
                                "net": metric["ending_equity"] - INITIAL}
    annual = {}
    for year in range(2021, 2026):
        first = (date(year, 1, 1) - START).days
        last = (date(year + 1, 1, 1) - START).days
        fold = _metrics_from_factors(baseline_factors[first:last])
        trade_count = sum(t.fill_entry.year == year for t in baseline_trades)
        annual[str(year)] = {"trades": trade_count,
                             "net_return": fold["ending_equity"] / INITIAL - 1,
                             "sharpe": fold["sharpe"],
                             "pass": trade_count >= 1 and fold["ending_equity"] > INITIAL
                             and fold["sharpe"] is not None and fold["sharpe"] > 0}
    months = {}
    prev = INITIAL
    for day, value in baseline_daily.items():
        if day.month == 12 and day.day == 31 or (day + timedelta(days=1)).month != day.month:
            key = day.strftime("%Y-%m")
            months[key] = value / prev - 1
            prev = value
    if len(months) != 60:
        raise LaunchStrategyFailure("60-month grid changed")
    positive = sum(x > 0 for x in months.values())
    streak = longest = 0
    for value in months.values():
        streak = streak + 1 if value <= 0 else 0
        longest = max(longest, streak)
    regimes = {name: {"trades": 0, "net_pnl": 0.0}
               for name in ("bull_high", "bull_low", "bear_high", "bear_low")}
    for trade in baseline_trades:
        regimes[trade.regime]["trades"] += 1
        regimes[trade.regime]["net_pnl"] += trade.net_pnl
    mc = monte_carlo_compressed(baseline_factors, paths=5000, block_days=21,
                                seed=MC_SEED)
    primary = metrics["base"]
    gate = {
        "primary_metrics": primary["cagr"] >= .40 and primary["sharpe"] is not None
        and primary["sharpe"] > 1.5 and primary["max_drawdown"] <= .20
        and primary["profit_factor"] is not None and primary["profit_factor"] > 1.3,
        "minimum_100_trades": len(baseline_trades) >= 100,
        "five_annual_walk_forward_folds": all(x["pass"] for x in annual.values()),
        "monthly_stability": positive >= 36 and longest <= 3,
        "four_market_states": all(x["trades"] >= 10 and x["net_pnl"] > 0
                                  for x in regimes.values()),
        "monte_carlo": mc["joint_pass_fraction"] >= .80,
        "double_slippage_profitable": metrics["double_slippage"]["ending_equity"] > INITIAL,
        "extra_bar_latency_profitable": metrics["extra_bar_latency"]["ending_equity"] > INITIAL,
        "independent_executable_oos": False,
    }
    gate["all_numeric"] = all(value for key, value in gate.items()
                              if key != "independent_executable_oos")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_oos"]
    return {
        "event_count": len(tape.events), "selected_trades": len(baseline_trades),
        "skips": selection,
        "stopped_trades": sum(t.stopped for t in baseline_trades),
        "actual_entry_participation_over_2pct": sum(
            t.entry_participation is None or t.entry_participation > .02
            for t in baseline_trades),
        "actual_exit_participation_over_2pct": sum(
            t.exit_participation is None or t.exit_participation > .02
            for t in baseline_trades),
        "metrics": metrics, "pnl_accounting": reconciliation,
        "annual_folds": annual, "monthly_returns": months,
        "monthly_summary": {"positive": positive,
                            "nonpositive": len(months) - positive,
                            "longest_nonpositive_streak": longest},
        "market_states": regimes, "monte_carlo": mc,
        "gate": gate,
        "evidence_grade": "price-blind historical transfer; no independent executable BBO/fill OOS",
        "source_hashes": tape.source_hashes,
    }
