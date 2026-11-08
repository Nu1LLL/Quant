"""Frozen issuance-ranked USD-M spread replay with hourly marks and real funding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from statistics import median
from typing import Iterable

import numpy as np
import pandas as pd

from research.pow_issuance_data import FundingEvent, HourBar, SYMBOLS, TerminalOpen
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass


UTC = timezone.utc
START = datetime(2021, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 1, tzinfo=UTC)
INPUT_START = datetime(2020, 12, 1, tzinfo=UTC)
INITIAL = 10000.0
LOT = .1
FEE = .0005
SLIP = .001
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261126


class ReplayFailure(ValueError):
    pass


def weekly_rankings(issuance: pd.DataFrame) -> dict[date, tuple[int, int]]:
    expected = pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")
    columns = tuple((symbol, metric) for symbol in SYMBOLS
                    for metric in ("IssTotNtv", "SplyCur"))
    if not issuance.index.equals(expected) or tuple(issuance.columns) != columns:
        raise ReplayFailure("issuance date grid or fixed source columns changed")
    values = issuance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ReplayFailure("invalid native issuance or supply value")
    result = {}
    for thursday in pd.date_range("2021-01-07", "2025-12-25", freq="W-THU"):
        day = thursday.date()
        sunday = day - timedelta(days=4)
        window = issuance.loc[pd.date_range(sunday - timedelta(days=27),
                                             sunday, tz="UTC", freq="D")]
        if len(window) != 28:
            raise ReplayFailure("incomplete 28-day native issuance window")
        scores = {symbol: (365.0 / 28.0) * float(window[(symbol, "IssTotNtv")].sum()) /
                  float(window[(symbol, "SplyCur")].iloc[-1]) for symbol in SYMBOLS}
        ranking = sorted(SYMBOLS, key=lambda symbol: (scores[symbol], symbol))
        result[day] = (SYMBOLS.index(ranking[0]), SYMBOLS.index(ranking[-1]))
    if len(result) != 260:
        raise ReplayFailure("weekly decision count changed")
    return result


@dataclass
class _Position:
    opened: date
    long_index: int
    short_index: int
    long_qty: float
    short_qty: float
    long_entry: float
    short_entry: float
    entry_equity: float
    fee_pnl: float
    funding_pnl: float = 0.0
    slippage_drag: float = 0.0


@dataclass
class _Account:
    name: str
    slip: float
    fee_rate: float
    latency_days: int
    cash: float = INITIAL
    position: _Position | None = None
    last_equity: float = INITIAL
    total_fees: float = 0.0
    total_funding: float = 0.0
    total_slippage: float = 0.0
    days: list[Daily] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    day: date | None = None
    day_start: float = INITIAL
    day_high: float = INITIAL
    day_low: float = INITIAL
    day_fee: float = 0.0
    day_exposed: int = 0

    def equity(self, prices: tuple[float, float, float]) -> float:
        pos = self.position
        if pos is None:
            return self.cash
        return (self.cash + pos.long_qty * (prices[pos.long_index] - pos.long_entry)
                + pos.short_qty * (pos.short_entry - prices[pos.short_index]))

    def _risk(self, equity: float, prices: tuple[float, float, float]) -> None:
        pos = self.position
        gross = 0.0 if pos is None else (pos.long_qty * prices[pos.long_index] +
                                          pos.short_qty * prices[pos.short_index])
        if (not math.isfinite(equity) or equity <= 0 or
                (pos is not None and gross / equity > 5)):
            raise ReplayFailure(f"{self.name}: ruined or above five-times gross leverage")

    def touch(self, prices: tuple[float, float, float]) -> None:
        equity = self.equity(prices)
        self._risk(equity, prices)
        if self.day is not None:
            self.day_high = max(self.day_high, equity)
            self.day_low = min(self.day_low, equity)
            self.day_exposed = max(self.day_exposed, int(self.position is not None))

    def begin_day(self, day: date) -> None:
        if self.day is not None:
            self.finish_day()
        self.day = day
        self.day_start = self.last_equity
        self.day_high = self.last_equity
        self.day_low = self.last_equity
        self.day_fee = 0.0
        self.day_exposed = int(self.position is not None)

    def finish_day(self) -> None:
        if self.day is None:
            raise ReplayFailure("cannot finish a day before account start")
        self.days.append(Daily(self.day, self.day_start, self.last_equity,
                               self.day_high, self.day_low, self.day_exposed,
                               self.day_fee))
        self.day = None

    def fund(self, event: FundingEvent, previous_close: tuple[float, float, float]) -> None:
        pos = self.position
        if pos is None:
            return
        index = SYMBOLS.index(event.symbol)
        if index == pos.long_index:
            pnl = -event.rate * pos.long_qty * previous_close[index]
        elif index == pos.short_index:
            pnl = event.rate * pos.short_qty * previous_close[index]
        else:
            return
        self.cash += pnl
        pos.funding_pnl += pnl
        self.total_funding += pnl
        self.touch(previous_close)

    def close(self, day: date, reference: tuple[float, float, float]) -> None:
        pos = self.position
        if pos is None:
            raise ReplayFailure("close without pair")
        long_exec = reference[pos.long_index] * (1 - self.slip)
        short_exec = reference[pos.short_index] * (1 + self.slip)
        long_notional = pos.long_qty * long_exec
        short_notional = pos.short_qty * short_exec
        if min(long_notional, short_notional) < 10:
            raise ReplayFailure(f"{day}: exit below ten USDT minimum")
        price_pnl = (pos.long_qty * (long_exec - pos.long_entry) +
                     pos.short_qty * (pos.short_entry - short_exec))
        fee = self.fee_rate * (long_notional + short_notional)
        self.cash += price_pnl - fee
        self.total_fees += fee
        self.day_fee += fee
        slip_drag = (pos.long_qty * reference[pos.long_index] +
                     pos.short_qty * reference[pos.short_index]) * self.slip
        self.total_slippage += slip_drag
        self.trades.append({"entry_day": str(pos.opened), "exit_day": str(day),
                            "long": SYMBOLS[pos.long_index],
                            "short": SYMBOLS[pos.short_index],
                            "long_quantity": pos.long_qty,
                            "short_quantity": pos.short_qty,
                            "entry_equity": pos.entry_equity,
                            "net_pair_pnl": self.cash - pos.entry_equity,
                            "price_pnl": price_pnl,
                            "funding_pnl": pos.funding_pnl,
                            "fee_pnl": pos.fee_pnl - fee,
                            "reference_slippage_drag": pos.slippage_drag + slip_drag})
        self.position = None
        self.touch(reference)

    def enter(self, day: date, reference: tuple[float, float, float],
              long_index: int, short_index: int,
              median_prior_volume: tuple[float, float, float]) -> None:
        if self.position is not None or long_index == short_index:
            raise ReplayFailure("invalid pair entry")
        equity = self.cash
        if not math.isfinite(equity) or equity <= 0:
            raise ReplayFailure("nonpositive entry equity")
        long_exec = reference[long_index] * (1 + self.slip)
        short_exec = reference[short_index] * (1 - self.slip)
        long_qty = math.floor((.5 * equity / long_exec) / LOT + 1e-9) * LOT
        short_qty = math.floor((.5 * equity / short_exec) / LOT + 1e-9) * LOT
        long_notional = long_qty * long_exec
        short_notional = short_qty * short_exec
        if min(long_notional, short_notional) < 10:
            raise ReplayFailure(f"{day}: entry below ten USDT minimum")
        if (long_notional > .01 * median_prior_volume[long_index] or
                short_notional > .01 * median_prior_volume[short_index]):
            raise ReplayFailure(f"{day}: one-percent liquidity capacity exceeded")
        fee = self.fee_rate * (long_notional + short_notional)
        self.cash -= fee
        self.total_fees += fee
        self.day_fee += fee
        slip_drag = (long_qty * reference[long_index] +
                     short_qty * reference[short_index]) * self.slip
        self.total_slippage += slip_drag
        self.position = _Position(day, long_index, short_index, long_qty, short_qty,
                                  long_exec, short_exec, equity, -fee,
                                  slippage_drag=slip_drag)
        self.touch(reference)

    def mark_hour(self, bar: HourBar) -> None:
        self.touch(bar.opening)
        pos = self.position
        if pos is not None:
            hi = list(bar.opening)
            lo = list(bar.opening)
            hi[pos.long_index] = bar.high[pos.long_index]
            hi[pos.short_index] = bar.low[pos.short_index]
            lo[pos.long_index] = bar.low[pos.long_index]
            lo[pos.short_index] = bar.high[pos.short_index]
            high_prices, low_prices = tuple(hi), tuple(lo)
            upper, lower = self.equity(high_prices), self.equity(low_prices)
            self._risk(lower, low_prices)
            self.day_high = max(self.day_high, upper)
            self.day_low = min(self.day_low, lower)
        self.touch(bar.close)
        self.last_equity = self.equity(bar.close)


def _funding_grid(events: Iterable[FundingEvent]) -> tuple[FundingEvent, ...]:
    ordered = tuple(events)
    anchors = int((END - START).total_seconds() // (8 * 3600)) + 1
    if len(ordered) != anchors * len(SYMBOLS):
        raise ReplayFailure("funding event count changed")
    for i in range(anchors):
        anchor = START + i * timedelta(hours=8)
        group = ordered[i * 3:i * 3 + 3]
        if ({event.symbol for event in group} != set(SYMBOLS) or
                any(not anchor <= event.time < anchor + timedelta(seconds=1)
                    or not math.isfinite(event.rate) for event in group)):
            raise ReplayFailure(f"funding event grid changed at {anchor}")
    return ordered


def _market_state(closes: deque[tuple[float, float, float]]) -> str:
    if len(closes) != 31:
        raise ReplayFailure("market regime lacks prior thirty-one completed closes")
    log_returns = [sum(math.log(now[i] / before[i]) for i in range(3)) / 3
                   for before, now in zip(list(closes)[:-1], list(closes)[1:])]
    trend = "bull" if sum(log_returns) > 0 else "bear"
    vol = "high" if float(np.std(log_returns, ddof=1)) * math.sqrt(365) >= .80 else "low"
    return f"{trend}_{vol}"


def run_replay(issuance: pd.DataFrame,
               bars: Iterable[HourBar | TerminalOpen],
               funding: Iterable[FundingEvent]) -> tuple[dict[str, _Account], dict[date, str], dict]:
    targets = weekly_rankings(issuance)
    events = _funding_grid(funding)
    accounts = {
        "net": _Account("net", SLIP, FEE, 0),
        "gross_no_cost": _Account("gross_no_cost", 0.0, 0.0, 0),
        "double_slippage": _Account("double_slippage", .002, FEE, 0),
        "one_day_latency": _Account("one_day_latency", SLIP, FEE, 1),
    }
    expected_time = INPUT_START
    previous_close: tuple[float, float, float] | None = None
    market_closes: deque[tuple[float, float, float]] = deque(maxlen=31)
    volumes: dict[str, deque[float]] = {s: deque(maxlen=30) for s in SYMBOLS}
    current_volume = [0.0, 0.0, 0.0]
    states: dict[date, str] = {}
    event_index = 0
    seen_terminal = False
    for bar in bars:
        if bar.time != expected_time:
            raise ReplayFailure(f"missing or duplicate hourly bar at {expected_time}")
        if isinstance(bar, TerminalOpen):
            if bar.time != END or previous_close is None or seen_terminal:
                raise ReplayFailure("invalid terminal BTC-equivalent boundary")
            while event_index < len(events) and events[event_index].time < END + timedelta(seconds=1):
                event = events[event_index]
                if not END <= event.time < END + timedelta(seconds=1):
                    raise ReplayFailure("invalid terminal funding timestamp")
                for account in accounts.values():
                    account.fund(event, previous_close)
                event_index += 1
            for account in accounts.values():
                account.touch(bar.opening)
                if account.position is not None:
                    account.close(END.date(), bar.opening)
                account.last_equity = account.cash
                account.finish_day()
            seen_terminal = True
            expected_time += timedelta(hours=1)
            break
        if not isinstance(bar, HourBar) or bar.time >= END:
            raise ReplayFailure("unexpected hourly item")
        time = bar.time
        expected_time += timedelta(hours=1)
        if time.hour == 0:
            if previous_close is not None:
                market_closes.append(previous_close)
            if time > INPUT_START:
                for i, symbol in enumerate(SYMBOLS):
                    volumes[symbol].append(current_volume[i])
                current_volume = [0.0, 0.0, 0.0]
            if time >= START:
                states[time.date()] = _market_state(market_closes)
                for account in accounts.values():
                    account.begin_day(time.date())
        for i in range(3):
            current_volume[i] += bar.quote_volume[i]
        while event_index < len(events) and events[event_index].time < time + timedelta(hours=1):
            event = events[event_index]
            if event.time < time or previous_close is None:
                raise ReplayFailure("funding event lacks completed prior hourly mark")
            for account in accounts.values():
                account.fund(event, previous_close)
            event_index += 1
        if time >= START:
            for account in accounts.values():
                account.touch(bar.opening)
                entry_day = time.date() - timedelta(days=account.latency_days)
                if time.hour == 0 and entry_day in targets:
                    pair = targets[entry_day]
                    old = account.position
                    old_pair = None if old is None else (old.long_index, old.short_index)
                    if old_pair != pair:
                        if old is not None:
                            account.close(time.date(), bar.opening)
                        if any(len(volumes[s]) != 30 for s in SYMBOLS):
                            raise ReplayFailure("entry lacks thirty completed volume days")
                        median_volume = tuple(float(median(volumes[s])) for s in SYMBOLS)
                        if any(not math.isfinite(v) or v <= 0 for v in median_volume):
                            raise ReplayFailure("prior volume capacity invalid")
                        account.enter(time.date(), bar.opening, *pair, median_volume)
                account.mark_hour(bar)
        previous_close = bar.close
    if not seen_terminal or event_index != len(events) or len(states) != 1826:
        raise ReplayFailure("incomplete five-year bars, funding or market states")
    if any(len(account.days) != 1826 or account.position is not None or
           not math.isclose(account.days[-1].end, account.cash, rel_tol=1e-10)
           for account in accounts.values()):
        raise ReplayFailure("terminal five-year account reconciliation failed")
    audit = {"weekly_decisions": len(targets),
             "funding_events": len(events), "hourly_formal_bars": 1826 * 24,
             "first_decision": str(min(targets)), "last_decision": str(max(targets))}
    return accounts, states, audit


def _bootstrap(rows: list[Daily]) -> dict:
    n = len(rows)
    factors = np.array([item.end / item.start for item in rows], dtype=float)
    high = np.array([item.high / item.start for item in rows], dtype=float)
    low = np.array([item.low / item.start for item in rows], dtype=float)
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    paths: list[tuple[float, float | None, float, float | None]] = []
    passes = ruined = undefined = 0
    for _ in range(MC_PATHS):
        starts = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        index = ((starts[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        f = factors[index]
        equity = INITIAL * np.cumprod(f)
        begins = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peaks = np.maximum.accumulate(np.r_[INITIAL, begins * high[index]])[1:]
        drawdown = float(np.max(1 - begins * low[index] / peaks))
        returns = f - 1
        sigma = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sigma * math.sqrt(365)) if sigma > 0 else None
        dollars = equity - begins
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        paths.append((cagr, sharpe, drawdown, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": drawdown, "profit_factor_daily_dollar": pf}):
            passes += 1

    def quantile(field: int) -> dict:
        vals = [item[field] for item in paths
                if item[field] is not None and math.isfinite(item[field])]
        return {f"p{q:02d}": float(np.percentile(vals, q)) if vals else None
                for q in (5, 50, 95)}

    return {"paths": MC_PATHS, "joint_passes": passes,
            "joint_pass_rate": passes / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0),
            "sharpe": quantile(1), "max_drawdown": quantile(2),
            "profit_factor": quantile(3)}


def evaluate(accounts: dict[str, _Account], states: dict[date, str], audit: dict) -> dict:
    expected = {"net", "gross_no_cost", "double_slippage", "one_day_latency"}
    if set(accounts) != expected:
        raise ReplayFailure("frozen account scenarios missing")
    net = accounts["net"]
    full = _numbers(net.days)
    annual = {str(y): _numbers([row for row in net.days if row.day.year == y])
              for y in range(2021, 2026)}
    holdout = _numbers([row for row in net.days if row.day.year in (2024, 2025)])
    months = _monthly(net.days)
    buckets: dict[str, dict] = {f"{direction}_{vol}": {"days": 0, "net_dollar_pnl": 0.0}
                                for direction in ("bull", "bear")
                                for vol in ("high", "low")}
    for day in net.days:
        bucket = buckets[states[day.day]]
        bucket["days"] += 1
        bucket["net_dollar_pnl"] += day.end - day.start
    mc = _bootstrap(net.days)
    entries_by_year = {str(y): sum(trade["entry_day"][:4] == str(y)
                                   for trade in net.trades)
                       for y in range(2021, 2026)}
    schedules = {name: [(t["entry_day"], t["exit_day"], t["long"], t["short"])
                        for t in account.trades] for name, account in accounts.items()}
    if any(schedules[name] != schedules["net"]
           for name in ("gross_no_cost", "double_slippage")):
        raise ReplayFailure("identical timing scenarios changed pair schedules")
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_pair_entries": len(net.trades) >= 20,
        "annual_walk_forward": all(item["end"] > item["start"] and
                                   item["sharpe"] is not None and item["sharpe"] > 0 and
                                   entries_by_year[y] > 0
                                   for y, item in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36 and
                             months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(item["days"] >= 90 and item["net_dollar_pnl"] > 0
                            for item in buckets.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": accounts["double_slippage"].cash > INITIAL,
        "one_day_latency": accounts["one_day_latency"].cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full,
            "gross_no_cost": _numbers(accounts["gross_no_cost"].days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": months, "regimes": buckets, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "inside_40_to_60_percent_target_band": .40 <= full["cagr"] <= .60,
            "completed_pairs": len(net.trades), "entries_by_year": entries_by_year,
            "fees_usdt": net.total_fees, "funding_usdt": net.total_funding,
            "reference_slippage_drag_usdt": net.total_slippage,
            "net_final_usdt": net.cash,
            "gross_final_usdt": accounts["gross_no_cost"].cash,
            "double_slippage_final_usdt": accounts["double_slippage"].cash,
            "one_day_latency_final_usdt": accounts["one_day_latency"].cash,
            "trades": net.trades, "execution_audit": audit}
