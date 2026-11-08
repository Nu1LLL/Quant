"""Frozen post-funding shock fade; event-clock futures account and full Gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes
from research.funding_settlement_fade_source import (
    DAY_BARS, EVAL_DAYS, SYMBOLS, WARMUP_BARS, SourceFailure, Tape,
)


INITIAL = 10_000.0
FEE = .0005
SLIP = .0005
MC_PATHS = 5_000
MC_BLOCK = 21
MC_SEED = 20261116
LOOKBACK = 30 * DAY_BARS


@dataclass(frozen=True)
class Event:
    day: date
    bar: int
    side: dict[str, int]
    capacity: dict[str, float]


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    slippage_drag: float
    terminal_cash: float
    event_entries: int
    fills_by_year: dict[str, int]


def _validate_tape(tape: Tape) -> None:
    total = WARMUP_BARS + EVAL_DAYS * DAY_BARS
    for symbol in SYMBOLS:
        arr = tape.prices.get(symbol)
        if (arr is None or arr.shape != (total, 5) or
                not np.isfinite(arr).all() or (arr[:, :4] <= 0).any()
                or (arr[:, 4] < 0).any()):
            raise SourceFailure(f"{symbol}: simulator tape invalid")
        if len(tape.funding_index.get(symbol, ())) != EVAL_DAYS * 3:
            raise SourceFailure(f"{symbol}: simulator funding clock invalid")


def plan_events(tape: Tape) -> tuple[Event, ...]:
    """Compute signal thresholds only from completed pre-event bars."""
    _validate_tape(tape)
    out = []
    first = date(2021, 1, 1)
    for day_number in range(EVAL_DAYS):
        day = first + timedelta(days=day_number)
        for slot in range(3):
            i = WARMUP_BARS + day_number * DAY_BARS + slot * 96
            sides: dict[str, int] = {}
            capacity: dict[str, float] = {}
            for symbol in SYMBOLS:
                if i not in tape.funding_index[symbol]:
                    raise SourceFailure(f"{symbol} {day}: unverified settlement event")
                arr = tape.prices[symbol]
                prior = arr[i - LOOKBACK:i]
                if prior.shape != (LOOKBACK, 5):
                    raise SourceFailure(f"{symbol} {day}: missing rolling window")
                moves = np.abs(np.log(prior[:, 3] / prior[:, 0]))
                median = float(np.median(moves))
                volume_median = float(np.median(prior[:, 4]))
                if (not math.isfinite(median) or median <= 0 or
                        not math.isfinite(volume_median) or volume_median <= 0):
                    raise SourceFailure(f"{symbol} {day}: invalid prior median")
                impulse = math.log(arr[i, 3] / arr[i, 0])
                sides[symbol] = (-1 if impulse > 0 else 1) if (
                    impulse != 0 and abs(impulse) >= 3 * median) else 0
                capacity[symbol] = .01 * volume_median
            out.append(Event(day, i, sides, capacity))
    if len(out) != EVAL_DAYS * 3:
        raise SourceFailure("settlement event plan incomplete")
    return tuple(out)


def _position_equity(before: float, entry_fees: float, positions: list[dict],
                     prices: dict[str, float]) -> float:
    return before - entry_fees + sum(
        p["side"] * p["quantity"] * (prices[p["symbol"]] - p["entry_exec"])
        for p in positions
    )


def simulate(tape: Tape, events: tuple[Event, ...], *, fee_rate: float = FEE,
             slippage: float = SLIP, latency_bars: int = 0) -> Simulation:
    if (fee_rate < 0 or slippage < 0 or latency_bars not in (0, 1)
            or len(events) != EVAL_DAYS * 3):
        raise ValueError("invalid frozen simulation scenario")
    equity = INITIAL
    all_fills: list[dict] = []
    days: list[Daily] = []
    fees = slippage_drag = 0.0
    event_entries = 0
    fills_by_year = {str(y): 0 for y in range(2021, 2026)}
    for day_number in range(EVAL_DAYS):
        day_events = events[day_number * 3:(day_number + 1) * 3]
        day_start = equity
        day_high = day_low = equity
        day_fees = 0.0
        active_today = False
        for event in day_events:
            if event.day != date(2021, 1, 1) + timedelta(days=day_number):
                raise SourceFailure("event date order changed")
            entry = event.bar + 2 + latency_bars
            exit_bar = entry + 12
            positions = []
            before = equity
            for symbol in SYMBOLS:
                side = event.side[symbol]
                if side == 0:
                    continue
                if any(entry <= t < exit_bar for t in tape.funding_index[symbol]):
                    raise SourceFailure(f"{symbol} {event.day}: funding while position open")
                ref = float(tape.prices[symbol][entry, 0])
                execution = ref * (1 + side * slippage)
                q = math.floor((.5 * before / max(ref, execution)) / .001 + 1e-10) * .001
                ref_notional = q * ref
                if q <= 0 or ref_notional < 10:
                    raise SourceFailure(f"{symbol} {event.day}: order below 10 USDT")
                if ref_notional > event.capacity[symbol] + 1e-8:
                    raise SourceFailure(f"{symbol} {event.day}: 1% quote-volume capacity failed")
                fee = q * execution * fee_rate
                positions.append({"symbol": symbol, "side": side, "quantity": q,
                                  "entry_exec": execution, "entry_fee": fee,
                                  "entry_ref_notional": ref_notional})
                all_fills.append({"day": str(event.day), "bar": entry, "symbol": symbol,
                                  "side": "buy" if side == 1 else "sell",
                                  "reference": ref, "execution": execution,
                                  "quantity": q, "notional": q * execution,
                                  "fee": fee, "slippage_drag": q * ref * slippage})
                fees += fee
                day_fees += fee
                slippage_drag += q * ref * slippage
            if not positions:
                continue
            active_today = True
            event_entries += 1
            fills_by_year[str(event.day.year)] += len(positions) * 2
            gross = sum(p["entry_ref_notional"] for p in positions)
            executed_gross = sum(p["quantity"] * p["entry_exec"] for p in positions)
            if gross > before + 1e-7 or executed_gross > before + 1e-7:
                raise SourceFailure(f"{event.day}: gross leverage exceeded 1x")
            entry_fees = sum(p["entry_fee"] for p in positions)
            start_mark = _position_equity(
                before, entry_fees, positions,
                {p["symbol"]: float(tape.prices[p["symbol"]][entry, 0])
                 for p in positions})
            day_high = max(day_high, start_mark)
            day_low = min(day_low, start_mark)
            for bar in range(entry, exit_bar):
                high_prices = {}
                low_prices = {}
                close_prices = {}
                adverse_gross = 0.0
                for p in positions:
                    o, hi, lo, close, _ = tape.prices[p["symbol"]][bar]
                    symbol = p["symbol"]
                    high_prices[symbol] = float(hi if p["side"] == 1 else lo)
                    low_prices[symbol] = float(lo if p["side"] == 1 else hi)
                    close_prices[symbol] = float(close)
                    adverse_gross += p["quantity"] * low_prices[symbol]
                mark_high = _position_equity(before, entry_fees, positions, high_prices)
                mark_low = _position_equity(before, entry_fees, positions, low_prices)
                mark_close = _position_equity(before, entry_fees, positions, close_prices)
                if (not all(math.isfinite(x) for x in (mark_high, mark_low, mark_close))
                        or mark_low <= 0 or mark_low <= .05 * adverse_gross):
                    raise SourceFailure(f"{event.day}: conservative margin proxy failed")
                day_high = max(day_high, mark_high, mark_close)
                day_low = min(day_low, mark_low, mark_close)
            exit_pnl = 0.0
            exit_fees = 0.0
            for p in positions:
                symbol, side, q = p["symbol"], p["side"], p["quantity"]
                ref = float(tape.prices[symbol][exit_bar, 0])
                execution = ref * (1 - side * slippage)
                exit_fee = q * execution * fee_rate
                exit_fees += exit_fee
                exit_pnl += side * q * (execution - p["entry_exec"])
                all_fills.append({"day": str(event.day), "bar": exit_bar,
                                  "symbol": symbol,
                                  "side": "sell" if side == 1 else "buy",
                                  "reference": ref, "execution": execution,
                                  "quantity": q, "notional": q * execution,
                                  "fee": exit_fee,
                                  "slippage_drag": q * ref * slippage})
                fees += exit_fee
                day_fees += exit_fee
                slippage_drag += q * ref * slippage
            equity = before + exit_pnl - entry_fees - exit_fees
            if not math.isfinite(equity) or equity <= 0:
                raise SourceFailure(f"{event.day}: account ruined")
            day_high = max(day_high, equity)
            day_low = min(day_low, equity)
        days.append(Daily(day_events[0].day, day_start, equity, day_high,
                          day_low, int(active_today), day_fees))
    if (len(days) != EVAL_DAYS or abs(days[-1].end - equity) > 1e-7
            or not math.isfinite(equity)):
        raise SourceFailure("daily account reconciliation failed")
    return Simulation(tuple(days), tuple(all_fills), fees, slippage_drag,
                      equity, event_entries, fills_by_year)


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factors = np.array([r.end / r.start for r in rows])
    highs = np.array([r.high / r.start for r in rows])
    lows = np.array([r.low / r.start for r in rows])
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    metrics: list[tuple[float, float | None, float, float | None]] = []
    passed = ruined = undefined = 0
    for _ in range(MC_PATHS):
        starts = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        idx = ((starts[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        factor = factors[idx]
        equity = INITIAL * np.cumprod(factor)
        begin = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peak = np.maximum.accumulate(np.r_[INITIAL, begin * highs[idx]])[1:]
        drawdown = float(np.max(1 - begin * lows[idx] / peak))
        ret = factor - 1
        sigma = float(np.std(ret, ddof=1))
        sharpe = float(np.mean(ret) / sigma * math.sqrt(365)) if sigma > 0 else None
        dollars = equity - begin
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        metrics.append((cagr, sharpe, drawdown, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe, "max_drawdown": drawdown,
                          "profit_factor_daily_dollar": pf}):
            passed += 1

    def quantiles(column: int) -> dict:
        values = [m[column] for m in metrics
                  if m[column] is not None and math.isfinite(m[column])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}

    return {"paths": MC_PATHS, "block_days": MC_BLOCK, "seed": MC_SEED,
            "joint_passes": passed, "joint_pass_rate": passed / MC_PATHS,
            "ruined": ruined, "undefined": undefined,
            "cagr": quantiles(0), "sharpe": quantiles(1),
            "max_drawdown": quantiles(2), "profit_factor": quantiles(3)}


def evaluate(tape: Tape, primary: Simulation, gross: Simulation,
             double_slip: Simulation, delayed: Simulation) -> dict:
    full = _numbers(primary.days)
    annual = {str(y): _numbers([r for r in primary.days if r.day.year == y])
              for y in range(2021, 2026)}
    holdout = _numbers([r for r in primary.days if r.day.year in (2024, 2025)])
    monthly = _monthly(primary.days)
    btc_closes = tape.prices["BTCUSDT"][DAY_BARS - 1::DAY_BARS, 3]
    dates = pd.date_range("2020-12-01", "2025-12-31", freq="D", tz="UTC")
    if len(btc_closes) != len(dates):
        raise SourceFailure("BTC daily regime grid incomplete")
    btc = pd.DataFrame({"close": btc_closes}, index=dates)
    regimes = _regimes(primary.days, btc)
    mc = _bootstrap(primary.days)
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_100_event_entries": primary.event_entries >= 100,
        "annual_walk_forward": all(
            x["end"] > x["start"] and x["sharpe"] is not None and x["sharpe"] > 0
            and x["profit_factor_daily_dollar"] is not None
            and x["profit_factor_daily_dollar"] > 1
            and primary.fills_by_year[year] > 0
            for year, x in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": monthly["months"] == 60 and monthly["positive"] >= 36
                             and monthly["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(x["days"] >= 90 and x["net_dollar_pnl"] > 0
                            for x in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_bar_latency": delayed.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": monthly, "regimes": regimes, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "event_entries": primary.event_entries,
            "fills_by_year": primary.fills_by_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "slippage_reference_drag_usdt": primary.slippage_drag,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_bar_latency_final_usdt": delayed.terminal_cash}
