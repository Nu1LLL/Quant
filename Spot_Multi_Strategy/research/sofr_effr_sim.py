"""Weekly frozen SOFR-minus-EFFR easing rule and complete spot account Gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes
from research.ww_cpi_disinflation_sim import _buy


START = date(2021, 1, 1)
END = date(2026, 1, 1)
INITIAL = 10_000.0
FEE = .001
SLIP = .0005
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261119


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    reference_slippage: float
    terminal_cash: float
    unavailable_mondays: tuple[str, ...]


def _target(monday: date, sofr: dict[date, float | None],
            effr: dict[date, float | None]) -> int | None:
    if monday.weekday() != 0:
        raise ValueError("decision day is not a UTC Monday")
    cutoff = monday - timedelta(days=7)
    common = sorted(day for day in sofr.keys() & effr.keys()
                    if day <= cutoff and sofr[day] is not None and effr[day] is not None)
    recent_dates = common[-20:]
    if (len(recent_dates) != 20 or (cutoff - recent_dates[-1]).days > 5
            or (recent_dates[-1] - recent_dates[0]).days > 40):
        return None
    spreads = []
    for day in recent_dates:
        a, b = sofr[day], effr[day]
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            raise SourceFailure("invalid matched rate observation")
        spreads.append(a - b)
    return int(sum(spreads[10:]) / 10 < sum(spreads[:10]) / 10)


def event_schedule(sofr: dict[date, float | None], effr: dict[date, float | None],
                   latency_days: int) -> tuple[dict[date, int], tuple[str, ...]]:
    if latency_days not in (0, 1):
        raise ValueError("invalid frozen latency")
    schedule = {}
    unavailable = []
    mondays = pd.date_range("2021-01-04", "2025-12-29", freq="W-MON", tz="UTC")
    if len(mondays) != 261:
        raise SourceFailure("weekly decision count changed")
    for timestamp in mondays:
        monday = timestamp.date()
        target = _target(monday, sofr, effr)
        if target is None:
            unavailable.append(str(monday))
            continue
        execution_day = monday + timedelta(days=latency_days)
        if execution_day in schedule or not START <= execution_day < END:
            raise SourceFailure("duplicate/out-of-sample repo-spread execution day")
        schedule[execution_day] = target
    return schedule, tuple(unavailable)


def simulate(sofr: dict[date, float | None], effr: dict[date, float | None],
             prices: pd.DataFrame, terminal_open: float,
             *, slippage: float = SLIP, latency_days: int = 0,
             fee_rate: float = FEE, buy_and_hold: bool = False) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen scenario")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("invalid BTC terminal open")
    if not prices.index.equals(pd.date_range("2020-12-01", "2025-12-31",
                                              tz="UTC", freq="D")):
        raise SourceFailure("BTC spot price day grid changed in simulator")
    schedule, unavailable = event_schedule(sofr, effr, latency_days)
    cash, qty = INITIAL, 0.0
    fees, slip_drag = 0.0, 0.0
    fills = []
    days = []
    prior_equity = INITIAL
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = prices.loc[ts]
        day_fee = 0.0
        trade_day = (day == START) if buy_and_hold else day in schedule
        if trade_day:
            target = 1 if buy_and_hold else schedule[day]
            if target != int(qty > 0):
                ref = float(row["open"])
                if target:
                    new_qty, notional, fee, residual = _buy(cash, ref, slippage, fee_rate)
                    prior = prices.loc[pd.date_range(day - timedelta(days=30),
                                                    day - timedelta(days=1),
                                                    tz="UTC", freq="D"), "quote_volume"]
                    if len(prior) != 30 or not np.isfinite(prior.to_numpy()).all():
                        raise SourceFailure(f"{day}: missing 30 completed quote-volume days")
                    median_volume = float(prior.median())
                    if median_volume <= 0 or notional > .01 * median_volume:
                        raise SourceFailure(f"{day}: 1% BTC spot capacity Gate failed")
                    qty, cash = new_qty, residual
                    execution = ref * (1 + slippage)
                    slip_drag += qty * ref * slippage
                    side = "buy"
                else:
                    execution = ref * (1 - slippage)
                    notional = qty * execution
                    if notional < 10:
                        raise SourceFailure(f"{day}: BTC sale below 10 USDT")
                    fee = notional * fee_rate
                    cash += notional - fee
                    slip_drag += qty * ref * slippage
                    new_qty = qty
                    qty = 0.0
                    side = "sell"
                fees += fee
                day_fee += fee
                fills.append({"day": str(day), "side": side, "reference": ref,
                              "execution": execution, "quantity": new_qty,
                              "notional": notional, "fee": fee})
        end_equity = cash + qty * float(row["close"])
        high_equity = max(prior_equity, cash + qty * float(row["high"]), end_equity)
        low_equity = min(prior_equity, cash + qty * float(row["low"]), end_equity)
        if not all(math.isfinite(x) and x > 0 for x in
                   (prior_equity, end_equity, high_equity, low_equity)):
            raise SourceFailure(f"{day}: invalid spot account equity")
        if cash < -1e-7 or qty < 0 or qty * float(row["close"]) > end_equity + 1e-7:
            raise SourceFailure(f"{day}: spot cash/exposure violation")
        days.append(Daily(day, prior_equity, end_equity, high_equity, low_equity,
                          int(qty > 0), day_fee))
        prior_equity = end_equity
    if qty:
        execution = terminal_open * (1 - slippage)
        notional = qty * execution
        if notional < 10:
            raise SourceFailure("terminal BTC sale below 10 USDT")
        fee = notional * fee_rate
        cash += notional - fee
        fees += fee
        slip_drag += qty * terminal_open * slippage
        fills.append({"day": str(END), "side": "sell", "reference": terminal_open,
                      "execution": execution, "quantity": qty,
                      "notional": notional, "fee": fee, "forced_terminal": True})
        final = days[-1]
        days[-1] = Daily(final.day, final.start, cash, max(final.high, cash),
                         min(final.low, cash), final.position, final.fee + fee)
    else:
        cash = prior_equity
    if len(days) != 1826 or cash <= 0 or abs(days[-1].end - cash) > 1e-7:
        raise SourceFailure("BTC repo-spread account terminal mismatch")
    return Simulation(tuple(days), tuple(fills), fees, slip_drag, cash, unavailable)


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factors = np.array([r.end / r.start for r in rows])
    high = np.array([r.high / r.start for r in rows])
    low = np.array([r.low / r.start for r in rows])
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    metrics: list[tuple[float, float | None, float, float | None]] = []
    passes = ruined = undefined = 0
    for _ in range(MC_PATHS):
        blocks = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        idx = ((blocks[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        daily = factors[idx]
        equity = INITIAL * np.cumprod(daily)
        begins = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peaks = np.maximum.accumulate(np.r_[INITIAL, begins * high[idx]])[1:]
        dd = float(np.max(1 - begins * low[idx] / peaks))
        returns = daily - 1
        sd = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sd * math.sqrt(365)) if sd > 0 else None
        dollars = equity - begins
        losses = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / losses) if losses > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        metrics.append((cagr, sharpe, dd, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": dd, "profit_factor_daily_dollar": pf}):
            passes += 1
    def quantile(field: int) -> dict:
        values = [x[field] for x in metrics if x[field] is not None and math.isfinite(x[field])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}
    return {"paths": MC_PATHS, "joint_passes": passes,
            "joint_pass_rate": passes / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             delayed: Simulation, hold: Simulation, prices: pd.DataFrame) -> dict:
    full = _numbers(primary.days)
    annual = {str(year): _numbers([r for r in primary.days if r.day.year == year])
              for year in range(2021, 2026)}
    holdout = _numbers([r for r in primary.days if r.day.year in (2024, 2025)])
    months = _monthly(primary.days)
    states = _regimes(primary.days, prices)
    mc = _bootstrap(primary.days)
    buys = sum(fill["side"] == "buy" for fill in primary.fills)
    fills_by_year = {str(year): sum(fill["day"][:4] == str(year)
                                   for fill in primary.fills)
                     for year in range(2021, 2026)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(stats["end"] > stats["start"] and
                                   stats["sharpe"] is not None and stats["sharpe"] > 0
                                   and fills_by_year[year] > 0
                                   for year, stats in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36
                             and months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(x["days"] >= 90 and x["net_dollar_pnl"] > 0
                            for x in states.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .8,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_day_latency": delayed.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "btc_buy_and_hold": _numbers(hold.days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": months, "regimes": states, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "buys": buys, "fills_by_year": fills_by_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "reference_slippage_usdt": primary.reference_slippage,
            "unavailable_mondays": primary.unavailable_mondays,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_day_latency_final_usdt": delayed.terminal_cash}
