"""Frozen weekly mempool-demand rule and 10,000-USDT spot/cash replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import statistics

import numpy as np
import pandas as pd

from research.btc_exchange_supply_source import SourceFailure
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes


START = date(2021, 1, 1)
END = date(2026, 1, 1)
INITIAL = 10_000.0
FEE = .001
SLIP = .0005
LOT = .00001
MC_PATHS = 5_000
MC_BLOCK = 21
MC_SEED = 20261114


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    reference_slippage: float
    terminal_cash: float
    unavailable_mondays: tuple[str, ...]


def weekly_target(monday: date, medians: dict[date, float],
                  counts: dict[date, int]) -> tuple[int, bool]:
    if monday.weekday() != 0:
        raise ValueError("mempool decision must be Monday")
    friday = monday - timedelta(days=3)
    days = [friday - timedelta(days=i) for i in range(34, -1, -1)]
    if any(counts.get(day, 0) < 90 for day in days):
        return 0, True
    values = [medians[day] for day in days]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise SourceFailure("mempool daily median invalid")
    baseline = statistics.median(values[:28])
    recent = statistics.median(values[28:])
    ratio = (1 + recent / 1_000_000) / (1 + baseline / 1_000_000)
    return int(ratio >= 1.5), False


def decision_schedule(medians: dict[date, float], counts: dict[date, int],
                      latency_days: int) -> tuple[dict[date, int], tuple[str, ...]]:
    if latency_days not in (0, 1):
        raise ValueError("invalid fixed latency")
    schedule: dict[date, int] = {}
    unavailable: list[str] = []
    eligible_years = {str(y): 0 for y in range(2021, 2026)}
    for ts in pd.date_range(date(2021, 1, 4), END - timedelta(days=1),
                            tz="UTC", freq="W-MON"):
        monday = ts.date()
        target, missing = weekly_target(monday, medians, counts)
        if missing:
            unavailable.append(str(monday))
        else:
            eligible_years[str(monday.year)] += 1
        execution = monday + timedelta(days=latency_days)
        if execution in schedule or not START <= execution < END:
            raise SourceFailure("duplicate/out-of-window mempool execution")
        schedule[execution] = target  # Unavailable means cash, never a skipped year.
    if (len(schedule) != 261 or len(unavailable) != 17
            or eligible_years != {"2021": 52, "2022": 47, "2023": 52,
                                  "2024": 41, "2025": 52}):
        raise SourceFailure("mempool date-only decision coverage changed")
    return schedule, tuple(unavailable)


def _buy(cash: float, reference: float, slippage: float,
         fee_rate: float) -> tuple[float, float, float, float]:
    execution = reference * (1 + slippage)
    qty = math.floor(cash / (execution * (1 + fee_rate)) / LOT + 1e-9) * LOT
    notional = qty * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if qty <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("mempool buy failed lot/notional/cash check")
    return qty, notional, fee, max(residual, 0.0)


def simulate(medians: dict[date, float], counts: dict[date, int],
             prices: pd.DataFrame, terminal_open: float,
             *, slippage: float = SLIP, latency_days: int = 0,
             fee_rate: float = FEE, buy_and_hold: bool = False) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen execution scenario")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("invalid terminal BTC spot open")
    expected = pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")
    if not prices.index.equals(expected):
        raise SourceFailure("BTC spot daily grid changed in simulator")
    schedule, unavailable = decision_schedule(medians, counts, latency_days)
    cash, qty = INITIAL, 0.0
    fees, slippage_drag = 0.0, 0.0
    fills: list[dict] = []
    days: list[Daily] = []
    prior_equity = INITIAL
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = prices.loc[ts]
        day_fee = 0.0
        trade_day = day == START if buy_and_hold else day in schedule
        if trade_day:
            target = 1 if buy_and_hold else schedule[day]
            if target != int(qty > 0):
                reference = float(row["open"])
                if target:
                    new_qty, notional, fee, cash = _buy(cash, reference, slippage, fee_rate)
                    past = prices.loc[pd.date_range(day - timedelta(days=30),
                                                    day - timedelta(days=1),
                                                    tz="UTC", freq="D"), "quote_volume"]
                    if len(past) != 30 or not np.isfinite(past.to_numpy()).all():
                        raise SourceFailure("missing prior 30 completed BTC volume days")
                    if float(past.median()) <= 0 or notional > .01 * float(past.median()):
                        raise SourceFailure(f"{day}: 1% spot capacity Gate failed")
                    qty = new_qty
                    execution = reference * (1 + slippage)
                    slippage_drag += qty * reference * slippage
                    side = "buy"
                else:
                    execution = reference * (1 - slippage)
                    notional = qty * execution
                    if notional < 10:
                        raise SourceFailure(f"{day}: sell below 10 USDT")
                    fee = notional * fee_rate
                    cash += notional - fee
                    slippage_drag += qty * reference * slippage
                    new_qty = qty
                    qty = 0.0
                    side = "sell"
                fees += fee
                day_fee += fee
                fills.append({"day": str(day), "side": side, "reference": reference,
                              "execution": execution, "quantity": new_qty,
                              "notional": notional, "fee": fee})
        end = cash + qty * float(row["close"])
        high = max(prior_equity, cash + qty * float(row["high"]), end)
        low = min(prior_equity, cash + qty * float(row["low"]), end)
        if not all(math.isfinite(v) and v > 0 for v in (prior_equity, end, high, low)):
            raise SourceFailure(f"{day}: invalid/ruined account equity")
        days.append(Daily(day, prior_equity, end, high, low, int(qty > 0), day_fee))
        prior_equity = end
    if qty:
        execution = terminal_open * (1 - slippage)
        notional = qty * execution
        if notional < 10:
            raise SourceFailure("terminal sell below 10 USDT")
        fee = notional * fee_rate
        cash += notional - fee
        fees += fee
        slippage_drag += qty * terminal_open * slippage
        fills.append({"day": str(END), "side": "sell", "reference": terminal_open,
                      "execution": execution, "quantity": qty,
                      "notional": notional, "fee": fee, "forced_terminal": True})
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash, max(last.high, cash),
                         min(last.low, cash), last.position, last.fee + fee)
    else:
        cash = prior_equity
    if len(days) != 1826 or cash <= 0 or abs(days[-1].end - cash) > 1e-7:
        raise SourceFailure("terminal account and daily factors do not reconcile")
    return Simulation(tuple(days), tuple(fills), fees, slippage_drag, cash, unavailable)


def _bootstrap(rows: tuple[Daily, ...]) -> dict:
    n = len(rows)
    factors = np.array([r.end / r.start for r in rows])
    highs = np.array([r.high / r.start for r in rows])
    lows = np.array([r.low / r.start for r in rows])
    rng = np.random.Generator(np.random.PCG64(MC_SEED))
    metrics: list[tuple[float, float | None, float, float | None]] = []
    passes = ruined = undefined = 0
    for _ in range(MC_PATHS):
        starts = rng.integers(0, n, size=math.ceil(n / MC_BLOCK))
        idx = ((starts[:, None] + np.arange(MC_BLOCK)) % n).ravel()[:n]
        equity = INITIAL * np.cumprod(factors[idx])
        beginning = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peaks = np.maximum.accumulate(np.r_[INITIAL, beginning * highs[idx]])[1:]
        drawdown = float(np.max(1 - beginning * lows[idx] / peaks))
        returns = factors[idx] - 1
        sd = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / sd * math.sqrt(365)) if sd > 0 else None
        dollars = equity - beginning
        loss = float(-dollars[dollars < 0].sum())
        pf = float(dollars[dollars > 0].sum() / loss) if loss > 0 else None
        cagr = float((equity[-1] / INITIAL) ** (365.25 / n) - 1)
        metrics.append((cagr, sharpe, drawdown, pf))
        if sharpe is None or pf is None:
            undefined += 1
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": drawdown, "profit_factor_daily_dollar": pf}):
            passes += 1
    def quantile(field: int) -> dict:
        values = [m[field] for m in metrics if m[field] is not None and math.isfinite(m[field])]
        return {f"p{q:02d}": float(np.percentile(values, q)) if values else None
                for q in (5, 50, 95)}
    return {"paths": MC_PATHS, "joint_passes": passes, "joint_pass_rate": passes / MC_PATHS,
            "ruined": ruined, "undefined": undefined,
            "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             delay: Simulation, hold: Simulation, prices: pd.DataFrame) -> dict:
    full = _numbers(primary.days)
    annual = {str(year): _numbers([r for r in primary.days if r.day.year == year])
              for year in range(2021, 2026)}
    holdout = _numbers([r for r in primary.days if r.day.year in (2024, 2025)])
    months = _monthly(primary.days)
    regimes = _regimes(primary.days, prices)
    mc = _bootstrap(primary.days)
    buys = sum(fill["side"] == "buy" for fill in primary.fills)
    fills_by_year = {str(y): sum(fill["day"][:4] == str(y) for fill in primary.fills)
                     for y in range(2021, 2026)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(v["end"] > v["start"] and v["sharpe"] is not None
                                   and v["sharpe"] > 0 and fills_by_year[y] > 0
                                   for y, v in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36
                             and months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(v["days"] >= 90 and v["net_dollar_pnl"] > 0
                            for v in regimes.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_day_latency": delay.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "btc_buy_and_hold": _numbers(hold.days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": months, "regimes": regimes, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "buys": buys, "fills_by_year": fills_by_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "slippage_reference_drag_usdt": primary.reference_slippage,
            "unavailable_mempool_mondays": primary.unavailable_mondays,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_day_latency_final_usdt": delay.terminal_cash}
