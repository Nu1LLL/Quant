"""Frozen U.S. initial-claims labor signal and BTC spot/cash replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

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
MC_SEED = 20261113


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    slippage_drag: float
    terminal_cash: float


def claims_target(week: date, claims: dict[date, int | None]) -> int:
    if week.weekday() != 5:
        raise SourceFailure("ICSA reference date is not a Saturday")
    recent = [claims.get(week - timedelta(weeks=k)) for k in range(4)]
    prior = [claims.get(week - timedelta(weeks=13 + k)) for k in range(4)]
    if any(type(v) is not int or v <= 0 for v in recent + prior):
        raise SourceFailure(f"ICSA source gap or nonpositive claim at {week}")
    return int(sum(recent) < sum(prior))


def signal_schedule(claims: dict[date, int | None], *, latency_weeks: int = 0
                    ) -> dict[date, int]:
    if latency_weeks not in (0, 1):
        raise ValueError("invalid frozen weekly execution delay")
    schedule: dict[date, int] = {}
    for week in sorted(claims):
        day = week + timedelta(days=9 + 7 * latency_weeks)
        if not START <= day < END:
            continue
        if day.weekday() != 0 or day in schedule:
            raise SourceFailure("ICSA Monday schedule is duplicated or misaligned")
        schedule[day] = claims_target(week, claims)
    if len(schedule) < 250:
        raise SourceFailure("ICSA five-year weekly schedule incomplete")
    return schedule


def _buy(cash: float, reference: float, slippage: float, fee_rate: float
         ) -> tuple[float, float, float, float]:
    execution = reference * (1 + slippage)
    raw_quantity = cash / (execution * (1 + fee_rate))
    quantity = math.floor(raw_quantity / LOT + 1e-9) * LOT
    notional = quantity * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if quantity <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("BTC buy failed lot/notional/cash checks")
    return quantity, notional, fee, max(residual, 0.0)


def simulate(claims: dict[date, int | None], prices: pd.DataFrame,
             terminal_open: float, *, slippage: float = SLIP,
             latency_weeks: int = 0, fee_rate: float = FEE,
             buy_and_hold: bool = False) -> Simulation:
    if latency_weeks not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen stress scenario")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("invalid terminal BTC spot open")
    if not prices.index.equals(pd.date_range(date(2020, 12, 1),
                                              date(2025, 12, 31), tz="UTC", freq="D")):
        raise SourceFailure("BTC spot price grid invalid in simulator")
    schedule = signal_schedule(claims, latency_weeks=latency_weeks)
    cash, quantity, prior_equity = INITIAL, 0.0, INITIAL
    total_fees = total_slippage = 0.0
    fills: list[dict] = []
    days: list[Daily] = []
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = prices.loc[ts]
        day_fee = 0.0
        if (day == START if buy_and_hold else day in schedule):
            target = 1 if buy_and_hold else schedule[day]
            if target != int(quantity > 0):
                reference = float(row["open"])
                if target == 1:
                    new_quantity, notional, fee, residual = _buy(
                        cash, reference, slippage, fee_rate)
                    prior_volume = prices.loc[pd.date_range(
                        day - timedelta(days=30), day - timedelta(days=1),
                        tz="UTC", freq="D"), "quote_volume"]
                    if len(prior_volume) != 30 or not np.isfinite(prior_volume.to_numpy()).all():
                        raise SourceFailure("missing prior 30 completed quote-volume days")
                    if (float(prior_volume.median()) <= 0 or
                            notional > .01 * float(prior_volume.median())):
                        raise SourceFailure(f"{day}: 1% size-capacity Gate failed")
                    quantity, cash = new_quantity, residual
                    execution = reference * (1 + slippage)
                    total_slippage += quantity * reference * slippage
                    side = "buy"
                else:
                    execution = reference * (1 - slippage)
                    notional = quantity * execution
                    if notional < 10:
                        raise SourceFailure(f"{day}: sale below 10 USDT proxy")
                    fee = notional * fee_rate
                    cash += notional - fee
                    total_slippage += quantity * reference * slippage
                    new_quantity = quantity
                    quantity = 0.0
                    side = "sell"
                total_fees += fee
                day_fee += fee
                fills.append({"day": str(day), "side": side, "reference": reference,
                              "execution": execution, "quantity": new_quantity,
                              "notional": notional, "fee": fee})
        end = cash + quantity * float(row["close"])
        high = max(prior_equity, cash + quantity * float(row["high"]), end)
        low = min(prior_equity, cash + quantity * float(row["low"]), end)
        if not all(math.isfinite(v) and v > 0 for v in (prior_equity, end, high, low)):
            raise SourceFailure(f"{day}: nonfinite or ruined spot account")
        if quantity * float(row["close"]) > end + 1e-7:
            raise SourceFailure(f"{day}: gross leverage exceeds 1x")
        days.append(Daily(day, prior_equity, end, high, low, int(quantity > 0), day_fee))
        prior_equity = end
    if quantity:
        execution = terminal_open * (1 - slippage)
        notional = quantity * execution
        if notional < 10:
            raise SourceFailure("terminal BTC sale below 10 USDT proxy")
        fee = notional * fee_rate
        cash += notional - fee
        total_fees += fee
        total_slippage += quantity * terminal_open * slippage
        fills.append({"day": str(END), "side": "sell", "reference": terminal_open,
                      "execution": execution, "quantity": quantity,
                      "notional": notional, "fee": fee, "forced_terminal": True})
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash, max(last.high, cash),
                         min(last.low, cash), last.position, last.fee + fee)
    else:
        cash = prior_equity
    if len(days) != 1826 or cash <= 0 or abs(days[-1].end - cash) > 1e-7:
        raise SourceFailure("terminal cash and daily factors do not reconcile")
    return Simulation(tuple(days), tuple(fills), total_fees, total_slippage, cash)


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
        factor = factors[idx]
        equity = INITIAL * np.cumprod(factor)
        begin = np.r_[INITIAL, equity[:-1]]
        if not np.isfinite(equity).all() or (equity <= 0).any():
            ruined += 1
            continue
        peaks = np.maximum.accumulate(np.r_[INITIAL, begin * highs[idx]])[1:]
        drawdown = float(np.max(1 - begin * lows[idx] / peaks))
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
        if _numeric_pass({"cagr": cagr, "sharpe": sharpe,
                          "max_drawdown": drawdown,
                          "profit_factor_daily_dollar": pf}):
            passes += 1

    def quantiles(column: int) -> dict:
        vals = [row[column] for row in metrics if row[column] is not None
                and math.isfinite(row[column])]
        return {f"p{q:02d}": float(np.percentile(vals, q)) if vals else None
                for q in (5, 50, 95)}

    return {"paths": MC_PATHS, "joint_passes": passes,
            "joint_pass_rate": passes / MC_PATHS, "ruined": ruined,
            "undefined": undefined, "cagr": quantiles(0),
            "sharpe": quantiles(1), "max_drawdown": quantiles(2),
            "profit_factor": quantiles(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             delayed: Simulation, hold: Simulation, prices: pd.DataFrame) -> dict:
    full = _numbers(primary.days)
    annual = {str(year): _numbers([r for r in primary.days if r.day.year == year])
              for year in range(2021, 2026)}
    holdout = _numbers([r for r in primary.days if r.day.year in (2024, 2025)])
    monthly = _monthly(primary.days)
    regimes = _regimes(primary.days, prices)
    monte_carlo = _bootstrap(primary.days)
    buys = sum(fill["side"] == "buy" for fill in primary.fills)
    fills_by_year = {str(y): sum(fill["day"][:4] == str(y) for fill in primary.fills)
                     for y in range(2021, 2026)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_buys": buys >= 20,
        "annual_walk_forward": all(stats["end"] > stats["start"] and
                                   stats["sharpe"] is not None and stats["sharpe"] > 0
                                   and fills_by_year[year] > 0
                                   for year, stats in annual.items()),
        "historical_2024_2025_holdout": _numeric_pass(holdout),
        "monthly_stability": monthly["months"] == 60 and monthly["positive"] >= 36
                             and monthly["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(row["days"] >= 90 and row["net_dollar_pnl"] > 0
                            for row in regimes.values()),
        "block_monte_carlo": monte_carlo["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_week_latency": delayed.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "btc_buy_and_hold": _numbers(hold.days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": monthly, "regimes": regimes, "monte_carlo": monte_carlo,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "buys": buys, "fills_by_year": fills_by_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "slippage_reference_drag_usdt": primary.slippage_drag,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_week_latency_final_usdt": delayed.terminal_cash}
