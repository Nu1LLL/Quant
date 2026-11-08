"""Frozen cross-chain native-transfer growth rotation, costs, and full Gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.btc_eth_transfer_growth_source import SourceFailure
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
ASSETS = ("BTC", "ETH")


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    slippage_drag: float
    terminal_cash: float


def transfer_target(monday: date, counts: dict[str, dict[date, int]]) -> str | None:
    if monday.weekday() != 0:
        raise SourceFailure("transfer-count decision is not Monday UTC")
    totals: dict[str, tuple[int, int]] = {}
    for asset in ASSETS:
        if asset not in counts:
            raise SourceFailure("missing native-transfer asset")
        recent = [counts[asset].get(monday - timedelta(days=k)) for k in range(3, 31)]
        previous = [counts[asset].get(monday - timedelta(days=k)) for k in range(31, 59)]
        if any(type(value) is not int or value <= 0 for value in recent + previous):
            raise SourceFailure(f"{asset}: missing/nonpositive transfer count before {monday}")
        totals[asset] = (sum(recent), sum(previous))
    lhs = totals["BTC"][0] * totals["ETH"][1]
    rhs = totals["ETH"][0] * totals["BTC"][1]
    return "BTC" if lhs > rhs else "ETH" if rhs > lhs else None


def signal_schedule(counts: dict[str, dict[date, int]], *, latency_days: int = 0
                    ) -> dict[date, str | None]:
    if latency_days not in (0, 1):
        raise ValueError("invalid frozen transfer execution delay")
    out = {}
    monday = date(2021, 1, 4)
    while monday < END:
        day = monday + timedelta(days=latency_days)
        if day >= END:
            break
        out[day] = transfer_target(monday, counts)
        monday += timedelta(days=7)
    if len(out) < 250:
        raise SourceFailure("five-year transfer decision schedule incomplete")
    return out


def _median_capacity(prices: pd.DataFrame, day: date) -> float:
    prior = prices.loc[pd.date_range(day - timedelta(days=30),
                                      day - timedelta(days=1), tz="UTC", freq="D"),
                       "quote_volume"]
    if len(prior) != 30 or not np.isfinite(prior.to_numpy()).all():
        raise SourceFailure(f"{day}: missing prior 30 completed quote-volume days")
    median = float(prior.median())
    if median <= 0:
        raise SourceFailure(f"{day}: nonpositive prior quote volume")
    return .01 * median


def _buy(cash: float, budget: float, price: float, slip: float,
         fee_rate: float) -> tuple[float, float, float, float]:
    execution = price * (1 + slip)
    raw = min(cash, budget) / (execution * (1 + fee_rate))
    quantity = math.floor(raw / LOT + 1e-9) * LOT
    notional = quantity * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if quantity <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("spot buy failed cash, lot or minimum notional")
    return quantity, notional, fee, max(residual, 0.0)


def simulate(counts: dict[str, dict[date, int]], prices: dict[str, pd.DataFrame],
             terminal_open: dict[str, float], *, slippage: float = SLIP,
             fee_rate: float = FEE, latency_days: int = 0,
             static_half_half: bool = False) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen transfer stress scenario")
    for asset in ASSETS:
        if (asset not in prices or not prices[asset].index.equals(
                pd.date_range(date(2020, 12, 1) if asset == "BTC" else date(2020, 11, 1),
                              date(2025, 12, 31), tz="UTC", freq="D"))):
            raise SourceFailure(f"{asset}: spot grid invalid in simulator")
        if asset not in terminal_open or not math.isfinite(terminal_open[asset]) or terminal_open[asset] <= 0:
            raise SourceFailure(f"{asset}: terminal open invalid")
    schedule = {} if static_half_half else signal_schedule(counts, latency_days=latency_days)
    cash = prior_equity = INITIAL
    quantity = {asset: 0.0 for asset in ASSETS}
    total_fees = total_slippage = 0.0
    fills: list[dict] = []
    days: list[Daily] = []

    def sell(asset: str, day: date, reference: float, *, terminal: bool = False) -> float:
        nonlocal cash, total_fees, total_slippage
        q = quantity[asset]
        execution = reference * (1 - slippage)
        notional = q * execution
        if q <= 0 or notional < 10:
            raise SourceFailure(f"{asset} {day}: sale below minimum notional")
        fee = notional * fee_rate
        cash += notional - fee
        total_fees += fee
        total_slippage += q * reference * slippage
        fills.append({"day": str(day), "asset": asset, "side": "sell",
                      "reference": reference, "execution": execution,
                      "quantity": q, "notional": notional, "fee": fee,
                      "forced_terminal": terminal})
        quantity[asset] = 0.0
        return fee

    def buy(asset: str, day: date, reference: float, budget: float) -> float:
        nonlocal cash, total_fees, total_slippage
        q, notional, fee, residual = _buy(cash, budget, reference, slippage, fee_rate)
        if notional > _median_capacity(prices[asset], day):
            raise SourceFailure(f"{asset} {day}: 1% size-capacity Gate failed")
        quantity[asset] = q
        cash = residual
        total_fees += fee
        total_slippage += q * reference * slippage
        fills.append({"day": str(day), "asset": asset, "side": "buy",
                      "reference": reference, "execution": reference * (1 + slippage),
                      "quantity": q, "notional": notional, "fee": fee})
        return fee

    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        rows = {asset: prices[asset].loc[ts] for asset in ASSETS}
        day_fee = 0.0
        if static_half_half and day == START:
            for asset in ASSETS:
                day_fee += buy(asset, day, float(rows[asset]["open"]), INITIAL / 2)
        elif not static_half_half and day in schedule:
            target = schedule[day]
            current = next((asset for asset in ASSETS if quantity[asset] > 0), None)
            if sum(quantity[a] > 0 for a in ASSETS) > 1:
                raise SourceFailure("rotation account held multiple assets")
            if target != current:
                if current is not None:
                    day_fee += sell(current, day, float(rows[current]["open"]))
                if target is not None:
                    day_fee += buy(target, day, float(rows[target]["open"]), cash)
        end = cash + sum(quantity[a] * float(rows[a]["close"]) for a in ASSETS)
        high = max(prior_equity, cash + sum(quantity[a] * float(rows[a]["high"]) for a in ASSETS), end)
        low = min(prior_equity, cash + sum(quantity[a] * float(rows[a]["low"]) for a in ASSETS), end)
        if (not math.isfinite(cash) or cash < 0 or
                not all(math.isfinite(x) and x > 0
                        for x in (prior_equity, high, low, end))):
            raise SourceFailure(f"{day}: invalid or ruined spot account")
        if sum(quantity[a] * float(rows[a]["close"]) for a in ASSETS) > end + 1e-7:
            raise SourceFailure(f"{day}: gross leverage above 1x")
        days.append(Daily(day, prior_equity, end, high, low,
                          int(any(quantity[a] > 0 for a in ASSETS)), day_fee))
        prior_equity = end
    if any(quantity[a] > 0 for a in ASSETS):
        terminal_fee = 0.0
        for asset in ASSETS:
            if quantity[asset] > 0:
                terminal_fee += sell(asset, END, terminal_open[asset], terminal=True)
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash, max(last.high, cash),
                         min(last.low, cash), last.position, last.fee + terminal_fee)
    else:
        cash = prior_equity
    if len(days) != 1826 or cash <= 0 or abs(days[-1].end - cash) > 1e-7:
        raise SourceFailure("terminal cash and 1,826 daily factors do not reconcile")
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
             delayed: Simulation, benchmark: Simulation,
             prices: dict[str, pd.DataFrame]) -> dict:
    full = _numbers(primary.days)
    annual = {str(y): _numbers([r for r in primary.days if r.day.year == y])
              for y in range(2021, 2026)}
    holdout = _numbers([r for r in primary.days if r.day.year in (2024, 2025)])
    monthly = _monthly(primary.days)
    regimes = _regimes(primary.days, prices["BTC"])
    monte_carlo = _bootstrap(primary.days)
    entries = sum(fill["side"] == "buy" for fill in primary.fills)
    fills_by_year = {str(y): sum(fill["day"][:4] == str(y) for fill in primary.fills)
                     for y in range(2021, 2026)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_20_entries": entries >= 20,
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
        "one_day_latency": delayed.terminal_cash > INITIAL,
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "same_cost_50_50_buy_and_hold": _numbers(benchmark.days),
            "holdout_2024_2025": holdout, "annual": annual,
            "monthly": monthly, "regimes": regimes, "monte_carlo": monte_carlo,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "entries": entries, "fills_by_year": fills_by_year,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "slippage_reference_drag_usdt": primary.slippage_drag,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_day_latency_final_usdt": delayed.terminal_cash}
