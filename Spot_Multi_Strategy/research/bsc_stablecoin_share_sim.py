"""Frozen BSC stablecoin-share signal and BNB spot/cash account replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.bsc_stablecoin_share_source import SourceFailure
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes


START = date(2021, 1, 1)
END = date(2026, 1, 1)
INITIAL = 10000.0
FEE = .001
SLIP = .001
LOT = .001
MC_PATHS = 5000
MC_BLOCK = 21
MC_SEED = 20261113


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    reference_slippage: float
    terminal_cash: float


def weekly_target(wednesday: date, share: dict[date, float]) -> int:
    if wednesday.weekday() != 2:
        raise ValueError("BSC share decision must act on Wednesday")
    sunday = wednesday - timedelta(days=3)
    old = sunday - timedelta(days=28)
    try:
        recent, prior = share[sunday], share[old]
    except KeyError as exc:
        raise SourceFailure(f"missing completed BSC share day {exc}") from exc
    if not all(math.isfinite(x) and 0 < x <= 1 for x in (recent, prior)):
        raise SourceFailure("invalid BSC stablecoin share")
    return int(recent > prior)


def _buy(cash: float, reference: float, slippage: float,
         fee_rate: float) -> tuple[float, float, float, float]:
    execution = reference * (1 + slippage)
    raw = cash / (execution * (1 + fee_rate))
    qty = math.floor(raw / LOT + 1e-9) * LOT
    notional = qty * execution
    fee = notional * fee_rate
    residual = cash - notional - fee
    if qty <= 0 or notional < 10 or residual < -1e-7:
        raise SourceFailure("buy failed lot/notional/cash checks")
    return qty, notional, fee, max(residual, 0.0)


def simulate(share: dict[date, float], prices: pd.DataFrame,
             terminal_open: float,
             *, slippage: float = SLIP, latency_days: int = 0,
             fee_rate: float = FEE, buy_and_hold: bool = False) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise ValueError("invalid frozen scenario")
    if not math.isfinite(terminal_open) or terminal_open <= 0:
        raise SourceFailure("invalid terminal spot open")
    if not prices.index.equals(pd.date_range(date(2020, 12, 1),
                                              date(2025, 12, 31), tz="UTC", freq="D")):
        raise SourceFailure("BNB spot price grid invalid in simulator")
    cash, qty = INITIAL, 0.0
    total_fees, total_slippage = 0.0, 0.0
    fills: list[dict] = []
    days: list[Daily] = []
    prior_equity = INITIAL
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        row = prices.loc[ts]
        day_fee = 0.0
        trade_day = ((day == START) if buy_and_hold else
                     (day.weekday() == 3 + latency_days and
                      day - timedelta(days=latency_days) >= START))
        if trade_day:
            wednesday = day - timedelta(days=1 + latency_days)
            target = 1 if buy_and_hold else weekly_target(wednesday, share)
            if target != int(qty > 0):
                reference = float(row["open"])
                if target == 1:
                    new_qty, notional, fee, residual = _buy(cash, reference,
                                                            slippage, fee_rate)
                    lookback = prices.loc[pd.date_range(day - timedelta(days=30),
                                                        day - timedelta(days=1),
                                                        tz="UTC", freq="D")]
                    prior_volume = lookback["quote_volume"]
                    if len(prior_volume) != 30 or not np.isfinite(prior_volume.to_numpy()).all():
                        raise SourceFailure("missing prior 30 completed BNB quote-volume days")
                    median_volume = float(prior_volume.median())
                    if median_volume <= 0 or notional > .01 * median_volume:
                        raise SourceFailure(f"{day}: 1% fixed size-capacity Gate failed")
                    qty, cash = new_qty, residual
                    execution = reference * (1 + slippage)
                    total_slippage += qty * reference * slippage
                    side = "buy"
                else:
                    execution = reference * (1 - slippage)
                    notional = qty * execution
                    if notional < 10:
                        raise SourceFailure(f"{day}: sell below 10 USDT notional proxy")
                    fee = notional * fee_rate
                    cash += notional - fee
                    total_slippage += qty * reference * slippage
                    side = "sell"
                    new_qty = qty
                    qty = 0.0
                total_fees += fee
                day_fee += fee
                fills.append({"day": str(day), "side": side, "reference": reference,
                              "execution": execution, "quantity": new_qty,
                              "notional": notional, "fee": fee})
        end = cash + qty * float(row["close"])
        high = max(prior_equity, cash + qty * float(row["high"]), end)
        low = min(prior_equity, cash + qty * float(row["low"]), end)
        if not all(math.isfinite(x) and x > 0 for x in (prior_equity, high, low, end)):
            raise SourceFailure(f"{day}: nonfinite or ruined spot account")
        days.append(Daily(day, prior_equity, end, high, low, int(qty > 0), day_fee))
        prior_equity = end
    if qty:
        execution = terminal_open * (1 - slippage)
        notional = qty * execution
        if notional < 10:
            raise SourceFailure("terminal sale below 10 USDT notional proxy")
        fee = notional * fee_rate
        cash += notional - fee
        total_fees += fee
        total_slippage += qty * terminal_open * slippage
        fills.append({"day": str(END), "side": "sell", "reference": terminal_open,
                      "execution": execution, "quantity": qty,
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
    return {"paths": MC_PATHS, "joint_passes": passes, "joint_pass_rate": passes / MC_PATHS,
            "ruined": ruined, "undefined": undefined,
            "cagr": quantile(0), "sharpe": quantile(1),
            "max_drawdown": quantile(2), "profit_factor": quantile(3)}


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             extra_day: Simulation, hold: Simulation, prices: pd.DataFrame) -> dict:
    full = _numbers(primary.days)
    boundaries = [date(year, 1, 1) for year in range(2021, 2027)]
    folds = {f"Y{i+1}": _numbers([x for x in primary.days
                                  if boundaries[i] <= x.day < boundaries[i+1]])
             for i in range(5)}
    holdout = _numbers([x for x in primary.days if x.day >= date(2024, 1, 1)])
    months = _monthly(primary.days)
    states = _regimes(primary.days, prices)
    mc = _bootstrap(primary.days)
    buys = sum(x["side"] == "buy" for x in primary.fills)
    hold_stats = _numbers(hold.days)
    fills_by_fold = {f"Y{i+1}": sum(boundaries[i] <= date.fromisoformat(x["day"]) < boundaries[i+1]
                                    for x in primary.fills)
                     for i in range(5)}
    entries_by_fold = {f"Y{i+1}": sum(x["side"] == "buy" and
                                      boundaries[i] <= date.fromisoformat(x["day"]) < boundaries[i+1]
                                      for x in primary.fills)
                       for i in range(5)}
    held_by_fold = {f"Y{i+1}": any(boundaries[i] <= x.day < boundaries[i+1] and x.position
                                   for x in primary.days)
                    for i in range(5)}
    checks = {
        "numerical_full": _numeric_pass(full),
        "min_10_entries": buys >= 10,
        "annual_walk_forward": all(v["end"] > v["start"] and v["sharpe"] is not None and
                                   v["sharpe"] > 0 and (entries_by_fold[y] > 0 or held_by_fold[y])
                                   for y, v in folds.items()),
        "historical_Y4_Y5_holdout": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36 and
                             months["longest_nonpositive_streak"] <= 3,
        "four_regimes": all(x["days"] >= 90 and x["net_dollar_pnl"] > 0
                            for x in states.values()),
        "block_monte_carlo": mc["joint_pass_rate"] >= .80,
        "double_slippage": double_slip.terminal_cash > INITIAL,
        "one_day_latency": extra_day.terminal_cash > INITIAL,
        "incremental_alpha_vs_buy_and_hold": (
            hold_stats["sharpe"] is not None and full["sharpe"] is not None
            and full["cagr"] > hold_stats["cagr"]
            and full["sharpe"] > hold_stats["sharpe"]),
        "independent_executable_oos": False,
    }
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "bnb_buy_and_hold": hold_stats,
            "holdout_Y4_Y5": holdout, "folds": folds,
            "monthly": months, "regimes": states, "monte_carlo": mc,
            "checks": checks, "all_gate_passed": all(checks.values()),
            "buys": buys, "fills_by_fold": fills_by_fold,
            "entries_by_fold": entries_by_fold,
            "held_by_fold": held_by_fold,
            "fills": primary.fills, "fees_usdt": primary.fees,
            "slippage_reference_drag_usdt": primary.reference_slippage,
            "net_final_usdt": primary.terminal_cash,
            "gross_final_usdt": gross.terminal_cash,
            "double_slippage_final_usdt": double_slip.terminal_cash,
            "one_day_latency_final_usdt": extra_day.terminal_cash}
