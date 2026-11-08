"""One preregistered DEX-activity ranking, spot ledger and full five-year Gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd

from research.btc_eth_pairs_metrics import monte_carlo_compressed
from research.eth_defi_tvl_sim import Daily, _monthly, _numbers, _numeric_pass, _regimes


START, END = date(2021, 1, 1), date(2026, 1, 1)
INITIAL = 10_000.0
FEE, SLIP = .001, .002
LOT = {"BNBUSDT": .001, "ETHUSDT": .0001, "TRXUSDT": 1.0}
THRESHOLD = math.log(1.25)
STOP = .88
MC_SEED = 20261129


class RotationFailure(ValueError):
    pass


@dataclass(frozen=True)
class Simulation:
    days: tuple[Daily, ...]
    fills: tuple[dict, ...]
    fees: float
    reference_slippage: float
    terminal_cash: float
    targets_sha256: str


def weekly_target(thursday: date, volume: pd.DataFrame) -> str | None:
    if thursday.weekday() != 3 or not START <= thursday < END:
        raise RotationFailure("target requested outside a formal Thursday")
    sunday = thursday - timedelta(days=4)
    if sunday.weekday() != 6:
        raise RotationFailure("DEX signal is not tied to completed Sunday")
    scored = []
    for symbol in sorted(LOT):
        recent_dates = pd.date_range(sunday - timedelta(days=13), sunday,
                                     tz="UTC", freq="D")
        prior_dates = pd.date_range(sunday - timedelta(days=27),
                                    sunday - timedelta(days=14), tz="UTC", freq="D")
        try:
            recent = volume.loc[recent_dates, symbol]
            prior = volume.loc[prior_dates, symbol]
        except KeyError as exc:
            raise RotationFailure("DEX signal warmup/date missing") from exc
        if len(recent) != 14 or len(prior) != 14:
            raise RotationFailure("DEX 14/14-day signal window incomplete")
        new, old = float(recent.sum()), float(prior.sum())
        if not all(math.isfinite(x) and x > 0 for x in (new, old)):
            raise RotationFailure("invalid DEX signal volume")
        scored.append((math.log(new / old), symbol))
    best_score, best_symbol = min(scored, key=lambda x: (-x[0], x[1]))
    return best_symbol if best_score >= THRESHOLD else None


def frozen_targets(volume: pd.DataFrame) -> dict[date, str | None]:
    expected = pd.date_range(date(2020, 12, 1), date(2025, 12, 31),
                             tz="UTC", freq="D")
    if not volume.index.equals(expected) or set(volume.columns) != set(LOT):
        raise RotationFailure("DEX series grid/symbol changed")
    output = {}
    day = date(2021, 1, 7)
    while day < END:
        output[day] = weekly_target(day, volume)
        day += timedelta(days=7)
    if len(output) != 260:
        raise RotationFailure("weekly DEX decision count changed")
    return output


def _capacity(prices: dict[str, pd.DataFrame], symbol: str, day: date,
              notional: float) -> None:
    dates = pd.date_range(day - timedelta(days=30), day - timedelta(days=1),
                          tz="UTC", freq="D")
    try:
        values = prices[symbol].loc[dates, "quote_volume"]
    except KeyError as exc:
        raise RotationFailure("prior 30-day quote-volume source missing") from exc
    if (len(values) != 30 or not np.isfinite(values.to_numpy()).all()
            or (values.to_numpy() <= 0).any()
            or notional > .01 * float(values.median())):
        raise RotationFailure(f"{day} {symbol}: fixed 1% capacity limit failed")


def simulate(targets: dict[date, str | None], prices: dict[str, pd.DataFrame],
             terminal: dict[str, float], *, slippage: float = SLIP,
             fee_rate: float = FEE, latency_days: int = 0) -> Simulation:
    if latency_days not in (0, 1) or slippage < 0 or fee_rate < 0:
        raise RotationFailure("invalid fixed scenario")
    if set(prices) != {"BTCUSDT", *LOT} or set(terminal) != {"BTCUSDT", *LOT}:
        raise RotationFailure("price universe changed")
    expected = pd.date_range(date(2020, 12, 1), date(2025, 12, 31), tz="UTC", freq="D")
    if any(not frame.index.equals(expected) for frame in prices.values()):
        raise RotationFailure("spot daily price grid changed")
    if len(targets) != 260 or any(day.weekday() != 3 for day in targets):
        raise RotationFailure("weekly targets changed")
    import hashlib
    schedule = hashlib.sha256()
    for day, symbol in sorted(targets.items()):
        schedule.update(f"{day}:{symbol or 'CASH'}\n".encode())
    cash, symbol, quantity, entry_fill = INITIAL, None, 0.0, None
    pending_stop: tuple[float, float] | None = None
    fees = slip_drag = 0.0
    fills = []
    days: list[Daily] = []
    prior_equity = INITIAL

    def sell(day: date, reference: float, reason: str) -> float:
        nonlocal cash, symbol, quantity, entry_fill, fees, slip_drag
        if symbol is None or quantity <= 0 or not math.isfinite(reference) or reference <= 0:
            raise RotationFailure("invalid spot sale")
        execution = reference * (1 - slippage)
        notional = quantity * execution
        if notional < 10:
            raise RotationFailure(f"{day}: sale below 10 USDT proxy")
        fee = notional * fee_rate
        cash += notional - fee
        fees += fee
        slip_drag += quantity * reference * slippage
        fills.append({"day": str(day), "side": "sell", "symbol": symbol,
                      "reason": reason, "reference": reference,
                      "execution": execution, "quantity": quantity,
                      "notional": notional, "fee": fee})
        symbol, quantity, entry_fill = None, 0.0, None
        return fee

    def buy(day: date, target: str, reference: float) -> float:
        nonlocal cash, symbol, quantity, entry_fill, fees, slip_drag
        if symbol is not None or target not in LOT or not math.isfinite(reference) or reference <= 0:
            raise RotationFailure("invalid spot purchase")
        execution = reference * (1 + slippage)
        raw = cash / (execution * (1 + fee_rate))
        lot = LOT[target]
        new_quantity = math.floor(raw / lot + 1e-9) * lot
        notional = new_quantity * execution
        fee = notional * fee_rate
        if new_quantity <= 0 or notional < 10 or cash - notional - fee < -1e-7:
            raise RotationFailure(f"{day}: spot lot/notional/cash check failed")
        _capacity(prices, target, day, notional)
        cash = max(0.0, cash - notional - fee)
        symbol, quantity, entry_fill = target, new_quantity, execution
        fees += fee
        slip_drag += quantity * reference * slippage
        fills.append({"day": str(day), "side": "buy", "symbol": target,
                      "reason": "weekly_target", "reference": reference,
                      "execution": execution, "quantity": quantity,
                      "notional": notional, "fee": fee})
        return fee

    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        day_fee = 0.0
        transient = [prior_equity]
        if pending_stop is not None:
            if symbol is None:
                raise RotationFailure("pending stop without position")
            trigger_low, threshold = pending_stop
            reference = min(float(prices[symbol].loc[ts, "open"]), trigger_low, threshold)
            day_fee += sell(day, reference, "next_day_stop")
            transient.append(cash)
            pending_stop = None
        base_thursday = day - timedelta(days=latency_days)
        if base_thursday in targets and (latency_days == 0 or day.weekday() == 4):
            target = targets[base_thursday]
            if symbol != target:
                if symbol is not None:
                    day_fee += sell(day, float(prices[symbol].loc[ts, "open"]),
                                    "weekly_switch")
                    transient.append(cash)
                if target is not None:
                    day_fee += buy(day, target, float(prices[target].loc[ts, "open"]))
                    transient.append(cash + quantity * float(prices[target].loc[ts, "open"]))
        if symbol is None:
            close_equity = cash
            high_equity = max(transient)
            low_equity = min(transient)
            held = 0
        else:
            row = prices[symbol].loc[ts]
            close_equity = cash + quantity * float(row["close"])
            high_equity = max(*transient, cash + quantity * float(row["high"]), close_equity)
            low_equity = min(*transient, cash + quantity * float(row["low"]), close_equity)
            held = 1
            if float(row["low"]) <= STOP * float(entry_fill):
                pending_stop = (float(row["low"]), STOP * float(entry_fill))
        if (not all(math.isfinite(x) and x > 0 for x in
                    (cash + 1e-12, close_equity, high_equity, low_equity))
                or cash < -1e-7 or quantity < 0):
            raise RotationFailure(f"{day}: invalid or ruined spot account")
        days.append(Daily(day, prior_equity, close_equity,
                          high_equity, low_equity, held, day_fee))
        prior_equity = close_equity
    if symbol is not None:
        reference = terminal[symbol]
        reason = "forced_terminal"
        if pending_stop is not None:
            reference = min(reference, *pending_stop)
            reason = "terminal_after_stop"
        fee = sell(END, reference, reason)
        last = days[-1]
        days[-1] = Daily(last.day, last.start, cash, max(last.high, cash),
                         min(last.low, cash), last.position, last.fee + fee)
    else:
        cash = prior_equity
    if len(days) != 1826 or not math.isfinite(cash) or cash <= 0:
        raise RotationFailure("account length or terminal equity invalid")
    if not math.isclose(days[-1].end, cash, rel_tol=1e-10, abs_tol=1e-6):
        raise RotationFailure("terminal equity does not reconcile")
    if not math.isclose(sum(d.end - d.start for d in days), cash - INITIAL,
                        rel_tol=1e-9, abs_tol=1e-5):
        raise RotationFailure("daily dollar PnL does not reconcile")
    return Simulation(tuple(days), tuple(fills), fees, slip_drag, cash, schedule.hexdigest())


def equal_weight_buyhold(prices: dict[str, pd.DataFrame], terminal: dict[str, float],
                         *, slippage: float = SLIP, fee_rate: float = FEE) -> dict:
    first = date(2021, 1, 7)
    quantities = {}
    residual = INITIAL
    for symbol in sorted(LOT):
        reference = float(prices[symbol].loc[pd.Timestamp(first, tz="UTC"), "open"])
        execution = reference * (1 + slippage)
        lot = LOT[symbol]
        qty = math.floor((INITIAL / 3) / (execution * (1 + fee_rate)) / lot + 1e-9) * lot
        notional = qty * execution
        fee = notional * fee_rate
        if qty <= 0 or notional < 10 or residual - notional - fee < -1e-7:
            raise RotationFailure("buy-hold lot/cash check failed")
        _capacity(prices, symbol, first, notional)
        residual -= notional + fee
        quantities[symbol] = qty
    days = []
    prior = INITIAL
    for ts in pd.date_range(START, END - timedelta(days=1), tz="UTC", freq="D"):
        day = ts.date()
        if day < first:
            end = high = low = prior
        else:
            end = residual + sum(qty * float(prices[s].loc[ts, "close"])
                                 for s, qty in quantities.items())
            high = max(prior, residual + sum(qty * float(prices[s].loc[ts, "high"])
                                             for s, qty in quantities.items()))
            low = min(prior, residual + sum(qty * float(prices[s].loc[ts, "low"])
                                            for s, qty in quantities.items()))
        days.append(Daily(day, prior, end, high, low, int(day >= first), 0.0))
        prior = end
    final = residual + sum(qty * terminal[s] * (1 - slippage) * (1 - fee_rate)
                           for s, qty in quantities.items())
    last = days[-1]
    days[-1] = Daily(last.day, last.start, final, max(last.high, final),
                     min(last.low, final), last.position, last.fee)
    return _numbers(days)


def _mc(days: tuple[Daily, ...]) -> dict:
    factors = np.empty((len(days), 4), dtype=float)
    for i, row in enumerate(days):
        factors[i] = (row.end / row.start, row.high / row.start,
                      row.low / row.start, 1 - row.low / row.high)
    return monte_carlo_compressed(factors, paths=5000, block_days=21, seed=MC_SEED)


def evaluate(primary: Simulation, gross: Simulation, double_slip: Simulation,
             extra_day: Simulation, hold: dict, btc: pd.DataFrame) -> dict:
    if not (primary.targets_sha256 == gross.targets_sha256 ==
            double_slip.targets_sha256 == extra_day.targets_sha256):
        raise RotationFailure("stress scenarios changed ranked targets")
    full = _numbers(primary.days)
    annual = {str(year): _numbers([row for row in primary.days if row.day.year == year])
              for year in range(2021, 2026)}
    holdout = _numbers([row for row in primary.days if row.day.year >= 2024])
    months = _monthly(primary.days)
    regimes = _regimes(primary.days, btc)
    mc = _mc(primary.days)
    entries = sum(fill["side"] == "buy" for fill in primary.fills)
    years_pass = {}
    for year, score in annual.items():
        y = int(year)
        exposed = any(row.position for row in primary.days if row.day.year == y)
        years_pass[year] = (exposed and score["end"] > score["start"]
                            and score["sharpe"] is not None and score["sharpe"] > 0)
    gate = {
        "numerical_full": _numeric_pass(full),
        "at_least_20_buys": entries >= 20,
        "annual_forward_folds": all(years_pass.values()),
        "historical_holdout_numeric": _numeric_pass(holdout),
        "monthly_stability": months["months"] == 60 and months["positive"] >= 36
        and months["longest_nonpositive_streak"] <= 3,
        "four_btc_states": all(state["days"] >= 90 and state["net_dollar_pnl"] > 0
                               for state in regimes.values()),
        "monte_carlo": mc["joint_pass_fraction"] >= .80,
        "double_slippage_profitable": double_slip.terminal_cash > INITIAL,
        "friday_latency_profitable": extra_day.terminal_cash > INITIAL,
        "beats_equal_weight_buyhold": (full["sharpe"] is not None
                                      and hold["sharpe"] is not None
                                      and full["cagr"] > hold["cagr"]
                                      and full["sharpe"] > hold["sharpe"]),
        "independent_executable_oos": False,
    }
    gate["all_numeric"] = all(value for key, value in gate.items()
                              if key != "independent_executable_oos")
    gate["full"] = gate["all_numeric"] and gate["independent_executable_oos"]
    return {"full": full, "gross_no_cost": _numbers(gross.days),
            "double_slippage": _numbers(double_slip.days),
            "friday_latency": _numbers(extra_day.days),
            "equal_weight_buyhold": hold, "annual": annual,
            "annual_pass": years_pass, "holdout_2024_2025": holdout,
            "monthly": months, "regimes": regimes, "monte_carlo": mc,
            "fees": primary.fees, "reference_slippage": primary.reference_slippage,
            "target_schedule_sha256": primary.targets_sha256,
            "fills": primary.fills, "completed_buys": entries,
            "net_final_usdt": primary.terminal_cash, "gate": gate}
