"""One configuration of the preregistered three-coin trade-size hypothesis."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from hashlib import sha256
import math

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .trade_size_pressure_data import BASE, SYMBOLS, TradeSizeTape


BASE_DAY = BASE.date()
START = date(2021, 3, 9)
END = date(2026, 10, 1)
START_BAR = (START - BASE_DAY).days * 288
END_BAR = (END - BASE_DAY).days * 288
N_DAYS = (END - START).days
FIRST_WEEK = date(2020, 12, 7)
LAST_WEEK = date(2026, 9, 14)
FEE = 0.0005
INITIAL = 10_000.0
LOT = 0.001


class TradeSizeStrategyFailure(ValueError):
    def __init__(self, reason: str, *, bar: int | None = None,
                 equity: float | None = None):
        super().__init__(reason)
        self.bar = bar
        self.equity = equity


def build_schedule(tape: TradeSizeTape) -> tuple[dict[date, tuple[int, int] | None], str]:
    if (tape.opening.shape != (END_BAR, 3)
            or any(x.shape != tape.opening.shape for x in
                   (tape.high, tape.low, tape.close, tape.quote_volume, tape.trade_count))):
        raise TradeSizeStrategyFailure("registered source shape changed")
    weeks = (LAST_WEEK - FIRST_WEEK).days // 7 + 1
    if weeks != 302 or (LAST_WEEK - FIRST_WEEK).days % 7:
        raise TradeSizeStrategyFailure("complete-week grid changed")
    sizes = np.empty((weeks, 3), dtype=np.float64)
    for i in range(weeks):
        monday = FIRST_WEEK + timedelta(weeks=i)
        start = (monday - BASE_DAY).days * 288
        end = start + 2016
        if start < 0 or end > END_BAR:
            raise TradeSizeStrategyFailure("weekly signal exceeds price source")
        volume = np.sum(tape.quote_volume[start:end], axis=0)
        count = np.sum(tape.trade_count[start:end], axis=0)
        if (not np.all(np.isfinite(volume)) or not np.all(np.isfinite(count))
                or np.any(volume <= 0) or np.any(count <= 0)):
            raise TradeSizeStrategyFailure("empty/nonfinite weekly size source")
        sizes[i] = volume / count
    decisions = {}
    digest = sha256()
    for i in range(13, weeks + 1):
        current = sizes[i - 1]
        baseline = np.mean(sizes[i - 13:i - 1], axis=0)
        if np.any(baseline <= 0) or not np.all(np.isfinite(baseline)):
            raise TradeSizeStrategyFailure("invalid 12-week size baseline")
        abnormal = np.log(current / baseline)
        order = sorted(range(3), key=lambda col: (float(abnormal[col]), SYMBOLS[col]))
        pair = ((order[0], order[-1])
                if float(abnormal[order[-1]] - abnormal[order[0]]) >= .15 else None)
        entry = FIRST_WEEK + timedelta(weeks=i, days=1)
        decisions[entry] = pair
        digest.update(f"{entry}:{SYMBOLS[pair[0]]}:{SYMBOLS[pair[1]]}\n".encode()
                      if pair else f"{entry}:flat\n".encode())
    if len(decisions) != 290 or min(decisions) != START or max(decisions) != date(2026, 9, 22):
        raise TradeSizeStrategyFailure("registered decision grid changed")
    return decisions, digest.hexdigest()


def _events(decisions: dict[date, tuple[int, int] | None], delay: int) -> dict[int, tuple[int, int] | None]:
    if delay not in (1, 2):
        raise ValueError("delay must be 00:05 or 00:10")
    events = {(entry - BASE_DAY).days * 288 + delay: pair
              for entry, pair in decisions.items()}
    terminal = (date(2026, 9, 29) - BASE_DAY).days * 288 + delay
    if terminal in events:
        raise TradeSizeStrategyFailure("last exit overlaps a registered entry")
    events[terminal] = None
    return events


def replay(tape: TradeSizeTape, decisions: dict[date, tuple[int, int] | None],
           *, gross: float, slippage: float, delay: int,
           no_funding_credit: bool = False,
           shared_capacity: dict[date, bool] | None = None) -> dict:
    if gross not in (1.0, 2.0):
        raise ValueError("gross must be the frozen 1x or 2x")
    orders = _events(decisions, delay)
    boundary = np.empty(N_DAYS + 1, dtype=np.float64)
    maxima = np.empty(N_DAYS, dtype=np.float64)
    minima = np.empty(N_DAYS, dtype=np.float64)
    intraday_dd = np.empty(N_DAYS, dtype=np.float64)
    active_days = np.zeros(N_DAYS, dtype=np.bool_)
    regular_path = np.empty(288, dtype=np.float64)
    adverse_path = np.empty(288, dtype=np.float64)
    cash = INITIAL
    q = [0.0, 0.0, 0.0]
    entry_mid = [0.0, 0.0, 0.0]
    fees = slips = funding_total = price_pnl = 0.0
    entered = capacity_cash = forced_exits = 0
    force_bar = -1
    capacity = {}
    day = -1
    for bar in range(START_BAR, END_BAR):
        within = bar % 288
        opened = [float(tape.opening[bar, col]) for col in range(3)]
        pre_open = cash + sum(q[col] * opened[col] for col in range(3))
        if within == 0:
            day += 1
            boundary[day] = pre_open
        if not math.isfinite(pre_open) or pre_open <= 0:
            raise TradeSizeStrategyFailure("nonpositive/nonfinite open equity",
                                           bar=bar, equity=pre_open)
        if bar in orders or bar == force_bar:
            for col, quantity in enumerate(q):
                if quantity:
                    mid = opened[col]
                    fill = mid * (1 - math.copysign(slippage, quantity))
                    fee = FEE * abs(quantity * fill)
                    cash += quantity * fill - fee
                    fees += fee
                    slips += abs(quantity) * abs(fill - mid)
                    price_pnl += quantity * (mid - entry_mid[col])
                    q[col] = 0.0
            if bar == force_bar:
                forced_exits += 1
                if bar in orders and orders[bar] is not None and shared_capacity is None:
                    entry = BASE_DAY + timedelta(days=bar // 288)
                    capacity[entry] = False
                force_bar = -1
            else:
                pair = orders[bar]
                if pair is not None:
                    entry = BASE_DAY + timedelta(days=bar // 288)
                    target = cash * gross / 2
                    if shared_capacity is None:
                        allowed = all(
                            target <= .01 * float(np.median(tape.quote_volume[bar - 288:bar, col]))
                            for col in pair
                        )
                        capacity[entry] = allowed
                    else:
                        if entry not in shared_capacity:
                            raise TradeSizeStrategyFailure("capacity decision missing")
                        allowed = shared_capacity[entry]
                    if not allowed:
                        capacity_cash += 1
                    else:
                        sizes = [math.floor(target / opened[col] / LOT) * LOT
                                 for col in pair]
                        if any(sizes[j] * opened[col] < 5 for j, col in enumerate(pair)):
                            capacity_cash += 1
                        else:
                            for col, signed_qty in ((pair[0], sizes[0]),
                                                    (pair[1], -sizes[1])):
                                mid = opened[col]
                                fill = mid * (1 + math.copysign(slippage, signed_qty))
                                fee = FEE * abs(signed_qty * fill)
                                cash -= signed_qty * fill + fee
                                fees += fee
                                slips += abs(signed_qty) * abs(fill - mid)
                                q[col] = signed_qty
                                entry_mid[col] = mid
                            entered += 1
        for col, rate in tape.funding.get(bar, ()):
            payment = -q[col] * opened[col] * rate
            if no_funding_credit:
                payment = min(payment, 0.0)
            cash += payment
            funding_total += payment
        post_open = cash + sum(q[col] * opened[col] for col in range(3))
        regular = cash + sum(q[col] * float(tape.close[bar, col]) for col in range(3))
        adverse = cash + sum(
            q[col] * float(tape.low[bar, col] if q[col] > 0 else tape.high[bar, col])
            for col in range(3)
        )
        regular_path[within] = max(pre_open, post_open, regular)
        adverse_path[within] = min(pre_open, post_open, regular, adverse)
        gross_exposure = sum(abs(q[col] * float(tape.close[bar, col])) for col in range(3))
        if (not math.isfinite(adverse) or adverse <= 0
                or (gross_exposure and adverse <= .005 * gross_exposure)):
            raise TradeSizeStrategyFailure("maintenance/ruin at adverse 5m mark",
                                           bar=bar, equity=adverse)
        if gross_exposure and regular > 0 and gross_exposure / regular > 5:
            raise TradeSizeStrategyFailure("hard 5x leverage cap breached",
                                           bar=bar, equity=regular)
        if gross_exposure and regular > 0 and gross_exposure / regular > 3 and force_bar < 0:
            force_bar = bar + 1
        if any(q):
            active_days[day] = True
        if within == 287:
            start_equity = boundary[day]
            peak = np.maximum.accumulate(np.maximum(regular_path, start_equity))
            maxima[day] = max(start_equity, float(np.max(regular_path)))
            minima[day] = min(start_equity, float(np.min(adverse_path)))
            intraday_dd[day] = float(np.max(1.0 - adverse_path / peak))
    if day != N_DAYS - 1 or any(q) or force_bar >= 0:
        raise TradeSizeStrategyFailure("terminal account/date grid incomplete")
    boundary[N_DAYS] = cash
    factors = np.empty((N_DAYS, 4), dtype=np.float64)
    for i in range(N_DAYS):
        start, end = boundary[i:i + 2]
        high = max(start, end, maxima[i])
        low = min(start, end, minima[i])
        factors[i] = (end / start, high / start, low / start,
                      max(intraday_dd[i], 1.0 - low / high))
    metrics = _metrics_from_factors(factors)
    if not math.isclose(metrics["ending_equity"], cash, rel_tol=1e-8, abs_tol=1e-5):
        raise TradeSizeStrategyFailure("compressed daily/equity mismatch")
    if not math.isclose(cash - INITIAL, price_pnl - fees - slips + funding_total,
                        rel_tol=1e-7, abs_tol=.01):
        raise TradeSizeStrategyFailure("price/fee/slippage/funding mismatch")
    return {"factors": factors, "metrics": metrics, "equity": cash,
            "fees": fees, "slippage": slips, "funding": funding_total,
            "price_pnl": price_pnl, "entered_weeks": entered,
            "capacity_cash_weeks": capacity_cash, "forced_exits": forced_exits,
            "active_days": active_days, "capacity": capacity}


def evaluate(tape: TradeSizeTape, decisions: dict[date, tuple[int, int] | None]) -> dict:
    formal = replay(tape, decisions, gross=2., slippage=.0005, delay=1)
    capacity = formal["capacity"]
    runs = {
        "formal_2x": formal,
        "prerequisite_1x": replay(tape, decisions, gross=1., slippage=.0005,
                                  delay=1, shared_capacity=capacity),
        "double_slippage": replay(tape, decisions, gross=2., slippage=.001,
                                   delay=1, shared_capacity=capacity),
        "extra_bar_latency": replay(tape, decisions, gross=2., slippage=.0005,
                                    delay=2, shared_capacity=capacity),
        "no_funding_credit": replay(tape, decisions, gross=2., slippage=.0005,
                                    delay=1, no_funding_credit=True,
                                    shared_capacity=capacity),
    }
    factors = formal["factors"]
    dates = [START + timedelta(days=i) for i in range(N_DAYS)]
    months: dict[str, list[int]] = defaultdict(list)
    for i, day in enumerate(dates):
        months[day.strftime("%Y-%m")].append(i)
    if len(months) != 67:
        raise TradeSizeStrategyFailure("67-month formal grid changed")
    monthly = {month: float(np.prod(factors[index, 0]) - 1)
               for month, index in sorted(months.items())}
    positive = sum(value > 0 for value in monthly.values())
    longest = streak = 0
    for value in monthly.values():
        streak = 0 if value > 0 else streak + 1
        longest = max(longest, streak)
    annual = {}
    for year in range(2021, 2027):
        index = [i for i, day in enumerate(dates) if day.year == year]
        metrics = _metrics_from_factors(factors[index])
        active = int(np.count_nonzero(formal["active_days"][index]))
        annual[str(year)] = {
            "days": len(index), "active_days": active,
            "net_return": metrics["ending_equity"] / INITIAL - 1,
            "sharpe": metrics["sharpe"],
            "pass": active > 0 and metrics["ending_equity"] > INITIAL
            and metrics["sharpe"] is not None and metrics["sharpe"] > 0,
        }
    holdout_index = [i for i, day in enumerate(dates) if day >= date(2025, 1, 1)]
    holdout = _metrics_from_factors(factors[holdout_index])
    btc = tape.opening[::288, SYMBOLS.index("BTCUSDT")]
    states = {key: {"days": 0, "net_pnl": 0.0} for key in
              ("bull_high", "bull_low", "bear_high", "bear_low")}
    equity = INITIAL
    for i, day in enumerate(dates):
        global_day = (day - BASE_DAY).days
        window = btc[global_day - 30:global_day + 1]
        if len(window) != 31:
            raise TradeSizeStrategyFailure("30-day BTC regime history unavailable")
        returns = np.diff(np.log(window))
        state = ("bull" if math.log(float(window[-1] / window[0])) > 0 else "bear")
        state += "_high" if float(np.std(returns, ddof=1) * math.sqrt(365)) >= .8 else "_low"
        pnl = equity * (factors[i, 0] - 1)
        states[state]["days"] += 1
        states[state]["net_pnl"] += float(pnl)
        equity += pnl
    mc = monte_carlo_compressed(factors, paths=5000, block_days=21, seed=20261119)
    m = formal["metrics"]
    prereq = runs["prerequisite_1x"]["metrics"]
    numeric = (m["cagr"] >= .4 and m["sharpe"] is not None and m["sharpe"] > 1.5
               and m["max_drawdown"] <= .2 and m["profit_factor"] is not None
               and m["profit_factor"] > 1.3)
    holdout_numeric = (holdout["cagr"] >= .4 and holdout["sharpe"] is not None
                       and holdout["sharpe"] > 1.5 and holdout["max_drawdown"] <= .2
                       and holdout["profit_factor"] is not None
                       and holdout["profit_factor"] > 1.3)
    gate = {
        "prerequisite_1x": (prereq["ending_equity"] > INITIAL
                           and prereq["sharpe"] is not None and prereq["sharpe"] > 1.5
                           and prereq["profit_factor"] is not None
                           and prereq["profit_factor"] > 1.3
                           and prereq["max_drawdown"] <= .2),
        "formal_numeric": numeric,
        "active_weeks": formal["entered_weeks"] >= 30,
        "monthly": positive >= 41 and longest <= 3,
        "annual": all(row["pass"] for row in annual.values()),
        "retrospective_holdout": holdout_numeric,
        "regimes": all(row["days"] >= 90 and row["net_pnl"] > 0
                       for row in states.values()),
        "monte_carlo": mc["joint_pass_fraction"] >= .8,
        "stresses": all(runs[name]["equity"] > INITIAL for name in
                        ("double_slippage", "extra_bar_latency", "no_funding_credit")),
        "independent_executable_oos": False,
    }
    return {
        "accounts": {name: {key: value for key, value in run.items()
                            if key not in ("factors", "active_days", "capacity")}
                     for name, run in runs.items()},
        "scheduled_weeks": len(decisions),
        "signal_active_weeks": sum(pair is not None for pair in decisions.values()),
        "monthly": {"positive": positive, "nonpositive": len(monthly) - positive,
                    "longest_nonpositive_streak": longest,
                    "worst_month": min(monthly, key=monthly.get),
                    "worst_return": min(monthly.values()), "returns": monthly},
        "annual": annual,
        "holdout": holdout,
        "regimes": states,
        "monte_carlo": mc,
        "gate": {**gate, "all_passed": all(gate.values())},
    }
