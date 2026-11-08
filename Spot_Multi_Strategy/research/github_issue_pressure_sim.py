"""Frozen weekly BTC/ETH issue-pressure account and Gate replay."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import json
import math
from pathlib import Path

import numpy as np

from .btc_eth_pairs_metrics import _metrics_from_factors, monte_carlo_compressed
from .github_issue_pressure_data import PairTape
from .github_issue_pressure_source import FIRST_MONDAY, mondays


START = date(2021, 1, 5)
END = date(2026, 10, 1)
BASE = date(2021, 1, 1)
START_INDEX = (START - BASE).days * 288
END_INDEX = (END - BASE).days * 288
INITIAL = 10_000.0
FEE = 0.0005
LOT = 0.001


class StrategyFailure(ValueError):
    def __init__(self, message: str, *, bar: int | None = None, equity: float | None = None):
        super().__init__(message)
        self.bar = bar
        self.equity = equity


def load_fixed_counts(path: Path) -> tuple[dict[date, tuple[int, int]], str]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("prereg_commit") != "8cb7cdb" or source.get("queries") != 622:
        raise StrategyFailure("GitHub input provenance changed")
    weeks = mondays()
    rows = source.get("weeks")
    if not isinstance(rows, list) or len(rows) != len(weeks):
        raise StrategyFailure("GitHub input week grid changed")
    values = {}
    canonical = ""
    for monday, row in zip(weeks, rows, strict=True):
        btc, eth = row.get("btc"), row.get("eth")
        if (row.get("monday") != monday.isoformat() or type(btc) is not int
                or type(eth) is not int or not 0 <= btc < 1000 or not 0 <= eth < 1000):
            raise StrategyFailure("GitHub input value/grid changed")
        values[monday] = (btc, eth)
        canonical += f"{monday},{btc},{eth}\n"
    from hashlib import sha256
    digest = sha256(canonical.encode("ascii")).hexdigest()
    if digest != source.get("sha256") or digest != "808bd4c4fc0cdb7c789c0e48fcf9891c591517d17a13fc04fcf15fc3273c99ee":
        raise StrategyFailure("GitHub input hash changed")
    return values, digest


def schedule(counts: dict[date, tuple[int, int]]) -> dict[date, int]:
    weeks = mondays()
    if set(counts) != set(weeks):
        raise StrategyFailure("issue weeks incomplete")
    decisions: dict[date, int] = {}
    for current in range(13, len(weeks) + 1):
        monday = FIRST_MONDAY + timedelta(weeks=current)
        last = counts[weeks[current - 1]]
        history = [counts[weeks[j]] for j in range(current - 13, current - 1)]
        if len(history) != 12:
            raise StrategyFailure("12-week issue baseline incomplete")
        mean_btc = sum(x[0] for x in history) / 12
        mean_eth = sum(x[1] for x in history) / 12
        difference = ((last[1] - mean_eth) / math.sqrt(mean_eth + 1)
                      - (last[0] - mean_btc) / math.sqrt(mean_btc + 1))
        decisions[monday + timedelta(days=1)] = (1 if difference >= 1.5
                                                 else -1 if difference <= -1.5 else 0)
    if (len(decisions) != 299 or min(decisions) != START
            or max(decisions) != date(2026, 9, 22)):
        raise StrategyFailure("registered decision grid changed")
    return decisions


def _event_schedule(decisions: dict[date, int], delay: int, comparator: bool) -> dict[int, int]:
    if delay not in (1, 2):
        raise ValueError("delay must be 00:05 or 00:10")
    events = {((day - BASE).days * 288 + delay): (1 if comparator else sign)
              for day, sign in decisions.items()}
    last_exit = ((date(2026, 9, 29) - BASE).days * 288 + delay)
    if last_exit in events:
        raise StrategyFailure("terminal exit overlaps entry")
    events[last_exit] = 0
    return events


def replay(tape: PairTape, decisions: dict[date, int], *, slippage: float,
           delay: int, no_funding_credit: bool = False,
           comparator: bool = False) -> dict:
    if (tape.opening.shape != (END_INDEX, 2)
            or any(x.shape != tape.opening.shape for x in
                   (tape.high, tape.low, tape.close, tape.quote_volume))):
        raise StrategyFailure("registered 5m tape length/shape changed")
    events = _event_schedule(decisions, delay, comparator)
    daily_count = (END - START).days
    boundary = np.empty(daily_count + 1, dtype=np.float64)
    maxima = np.empty(daily_count, dtype=np.float64)
    minima = np.empty(daily_count, dtype=np.float64)
    local_dd = np.empty(daily_count, dtype=np.float64)
    active_days = np.zeros(daily_count, dtype=np.bool_)
    cash, q0, q1 = INITIAL, 0.0, 0.0
    entry_mid = [0.0, 0.0]
    fees = slips = funding_total = price_pnl = 0.0
    entered = capacity_cash = forced_exits = 0
    next_force = -1
    regular_path = np.empty(288, dtype=np.float64)
    adverse_path = np.empty(288, dtype=np.float64)
    day = -1
    for bar in range(START_INDEX, END_INDEX):
        within = bar % 288
        o0, o1 = float(tape.opening[bar, 0]), float(tape.opening[bar, 1])
        pre_open = cash + q0 * o0 + q1 * o1
        if within == 0:
            day += 1
            boundary[day] = pre_open
        if not math.isfinite(pre_open) or pre_open <= 0:
            raise StrategyFailure("nonpositive/nonfinite open equity", bar=bar, equity=pre_open)

        if bar in events or bar == next_force:
            # Close both legs at the registered or forced market order.
            if q0 or q1:
                for coin, quantity, opening in ((0, q0, o0), (1, q1, o1)):
                    if quantity:
                        fill = opening * (1 - math.copysign(slippage, quantity))
                        notional = abs(quantity * fill)
                        fee = FEE * notional
                        cash += quantity * fill - fee
                        fees += fee
                        slips += abs(quantity) * abs(fill - opening)
                        price_pnl += quantity * (opening - entry_mid[coin])
                q0 = q1 = 0.0
            if bar == next_force:
                forced_exits += 1
                next_force = -1
            else:
                sign = events[bar]
                if sign:
                    target = cash
                    if target <= 0:
                        raise StrategyFailure("nonpositive cash before entry", bar=bar, equity=target)
                    trailing0 = float(np.median(tape.quote_volume[bar - 288:bar, 0]))
                    trailing1 = float(np.median(tape.quote_volume[bar - 288:bar, 1]))
                    if target > .01 * trailing0 or target > .01 * trailing1:
                        capacity_cash += 1
                    else:
                        qty0 = math.floor(target / o0 / LOT) * LOT
                        qty1 = math.floor(target / o1 / LOT) * LOT
                        if min(qty0 * o0, qty1 * o1) < 5:
                            capacity_cash += 1
                        else:
                            q0, q1 = sign * qty0, -sign * qty1
                            for coin, quantity, opening in ((0, q0, o0), (1, q1, o1)):
                                fill = opening * (1 + math.copysign(slippage, quantity))
                                fee = FEE * abs(quantity * fill)
                                cash -= quantity * fill + fee
                                fees += fee
                                slips += abs(quantity) * abs(fill - opening)
                                entry_mid[coin] = opening
                            entered += 1
        for coin, rate in tape.funding.get(bar, ()):
            quantity = q0 if coin == 0 else q1
            payment = -quantity * (o0 if coin == 0 else o1) * rate
            if no_funding_credit:
                payment = min(payment, 0.0)
            cash += payment
            funding_total += payment

        post_open = cash + q0 * o0 + q1 * o1
        close_equity = cash + q0 * float(tape.close[bar, 0]) + q1 * float(tape.close[bar, 1])
        adverse = (cash + q0 * float(tape.low[bar, 0] if q0 > 0 else tape.high[bar, 0])
                   + q1 * float(tape.low[bar, 1] if q1 > 0 else tape.high[bar, 1]))
        regular_path[within] = max(pre_open, post_open, close_equity)
        adverse_path[within] = min(pre_open, post_open, adverse, close_equity)
        gross = abs(q0 * float(tape.close[bar, 0])) + abs(q1 * float(tape.close[bar, 1]))
        if (not math.isfinite(adverse) or adverse <= 0
                or (gross and adverse <= .005 * gross)):
            raise StrategyFailure("maintenance/ruin at adverse 5m mark",
                                  bar=bar, equity=adverse)
        if gross and close_equity > 0 and gross / close_equity > 3 and next_force < 0:
            next_force = bar + 1
        if (q0 or q1) and day >= 0:
            active_days[day] = True
        if within == 287:
            start = boundary[day]
            path_max = np.maximum.accumulate(np.maximum(regular_path, start))
            maxima[day] = max(start, float(np.max(regular_path)))
            minima[day] = min(start, float(np.min(adverse_path)))
            local_dd[day] = float(np.max(1.0 - adverse_path / path_max))

    if day != daily_count - 1 or q0 or q1 or next_force >= 0:
        raise StrategyFailure("terminal account position or day grid incomplete")
    boundary[daily_count] = cash
    factors = np.empty((daily_count, 4), dtype=np.float64)
    for j in range(daily_count):
        start, end = boundary[j:j + 2]
        factors[j] = (end / start,
                      max(maxima[j], start, end) / start,
                      min(minima[j], start, end) / start,
                      max(local_dd[j], 1 - min(minima[j], start, end) /
                          max(maxima[j], start, end)))
    metrics = _metrics_from_factors(factors)
    if not math.isclose(metrics["ending_equity"], cash, rel_tol=1e-8, abs_tol=1e-5):
        raise StrategyFailure("daily factor / account reconciliation failed")
    if not math.isclose(cash - INITIAL, price_pnl - fees - slips + funding_total,
                        rel_tol=1e-7, abs_tol=0.01):
        raise StrategyFailure("price, fee, slippage, funding reconciliation failed")
    return {"factors": factors, "metrics": metrics, "equity": cash,
            "fees": fees, "slippage": slips, "funding": funding_total,
            "price_pnl": price_pnl, "entered_weeks": entered,
            "capacity_cash_weeks": capacity_cash, "forced_exits": forced_exits,
            "active_days": active_days}


def evaluate(tape: PairTape, decisions: dict[date, int]) -> dict:
    scenarios = {
        "base": replay(tape, decisions, slippage=.0005, delay=1),
        "double_slippage": replay(tape, decisions, slippage=.0010, delay=1),
        "extra_bar_latency": replay(tape, decisions, slippage=.0005, delay=2),
        "no_funding_credit": replay(tape, decisions, slippage=.0005,
                                     delay=1, no_funding_credit=True),
    }
    comparator = replay(tape, decisions, slippage=.0005, delay=1, comparator=True)
    base = scenarios["base"]
    factors = base["factors"]
    dates = [START + timedelta(days=i) for i in range(len(factors))]
    monthly: dict[str, list[int]] = defaultdict(list)
    for i, day in enumerate(dates):
        monthly[day.strftime("%Y-%m")].append(i)
    if len(monthly) != 69:
        raise StrategyFailure("69-month grid changed")
    monthly_returns = {m: float(np.prod(factors[idx, 0]) - 1)
                       for m, idx in sorted(monthly.items())}
    positive = sum(v > 0 for v in monthly_returns.values())
    losing = longest = 0
    for value in monthly_returns.values():
        losing = losing + 1 if value <= 0 else 0
        longest = max(longest, losing)
    annual = {}
    for year in range(2021, 2027):
        idx = [i for i, day in enumerate(dates) if day.year == year]
        m = _metrics_from_factors(factors[idx])
        active = int(np.count_nonzero(base["active_days"][idx]))
        annual[str(year)] = {"days": len(idx), "active_days": active,
                             "return": m["ending_equity"] / INITIAL - 1,
                             "sharpe": m["sharpe"],
                             "pass": active > 0 and m["ending_equity"] > INITIAL
                             and m["sharpe"] is not None and m["sharpe"] > 0}
    holdout_idx = [i for i, day in enumerate(dates) if day >= date(2025, 1, 1)]
    holdout = _metrics_from_factors(factors[holdout_idx])
    daily_btc = tape.opening[::288, 0]
    regimes = {key: {"days": 0, "pnl": 0.0} for key in
               ("bull_high", "bull_low", "bear_high", "bear_low", "warmup")}
    equity = INITIAL
    for i, day in enumerate(dates):
        global_day = (day - BASE).days
        if global_day < 30:
            key = "warmup"
        else:
            prices = daily_btc[global_day - 30:global_day + 1]
            returns = np.diff(np.log(prices))
            bull = math.log(float(prices[-1] / prices[0])) > 0
            vol = float(np.std(returns, ddof=1) * math.sqrt(365))
            key = ("bull" if bull else "bear") + "_" + ("high" if vol >= .8 else "low")
        pnl = equity * (factors[i, 0] - 1)
        regimes[key]["days"] += 1
        regimes[key]["pnl"] += float(pnl)
        equity += pnl
    mc = monte_carlo_compressed(factors, paths=5000, block_days=21, seed=20261118)
    primary = base["metrics"]
    numeric = (primary["cagr"] >= .4 and primary["sharpe"] is not None
               and primary["sharpe"] > 1.5 and primary["max_drawdown"] <= .2
               and primary["profit_factor"] is not None and primary["profit_factor"] > 1.3)
    hold_numeric = (holdout["cagr"] >= .4 and holdout["sharpe"] is not None
                    and holdout["sharpe"] > 1.5 and holdout["max_drawdown"] <= .2
                    and holdout["profit_factor"] is not None and holdout["profit_factor"] > 1.3)
    gates = {
        "numeric": numeric,
        "weekly_entries": base["entered_weeks"] >= 30,
        "monthly": positive >= 42 and longest <= 3,
        "annual": all(row["pass"] for row in annual.values()),
        "retrospective_holdout": hold_numeric,
        "regimes": all(row["days"] >= 90 and row["pnl"] > 0
                       for key, row in regimes.items() if key != "warmup"),
        "monte_carlo": mc["joint_pass_fraction"] >= .8,
        "stresses": all(scenarios[name]["equity"] > INITIAL for name in
                        ("double_slippage", "extra_bar_latency", "no_funding_credit")),
        "independent_executable_oos": False,
    }
    return {
        "account": {name: {k: v for k, v in run.items()
                            if k not in ("factors", "active_days")}
                    for name, run in scenarios.items()},
        "comparator": {k: v for k, v in comparator.items()
                       if k not in ("factors", "active_days")},
        "scheduled_weeks": len(decisions),
        "scheduled_active": sum(bool(x) for x in decisions.values()),
        "monthly": {"positive": positive, "nonpositive": len(monthly) - positive,
                    "longest_nonpositive_streak": longest,
                    "worst_month": min(monthly_returns, key=monthly_returns.get),
                    "worst_return": min(monthly_returns.values()),
                    "returns": monthly_returns},
        "annual": annual,
        "holdout": holdout,
        "regimes": regimes,
        "monte_carlo": mc,
        "gate": {**gates, "all_passed": all(gates.values())},
    }
