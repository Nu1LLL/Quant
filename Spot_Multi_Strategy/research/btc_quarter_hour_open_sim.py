"""One-shot transfer test of preregistered BTC quarter-hour drift."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .weekday_seasonality_data import (
    MONTHS, PRICE_FIELDS, PRICE_MANIFESTS, _pins, _price_month,
)


ROOT = Path(__file__).parent
PREREG = ROOT / "preregistrations/2026-11-08_btc_quarter_hour_open.md"
OUTPUT = ROOT / "results/btc_quarter_hour_open_2026_11_08.json"
START = date(2021, 1, 1)
END = date(2026, 10, 1)
INITIAL = 10_000.0
FEE = 0.0005
BASE_SLIPPAGE = 0.0001
SEED = 20261119
PATHS = 5_000
BLOCK = 21
TRADES_PER_DAY = 72


def day_profile(opening: np.ndarray, low: np.ndarray, *, fee: float = FEE,
                slippage: float = BASE_SLIPPAGE, lag_bars: int = 0) -> tuple[float, float, float, float, float]:
    """Net day factor, low factor, high factor, local DD and gross factor."""
    if len(opening) != 288 or len(low) != 288 or lag_bars not in (0, 1):
        raise ValueError("incomplete day or undefined latency")
    per_roundtrip_cost = (1 - slippage) * (1 - fee) / ((1 + slippage) * (1 + fee))
    factor = gross = low_factor = high_factor = peak = 1.0
    local_dd = 0.0
    for hour in range(24):
        for quarter in (3, 6, 9):
            entry_idx = hour * 12 + quarter + lag_bars
            exit_idx = entry_idx + 1
            entry = opening[entry_idx]
            exit_price = opening[exit_idx]
            if not (math.isfinite(entry) and math.isfinite(exit_price)
                    and math.isfinite(low[entry_idx]) and min(entry, exit_price, low[entry_idx]) > 0):
                raise ValueError("invalid replay price")
            adverse = factor * (low[entry_idx] / entry) * per_roundtrip_cost
            low_factor = min(low_factor, adverse)
            local_dd = max(local_dd, 1 - adverse / peak)
            gross *= exit_price / entry
            factor *= (exit_price / entry) * per_roundtrip_cost
            local_dd = max(local_dd, 1 - factor / peak)
            peak = max(peak, factor)
            high_factor = max(high_factor, factor)
    return factor, low_factor, high_factor, local_dd, gross


def path_metrics(factors: np.ndarray, low: np.ndarray, high: np.ndarray,
                 local_dd: np.ndarray) -> dict:
    n = len(factors)
    if n < 2 or any(len(x) != n for x in (low, high, local_dd)) or np.any(factors <= 0):
        raise ValueError("bad path arrays")
    log_equity = np.concatenate(([0.0], np.cumsum(np.log(factors))))
    equity = INITIAL * np.exp(log_equity)
    starts = equity[:-1]
    previous_peak = np.maximum.accumulate(
        np.concatenate(([INITIAL], (starts * high)[:-1])))
    drawdown = float(max(np.max(1 - starts * low / previous_peak), np.max(local_dd)))
    daily_return = factors - 1
    std = float(np.std(daily_return, ddof=1))
    sharpe = float(np.mean(daily_return) / std * math.sqrt(365)) if std > 0 else None
    pnl = np.diff(equity)
    gains = float(np.sum(pnl[pnl > 0]))
    losses = float(-np.sum(pnl[pnl < 0]))
    profit_factor = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    return {
        "cagr": float(math.expm1(log_equity[-1] * 365.25 / n)),
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "profit_factor": profit_factor,
        "final_equity": float(equity[-1]),
        "practical_ruin_below_100": bool(np.any(equity < 100)),
        "underflow": bool(np.any(equity == 0)),
    }


def numeric_gate(metrics: dict) -> bool:
    return (metrics["cagr"] >= .4 and metrics["sharpe"] is not None
            and metrics["sharpe"] > 1.5 and metrics["max_drawdown"] <= .2
            and metrics["profit_factor"] > 1.3 and not metrics["practical_ruin_below_100"])


def _bootstrap(factors: np.ndarray, low: np.ndarray, high: np.ndarray,
               local_dd: np.ndarray) -> dict:
    rng = np.random.Generator(np.random.PCG64(SEED))
    n = len(factors)
    blocks = math.ceil(n / BLOCK)
    offsets = np.arange(BLOCK)
    samples = np.empty((PATHS, 4), dtype=float)
    wins = ruined = undefined = 0
    for k in range(PATHS):
        starts = rng.integers(0, n, size=blocks)
        index = ((starts[:, None] + offsets) % n).ravel()[:n]
        m = path_metrics(factors[index], low[index], high[index], local_dd[index])
        samples[k] = (m["cagr"], m["sharpe"] if m["sharpe"] is not None else np.nan,
                      m["max_drawdown"], m["profit_factor"])
        wins += numeric_gate(m)
        ruined += m["practical_ruin_below_100"]
        undefined += m["sharpe"] is None or not math.isfinite(m["profit_factor"])
    quantiles = {}
    for column, name in enumerate(("cagr", "sharpe", "max_drawdown", "profit_factor")):
        values = samples[:, column]
        finite = values[np.isfinite(values)]
        quantiles[name] = [float(x) for x in np.quantile(finite, [.05, .5, .95])] if len(finite) else None
    return {"paths": PATHS, "block_days": BLOCK, "seed": SEED,
            "joint_passes": wins, "joint_pass_fraction": wins / PATHS,
            "practically_ruined_paths": ruined, "undefined_paths": undefined,
            "p05_p50_p95": quantiles}


def _load() -> tuple[np.ndarray, dict]:
    pins = _pins(PRICE_MANIFESTS, PRICE_FIELDS, "5m")
    wanted = [pins[(month, "BTCUSDT")] for month in MONTHS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        months = list(pool.map(lambda pair: _price_month("BTCUSDT", *pair),
                               zip(MONTHS, wanted)))
    values = np.concatenate(months)
    if len(values) != (END - START).days * 288:
        raise ValueError("full registered day grid missing")
    return values, {month: row["sha256"] for month, row in zip(MONTHS, wanted)}


def _folds(factors: np.ndarray, lows: np.ndarray, highs: np.ndarray,
           local: np.ndarray) -> dict:
    out = {}
    for year in range(2021, 2027):
        begin = (date(year, 1, 1) - START).days
        finish = min((date(year + 1, 1) - START).days, len(factors))
        out[str(year)] = path_metrics(factors[begin:finish], lows[begin:finish],
                                      highs[begin:finish], local[begin:finish])
    start = (date(2024, 1, 1) - START).days
    out["post_paper_2024_2026"] = path_metrics(factors[start:], lows[start:],
                                                 highs[start:], local[start:])
    return out


def _months(factors: np.ndarray) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for i, factor in enumerate(factors):
        groups[str(START + timedelta(days=i))[:7]].append(float(factor))
    monthly = {name: float(math.expm1(sum(math.log(x) for x in group)))
               for name, group in groups.items()}
    streak = worst_streak = 0
    for value in monthly.values():
        streak = 0 if value > 0 else streak + 1
        worst_streak = max(streak, worst_streak)
    return {"values": monthly, "positive_count": sum(x > 0 for x in monthly.values()),
            "longest_nonpositive_streak": worst_streak}


def _states(opening: np.ndarray, factors: np.ndarray) -> dict:
    daily_open = opening[::288]
    prior_returns = np.log(daily_open[1:] / daily_open[:-1])
    equity = INITIAL * np.exp(np.concatenate(([0.0], np.cumsum(np.log(factors)))))
    pnl = np.diff(equity)
    out: dict[str, dict] = defaultdict(lambda: {"days": 0, "net_dollar_pnl": 0.0})
    for day in range(30, len(factors)):
        history = prior_returns[day - 30:day]
        bull = bool(np.sum(history) > 0)
        high = bool(np.std(history, ddof=1) * math.sqrt(365) >= .8)
        key = ("bull" if bull else "bear") + "_" + ("high" if high else "low")
        out[key]["days"] += 1
        out[key]["net_dollar_pnl"] += float(pnl[day])
    return dict(out)


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("formal quarter-hour result already sealed")
    prereg_hash = sha256(PREREG.read_bytes()).hexdigest()
    source, pins = _load()
    days = (END - START).days
    opening = source[:, 0]
    low_price = source[:, 2]
    cases = {"base": (BASE_SLIPPAGE, 0),
             "double_slippage": (BASE_SLIPPAGE * 2, 0),
             "one_bar_latency": (BASE_SLIPPAGE, 1)}
    arrays = {}
    for name, (slip, lag) in cases.items():
        profiles = np.asarray([day_profile(opening[i*288:(i+1)*288],
                                          low_price[i*288:(i+1)*288],
                                          slippage=slip, lag_bars=lag)
                               for i in range(days)])
        arrays[name] = tuple(profiles[:, i] for i in range(5))
    factor, low, high, local_dd, gross = arrays["base"]
    metrics = path_metrics(factor, low, high, local_dd)
    folds = _folds(factor, low, high, local_dd)
    months = _months(factor)
    states = _states(opening, factor)
    bootstrap = _bootstrap(factor, low, high, local_dd)
    stress = {name: path_metrics(*profile[:4]) for name, profile in arrays.items()
              if name != "base"}
    gross_log = float(np.sum(np.log(gross)))
    net_log = float(np.sum(np.log(factor)))
    checks = {
        "numeric": numeric_gate(metrics),
        "annual_forward": all(folds[str(y)]["final_equity"] > INITIAL
                              and folds[str(y)]["sharpe"] is not None
                              and folds[str(y)]["sharpe"] > 0 for y in range(2021, 2026))
                          and folds["2026"]["final_equity"] > INITIAL,
        "post_paper_each_year": all(folds[str(y)]["final_equity"] > INITIAL
                                    and folds[str(y)]["sharpe"] is not None
                                    and folds[str(y)]["sharpe"] > 0 for y in (2024, 2025, 2026)),
        "monthly": months["positive_count"] >= math.ceil(len(months["values"]) * .6)
                   and months["longest_nonpositive_streak"] <= 3,
        "states": len(states) == 4 and all(row["days"] >= 90
                                         and row["net_dollar_pnl"] > 0 for row in states.values()),
        "bootstrap": bootstrap["joint_pass_fraction"] >= .8,
        "stress": all(row["final_equity"] > INITIAL for row in stress.values()),
        "independent_executable_oos": False,
    }
    result = {
        "experiment_id": "btc_quarter_hour_open_2026_11_08",
        "prereg_sha256": prereg_hash,
        "source_months": len(pins),
        "source_pin_sha256": sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(),
        "period": [START.isoformat(), END.isoformat()],
        "days": days,
        "intended_trades": days * TRADES_PER_DAY,
        "fee_per_side": FEE,
        "slippage_per_side": BASE_SLIPPAGE,
        "leverage": 1,
        "gross_total_log_return": gross_log,
        "net_total_log_return": net_log,
        "gross_geometric_trade_bps": float(math.expm1(gross_log / (days * TRADES_PER_DAY)) * 10_000),
        "net_geometric_trade_bps": float(math.expm1(net_log / (days * TRADES_PER_DAY)) * 10_000),
        "primary_metrics": metrics,
        "annual_folds": folds,
        "months": months,
        "market_states": states,
        "bootstrap": bootstrap,
        "stress": stress,
        "checks": checks,
        "full_gate_pass": all(checks.values()),
    }
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"days": result["days"], "intended_trades": result["intended_trades"],
                      "primary_metrics": result["primary_metrics"],
                      "checks": result["checks"]}, sort_keys=True))
