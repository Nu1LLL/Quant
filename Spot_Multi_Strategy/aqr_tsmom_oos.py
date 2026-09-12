"""AQR post-publication monthly TSMOM validation helpers."""
import numpy as np
import pandas as pd

import portfolio_metrics


def load_factor_csv(path):
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "TSMOM", "TSMOM_CM", "TSMOM_EQ", "TSMOM_FI", "TSMOM_FX"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing AQR factor columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame["date"].dt.to_period("M").duplicated().any():
        raise ValueError("Duplicate AQR factor month")
    return frame.set_index("date")


def apply_scenario(raw_returns, leverage, annual_cost=0.02, annual_financing=0.04):
    raw_returns = pd.Series(raw_returns).dropna().astype(float).sort_index()
    implementation_cost = leverage * annual_cost / 12.0
    financing_cost = max(leverage - 1.0, 0.0) * annual_financing / 12.0
    result = leverage * raw_returns - implementation_cost - financing_cost
    result.name = "net_return"
    return result


def monthly_metrics(returns, initial_capital=10000.0):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    simulation = pd.DataFrame({"open_time": returns.index, "net_pnl": returns.values})
    metrics = portfolio_metrics.calculate_extended_metrics(simulation, initial_capital)
    yearly = portfolio_metrics.calendar_year_returns(simulation["open_time"], returns)
    counts = simulation.assign(year=simulation["open_time"].dt.year).groupby("year").size()
    complete = yearly[counts >= 12]
    rolling_mean = returns.rolling(36, min_periods=36).mean()
    rolling_std = returns.rolling(36, min_periods=36).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(12.0)
    metrics.update({
        "positive_complete_year_ratio": float((complete > 0).mean()) if len(complete) else 0.0,
        "complete_year_count": int(len(complete)),
        "worst_rolling_3y_sharpe": (
            float(rolling_sharpe.min()) if rolling_sharpe.notna().any() else np.nan
        ),
    })
    return metrics


def annual_walk_forward(returns):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    records = []
    for year, values in returns.groupby(returns.index.year):
        if len(values) != 12:
            continue
        metrics = monthly_metrics(values)
        records.append({"year": int(year), "observations": len(values), **metrics})
    folds = pd.DataFrame(records)
    if folds.empty:
        return folds, False
    folds["fold_pass"] = (
        (folds["total_return"] > 0)
        & (folds["profit_factor"] > 1.0)
        & (folds["max_drawdown"] >= -0.20)
    )
    return folds, bool(len(folds) >= 10 and folds["fold_pass"].mean() >= 0.80)


def evaluate(returns, independent_oos=True, costs_included=True):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    metrics = monthly_metrics(returns)
    folds, walk_forward_passed = annual_walk_forward(returns)
    years = metrics["elapsed_days"] / 365.25
    checks = {
        "sample_at_least_10y": years >= 10.0,
        "independent_oos": bool(independent_oos),
        "costs_included": bool(costs_included),
        "cagr_at_least_40pct": metrics["cagr"] >= 0.40,
        "sharpe_above_1_5": metrics["sharpe_ratio"] > 1.5,
        "max_drawdown_at_most_20pct": metrics["max_drawdown"] >= -0.20,
        "profit_factor_above_1_3": metrics["profit_factor"] > 1.3,
        "walk_forward_passed": walk_forward_passed,
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values()),
            "walk_forward_folds": folds}


def regime_metrics(returns, labels):
    frame = pd.concat([pd.Series(returns).rename("return"),
                       pd.Series(labels).rename("regime")], axis=1).dropna()
    records = []
    for regime, group in frame.groupby("regime"):
        records.append({"regime": regime, "observations": len(group),
                        **monthly_metrics(group["return"])})
    return pd.DataFrame(records)


def circular_block_monte_carlo(returns, simulations=5000, block_length=3, seed=27182818):
    sample = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(sample) < 2 or simulations < 1 or block_length < 1:
        raise ValueError("Returns, simulations and block length must be valid")
    n = len(sample)
    blocks = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(simulations, blocks))
    indices = (starts[:, :, None] + np.arange(block_length)) % n
    paths = sample[indices].reshape(simulations, -1)[:, :n]
    equity = np.cumprod(1.0 + paths, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    max_drawdown = np.min(equity / running_max - 1.0, axis=1)
    final = equity[:, -1]
    cagr = np.where(final > 0, final ** (12.0 / n) - 1.0, -1.0)
    std = paths.std(axis=1)
    sharpe = np.divide(paths.mean(axis=1) * np.sqrt(12.0), std,
                       out=np.zeros(simulations), where=std > 0)
    gains = np.where(paths > 0, paths, 0.0).sum(axis=1)
    losses = -np.where(paths < 0, paths, 0.0).sum(axis=1)
    profit_factor = np.divide(gains, losses, out=np.full(simulations, np.inf),
                              where=losses > 0)
    joint = ((cagr >= 0.40) & (sharpe > 1.5) & (max_drawdown >= -0.20)
             & (profit_factor > 1.3))
    return {
        "simulations": int(simulations), "block_length_months": int(block_length),
        "cagr_p05": float(np.percentile(cagr, 5)), "cagr_median": float(np.median(cagr)),
        "sharpe_p05": float(np.percentile(sharpe, 5)),
        "sharpe_median": float(np.median(sharpe)),
        "max_drawdown_p05": float(np.percentile(max_drawdown, 5)),
        "profit_factor_p05": float(np.percentile(profit_factor, 5)),
        "joint_target_probability": float(joint.mean()),
    }
