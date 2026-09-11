"""Strict OOS, walk-forward, regime and block-Monte-Carlo validation."""
import numpy as np
import pandas as pd

from cboe_options_benchmark import summarize_returns


def metrics_for_returns(returns):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    _, _, metrics = summarize_returns(returns)
    return metrics


def annual_walk_forward(returns, minimum_observations=240):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    records = []
    for year, values in returns.groupby(returns.index.year):
        if len(values) < minimum_observations:
            continue
        metrics = metrics_for_returns(values)
        records.append({"year": int(year), "observations": len(values), **metrics})
    folds = pd.DataFrame(records)
    if folds.empty:
        return folds, False
    fold_pass = (
        (folds["total_return"] > 0)
        & (folds["profit_factor"] > 1.0)
        & (folds["max_drawdown"] >= -0.20)
    )
    folds["fold_pass"] = fold_pass
    passed = len(folds) >= 5 and float(fold_pass.mean()) >= 0.80
    return folds, bool(passed)


def regime_metrics(returns, regimes, minimum_observations=20):
    frame = pd.concat(
        [pd.Series(returns).rename("return"), pd.Series(regimes).rename("regime")],
        axis=1, join="inner"
    ).dropna()
    records = []
    for regime, group in frame.groupby("regime"):
        if len(group) < minimum_observations:
            continue
        metrics = metrics_for_returns(group["return"])
        records.append({"regime": str(regime), "observations": len(group), **metrics})
    return pd.DataFrame(records)


def circular_block_monte_carlo(
    returns, simulations=5000, block_length=21, seed=9227465
):
    sample = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(sample) < 2 or simulations < 1 or block_length < 1:
        raise ValueError("Returns, simulations and block length must be valid")
    n = len(sample)
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(simulations, blocks))
    indices = (starts[:, :, None] + offsets) % n
    paths = sample[indices].reshape(simulations, -1)[:, :n]
    equity = np.cumprod(1.0 + paths, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    max_drawdown = np.min(equity / running_max - 1.0, axis=1)
    years = n / 252.0
    final = equity[:, -1]
    cagr = np.where(final > 0, final ** (1.0 / years) - 1.0, -1.0)
    std = paths.std(axis=1)
    sharpe = np.divide(
        paths.mean(axis=1) * np.sqrt(252.0), std,
        out=np.zeros(simulations), where=std > 0
    )
    gains = np.where(paths > 0, paths, 0.0).sum(axis=1)
    losses = -np.where(paths < 0, paths, 0.0).sum(axis=1)
    profit_factor = np.divide(
        gains, losses, out=np.full(simulations, np.inf), where=losses > 0
    )
    joint_pass = (
        (cagr >= 0.40) & (sharpe > 1.5) & (max_drawdown >= -0.20)
        & (profit_factor > 1.3)
    )
    return {
        "simulations": int(simulations),
        "block_length": int(block_length),
        "cagr_p05": float(np.percentile(cagr, 5)),
        "cagr_median": float(np.median(cagr)),
        "sharpe_p05": float(np.percentile(sharpe, 5)),
        "sharpe_median": float(np.median(sharpe)),
        "max_drawdown_p05": float(np.percentile(max_drawdown, 5)),
        "profit_factor_p05": float(np.percentile(profit_factor, 5)),
        "joint_target_probability": float(joint_pass.mean()),
    }


def evaluate_strict_oos(
    returns, independent_oos, costs_included, minimum_years=5.0
):
    returns = pd.Series(returns).dropna().astype(float).sort_index()
    metrics = metrics_for_returns(returns)
    elapsed_years = metrics["elapsed_days"] / 365.25
    folds, walk_forward_passed = annual_walk_forward(returns)
    checks = {
        "sample_at_least_5y": elapsed_years >= minimum_years,
        "independent_oos": bool(independent_oos),
        "costs_included": bool(costs_included),
        "cagr_at_least_40pct": metrics["cagr"] >= 0.40,
        "sharpe_above_1_5": metrics["sharpe_ratio"] > 1.5,
        "max_drawdown_at_most_20pct": metrics["max_drawdown"] >= -0.20,
        "profit_factor_above_1_3": metrics["profit_factor"] > 1.3,
        "walk_forward_passed": walk_forward_passed,
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "walk_forward_folds": folds,
    }
