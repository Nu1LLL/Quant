"""Frozen account model for the CVSIX market-neutral-income NAV audit."""
import numpy as np
import pandas as pd

import strict_validation


FROZEN_LEVERAGES = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0)


def validate_prices(prices, maximum_gap_days=10):
    prices = pd.Series(prices).dropna().astype(float).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate CVSIX NAV date")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("CVSIX adjusted NAV must be finite and positive")
    gaps = prices.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > maximum_gap_days:
        raise ValueError(f"Abnormally long CVSIX NAV gap: {int(gaps.max())} days")
    return prices


def run_scenario(
    prices, leverage, entry_sales_load=0.0, exit_sales_load=0.0,
    leg_cost=0.0010, annual_financing=0.04, initial_capital=10000.0,
):
    prices = validate_prices(prices)
    if leverage not in FROZEN_LEVERAGES:
        raise ValueError("Leverage is outside the preregistered scenarios")
    for name, value in (
        ("entry_sales_load", entry_sales_load),
        ("exit_sales_load", exit_sales_load),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    gross = prices.pct_change(fill_method=None).dropna()
    external_cost = pd.Series(0.0, index=gross.index, name="external_cost")
    external_cost.iloc[0] = leverage * (leg_cost + entry_sales_load)
    external_cost.iloc[-1] += leverage * (leg_cost + exit_sales_load)
    financing_cost = pd.Series(
        max(leverage - 1.0, 0.0) * annual_financing / 252.0,
        index=gross.index,
        name="financing_cost",
    )
    net = leverage * gross - external_cost - financing_cost
    failure = np.flatnonzero(net.to_numpy() <= -1.0)
    if len(failure):
        first = int(failure[0])
        net.iloc[first] = -1.0
        if first + 1 < len(net):
            net.iloc[first + 1:] = 0.0
    net.name = "net_return"
    detail = pd.DataFrame({
        "gross_fund_return": gross,
        "leverage": leverage,
        "external_cost": external_cost,
        "financing_cost": financing_cost,
        "net_return": net,
    })
    detail["equity"] = initial_capital * (1.0 + net).cumprod()
    detail["solvent"] = detail["equity"] > 0.0
    return net, detail


def evaluate_actual_path(
    returns, solvent, account_executable, strategy_continuity
):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True
    )
    folds = result["walk_forward_folds"]
    long_walk_forward = bool(
        len(folds) >= 30 and len(folds) > 0 and folds["fold_pass"].mean() >= 0.80
    )
    elapsed_years = result["metrics"]["elapsed_days"] / 365.25
    rolling_mean = returns.rolling(756, min_periods=756).mean()
    rolling_std = returns.rolling(756, min_periods=756).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252.0)
    worst_rolling = (
        float(rolling_sharpe.min()) if rolling_sharpe.notna().any() else np.nan
    )
    checks = dict(result["checks"])
    checks.pop("sample_at_least_5y", None)
    checks["sample_at_least_34_5y"] = elapsed_years >= 34.5
    checks["walk_forward_passed"] = long_walk_forward
    checks["worst_rolling_3y_sharpe_at_least_1"] = worst_rolling >= 1.0
    checks["account_executable_for_10000"] = bool(account_executable)
    checks["same_strategy_for_full_sample"] = bool(strategy_continuity)
    checks["solvent"] = bool(solvent)
    result["checks"] = checks
    result["metrics"]["worst_rolling_3y_sharpe"] = worst_rolling
    result["passed"] = all(checks.values())
    return result


def circular_block_monte_carlo_chunked(
    returns, simulations=5000, block_length=63, seed=19900904,
    batch_size=100,
):
    """Memory-bounded equivalent of the shared circular block bootstrap."""
    sample = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(sample) < 2 or simulations < 1 or block_length < 1 or batch_size < 1:
        raise ValueError("Returns, simulations, block length and batch must be valid")
    n = len(sample)
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    records = {name: [] for name in ("cagr", "sharpe", "drawdown", "pf")}
    complete = 0
    while complete < simulations:
        batch = min(batch_size, simulations - complete)
        starts = rng.integers(0, n, size=(batch, blocks))
        indices = (starts[:, :, None] + offsets) % n
        paths = sample[indices].reshape(batch, -1)[:, :n]
        equity = np.cumprod(1.0 + paths, axis=1)
        running_max = np.maximum.accumulate(equity, axis=1)
        drawdown = np.min(equity / running_max - 1.0, axis=1)
        years = n / 252.0
        final = equity[:, -1]
        cagr = np.where(final > 0, final ** (1.0 / years) - 1.0, -1.0)
        std = paths.std(axis=1)
        sharpe = np.divide(
            paths.mean(axis=1) * np.sqrt(252.0), std,
            out=np.zeros(batch), where=std > 0,
        )
        gains = np.where(paths > 0, paths, 0.0).sum(axis=1)
        losses = -np.where(paths < 0, paths, 0.0).sum(axis=1)
        profit_factor = np.divide(
            gains, losses, out=np.full(batch, np.inf), where=losses > 0,
        )
        records["cagr"].append(cagr)
        records["sharpe"].append(sharpe)
        records["drawdown"].append(drawdown)
        records["pf"].append(profit_factor)
        complete += batch
    cagr = np.concatenate(records["cagr"])
    sharpe = np.concatenate(records["sharpe"])
    drawdown = np.concatenate(records["drawdown"])
    profit_factor = np.concatenate(records["pf"])
    joint = (
        (cagr >= 0.40) & (sharpe > 1.5) & (drawdown >= -0.20)
        & (profit_factor > 1.3)
    )
    return {
        "simulations": int(simulations),
        "block_length": int(block_length),
        "cagr_p05": float(np.percentile(cagr, 5)),
        "cagr_median": float(np.median(cagr)),
        "sharpe_p05": float(np.percentile(sharpe, 5)),
        "sharpe_median": float(np.median(sharpe)),
        "max_drawdown_p05": float(np.percentile(drawdown, 5)),
        "profit_factor_p05": float(np.percentile(profit_factor, 5)),
        "joint_target_probability": float(joint.mean()),
    }
