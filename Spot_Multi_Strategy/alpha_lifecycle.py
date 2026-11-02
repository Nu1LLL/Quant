"""Multiple-testing-aware alpha activation, sizing and decay controls."""
from dataclasses import dataclass
from math import erfc, floor, sqrt

import numpy as np
import pandas as pd


KEY_COLUMNS = ("market", "venue", "symbol", "alpha_id")


@dataclass(frozen=True)
class LifecycleConfig:
    minimum_observations: int = 200
    rolling_window: int = 1000
    fdr: float = 0.05
    confidence_z: float = 1.6448536269514722  # one-sided 95%
    kelly_fraction: float = 0.25
    alpha_cap: float = 0.10
    symbol_cap: float = 0.20
    market_cap: float = 0.35
    cluster_cap: float = 0.20
    gross_cap: float = 1.00
    correlation_threshold: float = 0.70


def _hac_lag(n: int) -> int:
    return max(0, floor(4 * (n / 100) ** (2 / 9)))


def hac_mean_test(values, confidence_z=1.6448536269514722) -> dict:
    """One-sided test of a positive mean using a Bartlett HAC variance."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n < 2 or not np.isfinite(x).all():
        return {"observations": n, "mean": np.nan, "variance": np.nan,
                "standard_error": np.nan, "lower_confidence": np.nan,
                "p_value": np.nan, "hac_lag": 0}
    mean = float(x.mean())
    centered = x - mean
    lag = _hac_lag(n)
    long_run_variance = float(centered @ centered / n)
    for k in range(1, lag + 1):
        covariance = float(centered[k:] @ centered[:-k] / n)
        long_run_variance += 2 * (1 - k / (lag + 1)) * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = sqrt(long_run_variance / n)
    if standard_error <= 0:
        statistic = np.inf if mean > 0 else (-np.inf if mean < 0 else 0.0)
    else:
        statistic = mean / standard_error
    p_value = 0.5 * erfc(statistic / sqrt(2))
    return {
        "observations": n,
        "mean": mean,
        "variance": float(np.var(x, ddof=1)),
        "standard_error": standard_error,
        "lower_confidence": mean - confidence_z * standard_error,
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
        "hac_lag": lag,
    }


def benjamini_hochberg(p_values: pd.Series, fdr=0.05) -> pd.Series:
    """Return the discoveries from a fixed Benjamini-Hochberg procedure."""
    p_values = pd.Series(p_values, dtype=float)
    valid = p_values.dropna().sort_values(kind="mergesort")
    accepted = pd.Series(False, index=p_values.index)
    if valid.empty:
        return accepted
    thresholds = fdr * np.arange(1, len(valid) + 1) / len(valid)
    passing = valid.to_numpy() <= thresholds
    if passing.any():
        cutoff = valid.iloc[np.flatnonzero(passing)[-1]]
        accepted.loc[p_values.le(cutoff)] = True
    return accepted


def evaluate_lifecycle(returns: pd.DataFrame,
                       config=LifecycleConfig()) -> pd.DataFrame:
    required = {*KEY_COLUMNS, "timestamp", "net_return"}
    if not required.issubset(returns.columns):
        raise ValueError(f"Alpha returns missing {sorted(required - set(returns.columns))}")
    source = returns.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
    source = source.sort_values([*KEY_COLUMNS, "timestamp"])
    records = []
    for key, group in source.groupby(list(KEY_COLUMNS), sort=True):
        test = hac_mean_test(
            group["net_return"].iloc[-config.rolling_window:], config.confidence_z
        )
        records.append(dict(zip(KEY_COLUMNS, key), **test))
    result = pd.DataFrame(records)
    eligible = result["observations"].ge(config.minimum_observations)
    result["fdr_pass"] = False
    result.loc[eligible, "fdr_pass"] = benjamini_hochberg(
        result.loc[eligible, "p_value"], config.fdr
    )
    result["status"] = "DISABLED"
    result.loc[~eligible, "status"] = "SHADOW"
    active = eligible & result["fdr_pass"] & result["lower_confidence"].gt(0)
    result.loc[active, "status"] = "ACTIVE"
    raw_kelly = config.kelly_fraction * result["mean"].div(
        result["variance"].replace(0.0, np.nan)
    )
    result["raw_weight"] = raw_kelly.clip(lower=0.0, upper=config.alpha_cap).fillna(0.0)
    result.loc[result["status"].ne("ACTIVE"), "raw_weight"] = 0.0
    return result


def _correlation_clusters(returns: pd.DataFrame, active_keys: pd.DataFrame,
                          threshold: float) -> dict:
    work = returns.copy()
    work["series_id"] = work[list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1)
    active_ids = active_keys[list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1).tolist()
    pivot = work[work["series_id"].isin(active_ids)].pivot_table(
        index="timestamp", columns="series_id", values="net_return", aggfunc="last"
    )
    correlation = pivot.corr(min_periods=50).abs()
    parent = {identifier: identifier for identifier in active_ids}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for i, left in enumerate(active_ids):
        for right in active_ids[i + 1:]:
            if left in correlation.index and right in correlation and correlation.loc[left, right] > threshold:
                union(left, right)
    return {identifier: find(identifier) for identifier in active_ids}


def allocate_active_weights(lifecycle: pd.DataFrame, returns: pd.DataFrame,
                            config=LifecycleConfig()) -> pd.DataFrame:
    """Apply correlation, symbol, market and portfolio caps to active alphas."""
    result = lifecycle.copy()
    result["weight"] = result["raw_weight"].astype(float)
    active = result["weight"].gt(0)
    result["cluster_id"] = "INACTIVE"
    if not active.any():
        return result
    clusters = _correlation_clusters(returns, result.loc[active], config.correlation_threshold)
    identifiers = result.loc[active, list(KEY_COLUMNS)].astype(str).agg("|".join, axis=1)
    result.loc[active, "cluster_id"] = identifiers.map(clusters).to_numpy()

    def cap_groups(columns, cap):
        totals = result.groupby(columns)["weight"].transform("sum")
        scale = np.minimum(1.0, cap / totals.replace(0.0, np.nan))
        result["weight"] *= scale.fillna(1.0)

    cap_groups(["cluster_id"], config.cluster_cap)
    cap_groups(["market", "venue", "symbol"], config.symbol_cap)
    cap_groups(["market"], config.market_cap)
    gross = result["weight"].abs().sum()
    if gross > config.gross_cap:
        result["weight"] *= config.gross_cap / gross
    return result
