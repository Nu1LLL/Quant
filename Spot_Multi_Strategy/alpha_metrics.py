"""Alpha预测质量的统计工具。

这个模块专门产生"前瞻收益标签"（forward return），并且只在这里产生：
alphas/包下的所有信号计算函数完全不导入、不依赖本模块，保证
"用于研究的未来标签"和"用于交易的alpha分数"物理上是分离的两条代码路径。

forward_return使用t+delay到t+delay+horizon之间的开盘价，
对应engine.py里"信号在收盘形成、下一根K线开盘成交"的真实执行方式，
只作为研究标签使用，绝不会被回填进任何alpha的计算函数。
"""
import numpy as np
import pandas as pd


def forward_return(df, horizon, delay=1):
    """从t+delay根K线开盘到t+delay+horizon根K线开盘的持有收益，仅作研究标签。"""
    open_price = df["open"]
    entry_price = open_price.shift(-delay)
    exit_price = open_price.shift(-(delay + horizon))
    return exit_price / entry_price - 1


def _paired(signal, forward_ret):
    paired = pd.concat(
        [signal.rename("signal"), forward_ret.rename("forward_return")],
        axis=1
    ).dropna()
    return paired


def pearson_ic(signal, forward_ret):
    paired = _paired(signal, forward_ret)
    if len(paired) < 3:
        return np.nan, len(paired)
    ic = paired["signal"].corr(paired["forward_return"], method="pearson")
    return ic, len(paired)


def spearman_ic(signal, forward_ret):
    # 手动用秩相关代替pandas的method="spearman"，因为该实现在没有安装
    # scipy时会抛错；秩的Pearson相关系数在数学上就是Spearman相关系数。
    paired = _paired(signal, forward_ret)
    if len(paired) < 3:
        return np.nan, len(paired)
    ranked_signal = paired["signal"].rank()
    ranked_return = paired["forward_return"].rank()
    ic = ranked_signal.corr(ranked_return, method="pearson")
    return ic, len(paired)


def ic_t_stat(ic, observations):
    if pd.isna(ic) or observations < 3 or abs(ic) >= 1.0:
        return np.nan
    return ic * np.sqrt(observations - 2) / np.sqrt(1 - ic ** 2)


def rolling_ic_positive_ratio(signal, forward_ret, window=250):
    paired = _paired(signal, forward_ret)
    if len(paired) < window:
        return np.nan, 0

    rolling_ic = paired["signal"].rolling(window).corr(
        paired["forward_return"]
    )
    rolling_ic = rolling_ic.dropna()

    if rolling_ic.empty:
        return np.nan, 0

    return float((rolling_ic > 0).mean()), int(len(rolling_ic))


def bucket_returns(signal, forward_ret, buckets=3):
    paired = _paired(signal, forward_ret)
    if len(paired) < buckets * 10:
        return {"low": np.nan, "mid": np.nan, "high": np.nan, "spread": np.nan}

    try:
        labels = pd.qcut(
            paired["signal"],
            buckets,
            labels=["low", "mid", "high"][:buckets],
            duplicates="drop"
        )
    except ValueError:
        return {"low": np.nan, "mid": np.nan, "high": np.nan, "spread": np.nan}

    grouped = paired.groupby(labels, observed=True)["forward_return"].mean()
    low = float(grouped.get("low", np.nan))
    mid = float(grouped.get("mid", np.nan))
    high = float(grouped.get("high", np.nan))
    spread = (
        high - low
        if not (pd.isna(high) or pd.isna(low))
        else np.nan
    )

    return {"low": low, "mid": mid, "high": high, "spread": spread}


def turnover_proxy(signal):
    diffs = signal.diff().abs().dropna()
    if diffs.empty:
        return np.nan
    return float(diffs.mean())


def cost_adjusted_spread(
    spread,
    turnover,
    horizon,
    fee_rate,
    slippage_rate,
    cost_multiplier=1.0
):
    """用turnover_proxy粗略估算换手成本在该周期内对分桶价差的侵蚀。

    这是一个透明的估计，不是精确回测：假设信号每根K线的变化幅度
    turnover_proxy会转化为同等幅度的仓位调整，每次调整双边各产生
    一次fee+slippage。
    """
    if pd.isna(spread) or pd.isna(turnover):
        return np.nan

    per_bar_cost = turnover * (fee_rate + slippage_rate) * 2 * cost_multiplier
    estimated_drag = per_bar_cost * horizon

    return spread - estimated_drag


def compute_regime_labels(df, window=20):
    """用Kaufman效率比率把样本切成trending/mixed/ranging三个regime桶，
    只用于研究报告里的regime细分展示，不是一个交易规则。
    """
    close = df["close"]
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    efficiency_ratio = (
        net_change / path_length.replace(0, np.nan)
    ).clip(0, 1)

    valid = efficiency_ratio.dropna()
    if len(valid) < 30:
        return pd.Series(index=df.index, dtype="object")

    low_cut, high_cut = valid.quantile([1 / 3, 2 / 3])

    labels = pd.cut(
        efficiency_ratio,
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["ranging", "mixed", "trending"]
    )

    return labels


def performance_by_regime(signal, forward_ret, regime_labels):
    paired = pd.concat(
        [
            signal.rename("signal"),
            forward_ret.rename("forward_return"),
            regime_labels.rename("regime")
        ],
        axis=1
    ).dropna()

    result = {}
    for regime in ["trending", "mixed", "ranging"]:
        subset = paired[paired["regime"] == regime]
        if len(subset) < 10:
            result[regime] = {"ic": np.nan, "observations": len(subset)}
            continue

        ic = subset["signal"].corr(subset["forward_return"])
        result[regime] = {"ic": ic, "observations": len(subset)}

    return result


def performance_by_year(signal, forward_ret, open_time):
    open_time_series = pd.Series(
        pd.to_datetime(open_time, utc=True).values,
        index=signal.index,
        name="open_time"
    )

    paired = pd.concat(
        [
            signal.rename("signal"),
            forward_ret.rename("forward_return"),
            open_time_series
        ],
        axis=1
    ).dropna()

    if paired.empty:
        return pd.DataFrame(
            columns=["year", "observations", "mean_forward_return", "ic"]
        )

    paired["year"] = paired["open_time"].dt.year

    records = []
    for year, group in paired.groupby("year"):
        ic = (
            group["signal"].corr(group["forward_return"])
            if len(group) >= 10
            else np.nan
        )
        records.append({
            "year": int(year),
            "observations": int(len(group)),
            "mean_forward_return": float(
                group["forward_return"].mean()
            ),
            "ic": ic
        })

    return pd.DataFrame(records)


def evaluate_alpha(
    symbol,
    alpha_signal,
    df,
    horizons=(1, 3, 6),
    delay=1,
    fee_rate=0.001,
    slippage_rate=0.0005,
    rolling_window=250,
    regime_labels=None
):
    """对一个AlphaSignal按多个前瞻窗口打分，返回一行一个字典的列表。"""
    open_time = pd.to_datetime(df["open_time"], utc=True)
    normalized = alpha_signal.normalized_signal
    turnover = turnover_proxy(normalized)

    if regime_labels is None:
        regime_labels = compute_regime_labels(df)

    rows = []
    for horizon in horizons:
        forward_ret = forward_return(df, horizon=horizon, delay=delay)

        pearson, observations = pearson_ic(normalized, forward_ret)
        spearman, _ = spearman_ic(normalized, forward_ret)
        t_stat = ic_t_stat(pearson, observations)
        positive_ratio, rolling_windows_used = rolling_ic_positive_ratio(
            normalized, forward_ret, window=rolling_window
        )
        buckets = bucket_returns(normalized, forward_ret)
        regime_perf = performance_by_regime(
            normalized, forward_ret, regime_labels
        )

        paired = _paired(normalized, forward_ret)
        if not paired.empty:
            sample_start = open_time.loc[paired.index].min()
            sample_end = open_time.loc[paired.index].max()
        else:
            sample_start = pd.NaT
            sample_end = pd.NaT

        row = {
            "symbol": symbol,
            "alpha_name": alpha_signal.name,
            "direction": alpha_signal.direction,
            "horizon_bars": horizon,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "observations": observations,
            "alpha_mean": float(normalized.mean(skipna=True)),
            "alpha_std": float(normalized.std(skipna=True)),
            "forward_return_correlation": pearson,
            "pearson_ic": pearson,
            "spearman_ic": spearman,
            "ic_t_stat": t_stat,
            "rolling_ic_positive_ratio": positive_ratio,
            "rolling_windows_used": rolling_windows_used,
            "low_bucket_return": buckets["low"],
            "mid_bucket_return": buckets["mid"],
            "high_bucket_return": buckets["high"],
            "top_minus_bottom_spread": buckets["spread"],
            "turnover_proxy": turnover,
        }

        for multiplier in (1, 2, 3):
            row[f"cost_adjusted_spread_{multiplier}x"] = cost_adjusted_spread(
                buckets["spread"],
                turnover,
                horizon,
                fee_rate,
                slippage_rate,
                cost_multiplier=multiplier
            )

        for regime, stats in regime_perf.items():
            row[f"regime_{regime}_ic"] = stats["ic"]
            row[f"regime_{regime}_observations"] = stats["observations"]

        rows.append(row)

    return rows


def evaluate_alpha_years(symbol, alpha_signal, df, horizons=(1, 3, 6), delay=1):
    """返回每个alpha、每个前瞻窗口按日历年拆分的长表。"""
    open_time = df["open_time"]
    records = []

    for horizon in horizons:
        forward_ret = forward_return(df, horizon=horizon, delay=delay)
        yearly = performance_by_year(
            alpha_signal.normalized_signal, forward_ret, open_time
        )
        yearly.insert(0, "horizon_bars", horizon)
        yearly.insert(0, "alpha_name", alpha_signal.name)
        yearly.insert(0, "symbol", symbol)
        records.append(yearly)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)
