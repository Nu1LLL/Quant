"""把多个已通过Alpha Research Gate的alpha合成一个组合目标敞口。

三种加权方式，全部在每个时间点只使用该时间点之前已经实现的信息：

1. 等权：每个入选alpha权重相同，不依赖任何历史表现估计。
2. IC加权：用滚动/扩张窗口的"已实现"IC做权重——在t时刻用来加权的
   IC，只使用exit时间早于等于t的信号-收益配对，绝不使用t时刻还未
   实现的前瞻收益。
3. 相关性惩罚加权：在IC加权（或等权）基础上，用滚动窗口内alpha彼此
   的历史相关性对权重打折，冗余度越高权重越低，避免用矩阵求逆
   （不稳定），只用简单的"平均绝对相关性"惩罚。

组合分数combined_alpha落在大致[-1,1]，映射到现货只做多的目标敞口
[0,1]：exposure = clip(combined_alpha, 0, 1)。负的组合分数代表空仓，
不代表做空。
"""
import numpy as np
import pandas as pd

import alpha_metrics


def normalized_signal_matrix(alpha_signals):
    return pd.DataFrame({
        name: signal.normalized_signal
        for name, signal in alpha_signals.items()
    })


def equal_weights(signal_matrix):
    """每个有值的alpha等权，缺失(NaN)的alpha在当根K线权重为0。"""
    available = signal_matrix.notna()
    counts = available.sum(axis=1).replace(0, np.nan)
    weights = available.div(counts, axis=0).fillna(0.0)
    return weights


def rolling_trailing_ic(
    signal,
    df,
    horizon=3,
    delay=1,
    window=500,
    min_periods=100
):
    """在t时刻只使用"已经实现"的信号-前瞻收益配对估计滚动IC。

    forward_return(horizon, delay)在第i行的值依赖到i+delay+horizon为止
    的价格，所以要在t时刻合法使用它，必须要求i+delay+horizon<=t，
    也就是把信号和收益都整体后移delay+horizon根K线，再做滚动相关。
    这样rolling_trailing_ic在第t行的值只用了截至第t行已经确定的信息，
    可以安全地当作t时刻的组合权重。
    """
    lag = delay + horizon
    forward_ret = alpha_metrics.forward_return(df, horizon=horizon, delay=delay)

    lagged_signal = signal.shift(lag)
    realized_return = forward_ret.shift(lag)

    return lagged_signal.rolling(window, min_periods=min_periods).corr(
        realized_return
    )


def ic_weights(alpha_signals, df, horizon=3, delay=1, window=500, min_periods=100):
    signal_matrix = normalized_signal_matrix(alpha_signals)

    trailing_ic = pd.DataFrame({
        name: rolling_trailing_ic(
            signal.normalized_signal, df,
            horizon=horizon, delay=delay,
            window=window, min_periods=min_periods
        )
        for name, signal in alpha_signals.items()
    })

    # 只有正的已实现IC才参与加权；如果某一行全部alpha的滚动IC都不为正
    # （或还没有足够样本），退化为等权，避免出现空仓组合。
    positive_ic = trailing_ic.clip(lower=0.0)
    available = signal_matrix.notna()
    positive_ic = positive_ic.where(available, 0.0)

    row_sum = positive_ic.sum(axis=1)
    fallback_rows = row_sum <= 0

    weights = positive_ic.div(row_sum.replace(0, np.nan), axis=0)
    fallback_weights = equal_weights(signal_matrix)
    weights = weights.where(~fallback_rows, fallback_weights)

    return weights.fillna(0.0)


def rolling_average_abs_correlation(signal_matrix, window=500, min_periods=100):
    """每个alpha相对其余alpha的滚动平均绝对相关性，只用过去窗口数据。"""
    columns = signal_matrix.columns
    avg_corr = pd.DataFrame(index=signal_matrix.index, columns=columns, dtype=float)

    for target in columns:
        others = [c for c in columns if c != target]
        if not others:
            avg_corr[target] = 0.0
            continue

        pairwise = pd.concat(
            [
                signal_matrix[target].rolling(
                    window, min_periods=min_periods
                ).corr(signal_matrix[other]).abs()
                for other in others
            ],
            axis=1
        )
        avg_corr[target] = pairwise.mean(axis=1)

    return avg_corr


def correlation_penalized_weights(
    base_weights,
    signal_matrix,
    window=500,
    min_periods=100
):
    avg_corr = rolling_average_abs_correlation(
        signal_matrix, window=window, min_periods=min_periods
    ).fillna(0.0)

    penalty = 1.0 / (1.0 + avg_corr)
    penalized = base_weights * penalty

    row_sum = penalized.sum(axis=1)
    normalized = penalized.div(row_sum.replace(0, np.nan), axis=0)

    return normalized.where(row_sum > 0, base_weights).fillna(0.0)


def combined_alpha_score(alpha_signals, weights):
    signal_matrix = normalized_signal_matrix(alpha_signals)
    aligned_weights = weights.reindex(columns=signal_matrix.columns).fillna(0.0)
    return (signal_matrix.fillna(0.0) * aligned_weights).sum(axis=1)


def target_exposure_from_combined_alpha(combined_alpha):
    """把[-1,1]的组合分数映射到现货只做多的目标敞口[0,1]。

    负的组合分数代表空仓，不代表做空；这是一个透明的V1线性映射，
    不是针对历史数据优化出来的曲线。
    """
    return combined_alpha.clip(lower=0.0, upper=1.0)


def build_ensemble(
    alpha_signals,
    df,
    method="equal",
    horizon=3,
    delay=1,
    window=500,
    min_periods=100
):
    """返回(weights, combined_alpha, target_exposure)。

    method: "equal" | "ic" | "corr_penalized"
    """
    signal_matrix = normalized_signal_matrix(alpha_signals)

    if method == "equal":
        weights = equal_weights(signal_matrix)
    elif method == "ic":
        weights = ic_weights(
            alpha_signals, df, horizon=horizon, delay=delay,
            window=window, min_periods=min_periods
        )
    elif method == "corr_penalized":
        base_weights = ic_weights(
            alpha_signals, df, horizon=horizon, delay=delay,
            window=window, min_periods=min_periods
        )
        weights = correlation_penalized_weights(
            base_weights, signal_matrix, window=window, min_periods=min_periods
        )
    else:
        raise ValueError(f"未知的集成方式：{method}")

    combined_alpha = combined_alpha_score(alpha_signals, weights)
    target_exposure = target_exposure_from_combined_alpha(combined_alpha)

    return weights, combined_alpha, target_exposure
