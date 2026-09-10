"""Alpha信号的滚动折(walk-forward)验证与Alpha Research Gate。

不是重新划出一段"新的隐藏测试区间"——README已经说明，2020年至今的
全部历史现在都算研究/开发数据。这里做的是把开发数据切成连续的折，
用于判断一个alpha的预测能力是否稳定，而不是只看一次全样本IC。

每个alpha在这里用一个简化的、单资产、仅做多的"标准化exposure"
翻译成逐根K线的模拟净收益（exposure=clip(normalized_signal,0,1)），
只用于alpha层面的诊断（fold IC/Sharpe/回撤/换手），
不是最终组合回测——最终组合回测在portfolio_backtest.py里，
使用ensemble.py产生的多alpha合成敞口和risk_overlay.py的风控层。
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

import alpha_metrics
from optimizer import build_validation_folds

DEFAULT_FOLD_COUNT = 6
DEFAULT_WARMUP_BARS = 500


def simulate_standalone_alpha(
    df,
    alpha_signal,
    delay=1,
    fee_rate=0.001,
    slippage_rate=0.0005
):
    """把一个alpha独立翻译成[0,1]敞口，产生逐根K线净收益序列。

    这是alpha层面的诊断工具，不是生产执行路径：真实组合敞口由
    ensemble.py合成多个alpha后，再经过risk_overlay.py风控。
    """
    bar_return = alpha_metrics.forward_return(df, horizon=1, delay=delay)
    exposure = alpha_signal.normalized_signal.clip(lower=0.0, upper=1.0)

    turnover = exposure.diff().abs()
    turnover.iloc[0] = exposure.iloc[0] if len(exposure) else np.nan

    gross_pnl = exposure * bar_return
    cost = turnover * (fee_rate + slippage_rate) * 2.0
    net_pnl = gross_pnl - cost

    return pd.DataFrame({
        "open_time": pd.to_datetime(df["open_time"], utc=True),
        "exposure": exposure,
        "turnover": turnover,
        "bar_return": bar_return,
        "net_pnl": net_pnl
    })


def _max_drawdown_from_returns(returns):
    valid = returns.dropna()
    if valid.empty:
        return 0.0
    equity = (1 + valid).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def _annualization_factor(open_time):
    if len(open_time) < 2:
        return 0.0
    bar_seconds = open_time.diff().dropna().median().total_seconds()
    if bar_seconds <= 0:
        return 0.0
    return 365.25 * 86400 / bar_seconds


@dataclass
class AlphaGateResult:
    symbol: str
    alpha_name: str
    passed: bool
    checks: Dict[str, bool]
    fold_table: pd.DataFrame
    yearly_pnl: pd.DataFrame
    summary: Dict[str, float] = field(default_factory=dict)


def evaluate_alpha_walk_forward(
    symbol,
    alpha_signal,
    df,
    fold_count=DEFAULT_FOLD_COUNT,
    warmup_bars=DEFAULT_WARMUP_BARS,
    delay=1,
    fee_rate=0.001,
    slippage_rate=0.0005,
    min_total_observations=1000,
    min_fold_observations=30,
    max_fold_pnl_share=0.50,
    max_year_pnl_share=0.60,
    max_cost_drag_ratio=0.70,
    min_positive_fold_ratio=0.60
):
    simulation = simulate_standalone_alpha(
        df, alpha_signal, delay=delay,
        fee_rate=fee_rate, slippage_rate=slippage_rate
    )
    annualization_factor = _annualization_factor(simulation["open_time"])

    folds = build_validation_folds(
        data_length=len(df),
        development_end=len(df),
        fold_count=fold_count,
        warmup_bars=warmup_bars
    )

    fold_records = []
    for fold in folds:
        fold_slice = simulation.iloc[fold["start"] - 1:fold["end"]]
        forward_ret = alpha_metrics.forward_return(
            df.iloc[fold["start"] - 1:fold["end"]].reset_index(drop=True),
            horizon=1,
            delay=delay
        )
        fold_signal = alpha_signal.normalized_signal.iloc[
            fold["start"] - 1:fold["end"]
        ].reset_index(drop=True)

        fold_ic, fold_ic_obs = alpha_metrics.pearson_ic(
            fold_signal, forward_ret
        )

        net_pnl = fold_slice["net_pnl"].dropna()
        observations = int(len(net_pnl))

        if observations > 1 and net_pnl.std(ddof=0) > 0:
            fold_sharpe = float(
                net_pnl.mean() / net_pnl.std(ddof=0)
                * np.sqrt(annualization_factor)
            )
        else:
            fold_sharpe = 0.0

        fold_return = float((1 + net_pnl).prod() - 1) if observations else 0.0
        fold_pnl_sum = float(net_pnl.sum()) if observations else 0.0
        fold_turnover = float(
            fold_slice["turnover"].dropna().mean()
        ) if observations else 0.0
        fold_max_drawdown = _max_drawdown_from_returns(net_pnl)

        fold_records.append({
            "symbol": symbol,
            "alpha_name": alpha_signal.name,
            "fold": fold["fold"],
            "start_time": (
                fold_slice["open_time"].iloc[0]
                if not fold_slice.empty else pd.NaT
            ),
            "end_time": (
                fold_slice["open_time"].iloc[-1]
                if not fold_slice.empty else pd.NaT
            ),
            "observations": observations,
            "fold_ic": fold_ic,
            "fold_ic_observations": fold_ic_obs,
            "fold_sharpe": fold_sharpe,
            "fold_return": fold_return,
            "fold_pnl_sum": fold_pnl_sum,
            "fold_turnover": fold_turnover,
            "fold_max_drawdown": fold_max_drawdown
        })

    fold_table = pd.DataFrame(fold_records)

    yearly = simulation.dropna(subset=["net_pnl"]).copy()
    yearly["year"] = yearly["open_time"].dt.year
    yearly_pnl = (
        yearly.groupby("year")["net_pnl"].sum().reset_index()
        if not yearly.empty
        else pd.DataFrame(columns=["year", "net_pnl"])
    )

    total_pnl = float(simulation["net_pnl"].sum(skipna=True))
    total_observations = int(simulation["net_pnl"].notna().sum())
    total_gross_pnl_abs = float(
        (simulation["exposure"] * simulation["bar_return"]).abs().sum(
            skipna=True
        )
    )
    total_cost = float(
        (
            simulation["turnover"]
            * (fee_rate + slippage_rate) * 2.0
        ).sum(skipna=True)
    )

    median_fold_ic = float(fold_table["fold_ic"].median(skipna=True))
    positive_fold_ratio = float(
        (fold_table["fold_ic"] > 0).mean()
    ) if not fold_table.empty else 0.0

    # 用"正贡献折/正贡献年份之和"做分母，而不是净PnL总和：
    # 当亏损的折/年份把净总和抵消到接近0时，用净总和做分母会让集中度
    # 比例在数学上被人为放大甚至失去意义。分母改成正贡献之和后，
    # 这个比例才是标准意义上的"收益集中度"（最好的一折/一年占了
    # 全部正贡献的多大比例），不受亏损年份大小的干扰。
    positive_fold_pnl_sum = (
        fold_table.loc[fold_table["fold_pnl_sum"] > 0, "fold_pnl_sum"].sum()
        if not fold_table.empty
        else 0.0
    )
    if positive_fold_pnl_sum > 0:
        max_fold_pnl_ratio = float(
            fold_table["fold_pnl_sum"].max() / positive_fold_pnl_sum
        )
    else:
        max_fold_pnl_ratio = float("inf")

    positive_year_pnl_sum = (
        yearly_pnl.loc[yearly_pnl["net_pnl"] > 0, "net_pnl"].sum()
        if not yearly_pnl.empty
        else 0.0
    )
    if positive_year_pnl_sum > 0:
        max_year_pnl_ratio = float(
            yearly_pnl["net_pnl"].max() / positive_year_pnl_sum
        )
    else:
        max_year_pnl_ratio = float("inf")

    cost_drag_ratio = (
        total_cost / total_gross_pnl_abs
        if total_gross_pnl_abs > 0
        else float("inf")
    )

    min_fold_observations_actual = (
        int(fold_table["observations"].min())
        if not fold_table.empty
        else 0
    )

    checks = {
        "折IC中位数为正": median_fold_ic > 0,
        f"至少{min_positive_fold_ratio:.0%}的折IC为正": (
            positive_fold_ratio >= min_positive_fold_ratio
        ),
        f"单折PnL占比不超过{max_fold_pnl_share:.0%}": (
            max_fold_pnl_ratio <= max_fold_pnl_share
        ),
        f"单一日历年PnL占比不超过{max_year_pnl_share:.0%}": (
            max_year_pnl_ratio <= max_year_pnl_share
        ),
        f"扣费后成本侵蚀比例不超过{max_cost_drag_ratio:.0%}": (
            cost_drag_ratio <= max_cost_drag_ratio
        ),
        f"总有效样本不少于{min_total_observations}根K线": (
            total_observations >= min_total_observations
        ),
        f"每折样本不少于{min_fold_observations}根K线": (
            min_fold_observations_actual >= min_fold_observations
        )
    }

    summary = {
        "median_fold_ic": median_fold_ic,
        "positive_fold_ratio": positive_fold_ratio,
        "max_fold_pnl_ratio": max_fold_pnl_ratio,
        "max_year_pnl_ratio": max_year_pnl_ratio,
        "cost_drag_ratio": cost_drag_ratio,
        "total_pnl": total_pnl,
        "total_observations": total_observations,
        "min_fold_observations": min_fold_observations_actual
    }

    return AlphaGateResult(
        symbol=symbol,
        alpha_name=alpha_signal.name,
        passed=all(checks.values()),
        checks=checks,
        fold_table=fold_table,
        yearly_pnl=yearly_pnl,
        summary=summary
    )


def evaluate_alpha_set_walk_forward(symbol, alpha_signals, df, **kwargs):
    results = {}
    for name, alpha_signal in alpha_signals.items():
        results[name] = evaluate_alpha_walk_forward(
            symbol, alpha_signal, df, **kwargs
        )
    return results
