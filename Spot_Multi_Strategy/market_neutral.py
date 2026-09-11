"""市场中性、无净杠杆的横截面多空组合层——完全独立于长仓专用的
risk_overlay.py/ensemble.py/walk_forward.py，不改动那几个模块的
任何既有行为，也不接入任何真实账户或资金。

核心思路：
1. 横截面去均值：每个时间点把各资产的alpha分数减去当时所有可用
   资产的均值，多空双方名义金额天然接近相等（去均值后各资产权重
   之和约等于0），而不是先做多头再单独叠加空头。
2. 无净杠杆：去均值后的权重按总绝对值(gross)缩放，超过gross_cap
   （默认1.0，即100%本金）就按比例缩小，从不放大——信号弱的时候
   总敞口天然更小，不会为了"用满额度"而人为加仓。
3. 做空腔通过Binance永续合约实现（现货本身不能卖空），所以除了
   fee/slippage还有资金费率成本/收益：多头在资金费率为正时付费，
   空头在资金费率为正时收费。用futures_data.py已经在抓的真实历史
   资金费率，因果对齐后近似把8小时一次的结算平摊到两根4小时K线上
   （fund_per_bar = funding_rate / 2），这是一个记录在案的简化，
   不是精确的结算时点建模。
4. 不同资产历史长度不一致时（比如SOL上市晚于BTC/ETH/BNB）用
   open_time做outer join对齐，缺失的资产在那个时间点自然被排除出
   横截面均值，不会被静默填0或拖累其他资产。
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

import alpha_metrics
import alphas.alternative_data as alternative_data_alphas
from optimizer import build_validation_folds

DEFAULT_FOLD_COUNT = 6
DEFAULT_WARMUP_BARS = 500


def align_by_open_time(frames, series_by_symbol):
    """把{symbol: Series}对齐到共同的open_time网格（outer join）。"""
    aligned = {}
    for symbol, series in series_by_symbol.items():
        open_time = pd.to_datetime(
            frames[symbol]["open_time"], utc=True
        ).reset_index(drop=True)
        aligned[symbol] = pd.Series(
            series.reset_index(drop=True).values, index=open_time
        )
    return pd.DataFrame(aligned).sort_index()


def cross_sectional_demean(master_frame):
    """每个时间点减去当时所有有值资产的横截面均值（幅度加权版本）。"""
    row_mean = master_frame.mean(axis=1, skipna=True)
    return master_frame.sub(row_mean, axis=0)


def cross_sectional_rank_demean(master_frame):
    """Fama-French式的排名版本：每个时间点把各资产的分数换成横截面
    百分位排名，再减去**当时那一行排名的实际均值**、乘以2映射到大致
    [-1,1]。

    注意：不能直接减去0.5——pandas的rank(pct=True)公式是rank/count，
    N个资产且没有并列时排名是{1/N,...,N/N}，均值是(N+1)/(2N)，
    只有N趋于无穷大时才趋近0.5。用实际行均值去中心化，才能保证
    每一行去中心化后的排名和精确为0（多空名义金额相等），不受N
    大小影响。

    和幅度加权的cross_sectional_demean相比，排名对"今天分数比昨天
    大了多少"这种连续幅度噪声不敏感——只要相对顺序没变，排名就不变，
    这是学术因子研究（Fama-French排序法）的标准做法，用来降低
    逐bar调仓的换手率，不是针对这份数据拟合出来的技巧。
    """
    rank = master_frame.rank(axis=1, pct=True, na_option="keep")
    row_mean = rank.mean(axis=1, skipna=True)
    return rank.sub(row_mean, axis=0) * 2.0


def build_cross_sectional_signal(
    frames, alpha_sets, alpha_name, method="demean"
):
    """返回一个日历时间对齐、横截面中性化后的信号矩阵（列=品种）。

    method="demean"：减去横截面均值（幅度加权）。
    method="rank"：换成横截面百分位排名（Fama-French式排序法）。
    """
    scores = {
        symbol: alpha_sets[symbol][alpha_name].normalized_signal
        for symbol in frames
        if alpha_name in alpha_sets.get(symbol, {})
    }
    if len(scores) < 2:
        raise ValueError(f"{alpha_name}至少需要2个品种才能做横截面对比")

    master = align_by_open_time(frames, scores)

    if method == "demean":
        return cross_sectional_demean(master)
    if method == "rank":
        return cross_sectional_rank_demean(master)
    raise ValueError(f"未知的横截面中性化方式：{method}")


def resample_to_rebalance_schedule(master_frame, rebalance_every_bars):
    """把目标只在每隔rebalance_every_bars根K线的时间点更新一次，
    中间沿用上一次的目标（前向填充）。

    动量/carry类因子在文献和实盘里几乎从不逐根K线换仓——AQR的TSMOM
    论文、经典的12-1动量构造都是月度或周度再平衡，这里默认给出接口，
    不是为了让某个alpha通过验收才加的：4小时K线逐根跟踪一个本来是
    周/月频的信号，产生的换手本身就不符合这类因子的正常使用方式。
    rebalance_every_bars=1时完全不做任何改变（逐根K线跟踪，向后兼容
    默认行为）。
    """
    if rebalance_every_bars <= 1:
        return master_frame

    schedule_mask = np.arange(len(master_frame)) % rebalance_every_bars == 0
    resampled = master_frame.copy()
    resampled.loc[~schedule_mask] = np.nan
    return resampled.ffill()


def build_cross_sectional_forward_return(frames, delay=1, horizon=1):
    """每个资产相对"当时横截面平均前瞻收益"的超额前瞻收益——只用于
    研究/Gate评估的标签，从不进入信号计算。
    """
    returns = {
        symbol: alpha_metrics.forward_return(df, horizon=horizon, delay=delay)
        for symbol, df in frames.items()
    }
    master = align_by_open_time(frames, returns)
    return cross_sectional_demean(master)


def scale_to_gross_cap(demeaned_scores, gross_cap=1.0):
    """去均值后的敞口按总绝对值缩放到不超过gross_cap，只缩小不放大。"""
    clipped = demeaned_scores.clip(-1.0, 1.0)
    gross = clipped.abs().sum(axis=1, skipna=True)
    scale = (gross_cap / gross.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    return clipped.mul(scale, axis=0)


def apply_no_trade_band(target_matrix, no_trade_band=0.05):
    """逐资产独立应用"目标变化幅度不够就不调仓"，返回(实际持仓矩阵,
    换手矩阵)。每个资产的持仓路径只依赖自己的历史目标序列，资产
    之间互相独立，可以按列分别跑（不需要risk_overlay.py那种因为
    回撤联动而必须全组合一起走的路径依赖循环）。gate诊断和真实
    组合回测共用这个函数，保证换手统计口径一致。
    """
    positions = pd.DataFrame(
        index=target_matrix.index, columns=target_matrix.columns, dtype=float
    )
    turnovers = pd.DataFrame(
        index=target_matrix.index, columns=target_matrix.columns, dtype=float
    )

    for symbol in target_matrix.columns:
        target = target_matrix[symbol].fillna(0.0).to_numpy()
        n = len(target)
        position = np.zeros(n)
        turnover = np.zeros(n)
        current = 0.0

        for t in range(n):
            if abs(target[t] - current) >= no_trade_band:
                turnover[t] = abs(target[t] - current)
                current = target[t]
            position[t] = current

        positions[symbol] = position
        turnovers[symbol] = turnover

    return positions, turnovers


def align_funding_rates(frames, funding_frames):
    """{symbol: 因果对齐到该品种自己K线网格的资金费率Series}，
    没有资金费率数据的品种用0填充（相当于假设该品种暂不参与做空腔的
    资金费率结算，而不是编造一个费率）。
    """
    aligned = {}
    for symbol, df in frames.items():
        if symbol in funding_frames:
            aligned[symbol] = alternative_data_alphas.align_funding_rate_to_bars(
                df, funding_frames[symbol]
            ).fillna(0.0)
        else:
            aligned[symbol] = pd.Series(0.0, index=range(len(df)))
    return aligned


def simulate_market_neutral_portfolio(
    frames,
    funding_frames,
    target_weights,
    delay=1,
    fee_rate=0.001,
    slippage_rate=0.0005,
    no_trade_band=0.05
):
    """target_weights是日历时间对齐的DataFrame（列=品种，值已经过
    scale_to_gross_cap处理），返回逐根K线的组合净收益和各资产明细。

    换手不交易带（no_trade_band）：横截面信号几乎每根K线都会有细微
    变化，如果照单全收会产生大量无意义的换手——这不是理论问题，
    是实测发现的真实问题：不加不交易带时，总成本占毛收益绝对值的
    比例看起来不高(cost_drag_ratio指标本身没有报警)，但因为信号的
    毛收益本身很薄、噪音很大，逐bar换手成本的绝对值足以吃掉全部
    净信号——所以这里和risk_overlay.py对长仓的处理一样，加上"目标
    变化幅度不够就不调仓"的逻辑，只是这里是逐资产独立判断（不需要
    像长仓那样引入回撤相关的路径依赖状态）。
    """
    funding_aligned = align_funding_rates(frames, funding_frames)
    symbols = [s for s in target_weights.columns if s in frames]

    # 先在共同的日历时间主索引上统一跑一次不交易带（按时间先后顺序，
    # 和gate用的是同一个函数、同一套逻辑），再把结果分别对齐回每个
    # 资产自己的K线网格——这样两条路径（gate诊断、完整组合回测）
    # 的换手统计口径完全一致。
    positions_master, turnovers_master = apply_no_trade_band(
        target_weights[symbols], no_trade_band=no_trade_band
    )

    per_asset_frames = {}
    for symbol in symbols:
        df = frames[symbol]
        open_time = pd.to_datetime(df["open_time"], utc=True)
        weight = positions_master[symbol].reindex(open_time).fillna(
            0.0
        ).reset_index(drop=True)
        turnover = turnovers_master[symbol].reindex(open_time).fillna(
            0.0
        ).reset_index(drop=True)

        bar_return = alpha_metrics.forward_return(
            df, horizon=1, delay=delay
        ).to_numpy()
        funding = funding_aligned[symbol].to_numpy()

        gross_pnl = weight * bar_return
        # 多头在funding为正时付费(-)，空头在funding为正时收费(+)，
        # 用weight的符号统一表达：funding_pnl = -weight * 每根K线funding
        funding_pnl = -weight * (funding / 2.0)
        cost = turnover * (fee_rate + slippage_rate) * 2.0
        net_pnl = gross_pnl + funding_pnl - cost

        per_asset_frames[symbol] = pd.DataFrame({
            "open_time": open_time.reset_index(drop=True),
            "weight": weight,
            "bar_return": bar_return,
            "funding_pnl": funding_pnl,
            "turnover": turnover,
            "cost": cost,
            "net_pnl": net_pnl
        })

    return per_asset_frames


def combine_portfolio_net_pnl(per_asset_frames, initial_capital, symbols):
    """汇总成一条组合权益曲线。

    注意：per_asset_frames[symbol]["weight"]来自scale_to_gross_cap，
    已经是"占组合总资金的比例"（sum(|weight_i|) <= gross_cap，比如
    1.0代表100%总资金），不是"占该资产独立子账户资金的比例"。所以
    每个资产的net_pnl本身就已经是"对组合总资金的收益贡献"，组合层面
    直接把各资产的net_pnl相加即可，不需要再乘以1/N的资金分配系数——
    那样会重复稀释，错误地把总敞口砍到1/N。
    """
    dollar_pnl = None

    for symbol in symbols:
        if symbol not in per_asset_frames:
            continue
        frame = per_asset_frames[symbol].set_index("open_time")
        contribution = frame["net_pnl"] * initial_capital
        dollar_pnl = (
            contribution if dollar_pnl is None
            else dollar_pnl.add(contribution, fill_value=0.0)
        )

    net_pnl = (dollar_pnl / initial_capital).sort_index()
    portfolio_df = pd.DataFrame({
        "open_time": net_pnl.index,
        "net_pnl": net_pnl.values
    }).reset_index(drop=True)

    gross_exposure = None
    for symbol in symbols:
        if symbol not in per_asset_frames:
            continue
        frame = per_asset_frames[symbol].set_index("open_time")
        abs_weight = frame["weight"].abs()
        gross_exposure = (
            abs_weight if gross_exposure is None
            else gross_exposure.add(abs_weight, fill_value=0.0)
        )
    portfolio_df["gross_exposure"] = gross_exposure.reindex(
        net_pnl.index
    ).fillna(0.0).values

    turnover_total = None
    cost_total = None
    for symbol in symbols:
        if symbol not in per_asset_frames:
            continue
        frame = per_asset_frames[symbol].set_index("open_time")
        turnover_total = (
            frame["turnover"] if turnover_total is None
            else turnover_total.add(frame["turnover"], fill_value=0.0)
        )
        cost_total = (
            frame["cost"] if cost_total is None
            else cost_total.add(frame["cost"], fill_value=0.0)
        )
    portfolio_df["turnover"] = turnover_total.reindex(
        net_pnl.index
    ).fillna(0.0).values
    portfolio_df["cost"] = cost_total.reindex(
        net_pnl.index
    ).fillna(0.0).values
    portfolio_df["position"] = portfolio_df["gross_exposure"] / 2.0
    portfolio_df["rebalanced"] = portfolio_df["turnover"] > 1e-9

    return portfolio_df


def _annualization_factor(open_time):
    if len(open_time) < 2:
        return 0.0
    bar_seconds = pd.Series(open_time).diff().dropna().median().total_seconds()
    if bar_seconds <= 0:
        return 0.0
    return 365.25 * 86400 / bar_seconds


def _max_drawdown_from_returns(returns):
    valid = returns.dropna()
    if valid.empty:
        return 0.0
    equity = (1 + valid).cumprod()
    return float((equity / equity.cummax() - 1).min())


@dataclass
class CrossSectionalGateResult:
    alpha_name: str
    passed: bool
    checks: Dict[str, bool]
    fold_table: pd.DataFrame
    summary: Dict[str, float] = field(default_factory=dict)


def evaluate_cross_sectional_alpha(
    alpha_name,
    frames,
    alpha_sets,
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
    min_positive_fold_ratio=0.60,
    no_trade_band=0.05,
    method="demean",
    rebalance_every_bars=1
):
    """把一个alpha家族横截面中性化后，当成市场中性信号跑6折
    walk-forward验收——criteria和walk_forward.py里长仓Gate完全一样，
    只是"敞口"和"收益"的定义换成了横截面中性化/超额版本，不做gate
    层面的资金费率建模（和长仓Gate一样，Gate是轻量诊断，完整成本
    放在portfolio层面的回测里）。

    method="demean"（幅度加权）或"rank"（Fama-French式百分位排名，
    见build_cross_sectional_signal的说明）。

    rebalance_every_bars：见resample_to_rebalance_schedule的说明——
    动量/carry类因子在文献里几乎从不逐根K线换仓，默认值1是逐根K线
    跟踪（向后兼容），设成比如42（4小时K线的周频）更接近这类因子
    实际的使用方式。

    敞口先用scale_to_gross_cap封顶到1.0（100%总资金，不能直接用
    未封顶的clip(-1,1)——那样4个资产各自最多到1.0，总gross敞口可能
    到4倍，不是"无净杠杆"的设定），再按rebalance_every_bars降频，
    最后用apply_no_trade_band过滤掉剩余的幅度不够的调仓——这是实测
    发现的必要步骤：没有不交易带时，换手成本的绝对值足以吃掉这类
    信号本来就很薄的毛收益，即使cost_drag_ratio这个相对指标看起来
    不高。
    """
    signal_matrix = build_cross_sectional_signal(
        frames, alpha_sets, alpha_name, method=method
    )
    forward_return_matrix = build_cross_sectional_forward_return(
        frames, delay=delay, horizon=1
    )

    symbols = [s for s in signal_matrix.columns if s in forward_return_matrix.columns]
    signal_matrix = signal_matrix[symbols]
    forward_return_matrix = forward_return_matrix[symbols]

    capped = scale_to_gross_cap(signal_matrix, gross_cap=1.0)
    scheduled = resample_to_rebalance_schedule(capped, rebalance_every_bars)
    exposure, turnover = apply_no_trade_band(scheduled, no_trade_band=no_trade_band)

    gross_pnl = exposure * forward_return_matrix
    cost = turnover * (fee_rate + slippage_rate) * 2.0
    # weight已经是"占组合总资金的比例"，组合净收益是各资产贡献之和，
    # 不是平均——见combine_portfolio_net_pnl里同样的说明。
    net_pnl = (gross_pnl - cost).sum(axis=1, skipna=True)
    net_pnl_by_asset_count = (gross_pnl - cost).notna().sum(axis=1)
    net_pnl = net_pnl.where(net_pnl_by_asset_count > 0)

    open_time = signal_matrix.index
    annualization_factor = _annualization_factor(open_time)
    n = len(signal_matrix)

    folds = build_validation_folds(
        data_length=n, development_end=n,
        fold_count=fold_count, warmup_bars=warmup_bars
    )

    fold_records = []
    for fold in folds:
        fold_slice = slice(fold["start"] - 1, fold["end"])
        fold_signal = signal_matrix.iloc[fold_slice].stack()
        fold_forward = forward_return_matrix.iloc[fold_slice].stack()
        paired = pd.concat(
            [fold_signal.rename("signal"), fold_forward.rename("forward")],
            axis=1
        ).dropna()

        if len(paired) >= 3:
            fold_ic = paired["signal"].corr(paired["forward"])
        else:
            fold_ic = np.nan

        fold_net_pnl = net_pnl.iloc[fold_slice].dropna()
        observations = int(len(fold_net_pnl))

        if observations > 1 and fold_net_pnl.std(ddof=0) > 0:
            fold_sharpe = float(
                fold_net_pnl.mean() / fold_net_pnl.std(ddof=0)
                * np.sqrt(annualization_factor)
            )
        else:
            fold_sharpe = 0.0

        fold_pnl_sum = float(fold_net_pnl.sum()) if observations else 0.0
        fold_max_drawdown = _max_drawdown_from_returns(fold_net_pnl)

        fold_records.append({
            "alpha_name": alpha_name,
            "fold": fold["fold"],
            "observations": observations,
            "pair_observations": int(len(paired)),
            "fold_ic": fold_ic,
            "fold_sharpe": fold_sharpe,
            "fold_pnl_sum": fold_pnl_sum,
            "fold_max_drawdown": fold_max_drawdown
        })

    fold_table = pd.DataFrame(fold_records)

    yearly = pd.DataFrame({
        "open_time": open_time,
        "net_pnl": net_pnl.values
    }).dropna()
    yearly["year"] = pd.to_datetime(yearly["open_time"]).dt.year
    yearly_pnl = yearly.groupby("year")["net_pnl"].sum().reset_index()

    total_pnl = float(net_pnl.sum(skipna=True))
    total_observations = int(net_pnl.notna().sum())
    # 用有符号的毛收益（不是绝对值之和）做成本侵蚀比例的分母：横截面
    # 去均值信号逐bar在0附近噪声很大，绝对值之和会被这些互相抵消的
    # 噪声大幅撑大，导致"成本/绝对值毛收益"这个比例看起来很低，但
    # 实际成本可能已经超过了真正可提取的（有符号）edge本身。
    total_gross_pnl_signed = float(gross_pnl.sum(skipna=True).sum())
    total_cost = float(cost.sum(skipna=True).sum())

    median_fold_ic = float(fold_table["fold_ic"].median(skipna=True))
    positive_fold_ratio = float(
        (fold_table["fold_ic"] > 0).mean()
    ) if not fold_table.empty else 0.0

    positive_fold_pnl_sum = fold_table.loc[
        fold_table["fold_pnl_sum"] > 0, "fold_pnl_sum"
    ].sum() if not fold_table.empty else 0.0
    max_fold_pnl_ratio = (
        float(fold_table["fold_pnl_sum"].max() / positive_fold_pnl_sum)
        if positive_fold_pnl_sum > 0 else float("inf")
    )

    positive_year_pnl_sum = yearly_pnl.loc[
        yearly_pnl["net_pnl"] > 0, "net_pnl"
    ].sum() if not yearly_pnl.empty else 0.0
    max_year_pnl_ratio = (
        float(yearly_pnl["net_pnl"].max() / positive_year_pnl_sum)
        if positive_year_pnl_sum > 0 else float("inf")
    )

    cost_drag_ratio = (
        total_cost / total_gross_pnl_signed
        if total_gross_pnl_signed > 0 else float("inf")
    )
    min_fold_observations_actual = (
        int(fold_table["observations"].min()) if not fold_table.empty else 0
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

    return CrossSectionalGateResult(
        alpha_name=alpha_name,
        passed=all(checks.values()),
        checks=checks,
        fold_table=fold_table,
        summary=summary
    )
