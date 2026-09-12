"""固定50/50等权组合DBMF和GLD两个已独立评估过的候选的净收益
序列。不做相关性驱动的权重优化——组合层面只做加权平均，两个
组成部分各自的手续费/滑点/融资成本已经在各自的实验里扣过，
这里不重复计费。规则见
reports/mini_medallion_dbmf_gold_combo_oos/PREREGISTRATION.md。
"""
import pandas as pd


def align_net_returns(dbmf_returns, gold_returns):
    """按日期内连接对齐两条净收益序列，只保留两者都有数据的日期。"""
    dbmf_returns = pd.Series(dbmf_returns).dropna().astype(float).sort_index()
    gold_returns = pd.Series(gold_returns).dropna().astype(float).sort_index()
    aligned = pd.concat(
        [dbmf_returns.rename("DBMF"), gold_returns.rename("GLD")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("DBMF and GLD have no overlapping return dates")
    return aligned


def combine_equal_weight(dbmf_returns, gold_returns):
    """组合净收益 = 0.5*DBMF净收益 + 0.5*GLD净收益，逐日固定权重，
    不做相关性驱动的动态调整——两条输入序列本身已经是各自实验
    扣费后的净收益，这里只是加权平均，不额外收费。
    """
    aligned = align_net_returns(dbmf_returns, gold_returns)
    combined = 0.5 * aligned["DBMF"] + 0.5 * aligned["GLD"]
    combined.name = "net_return"
    return combined, aligned


def pearson_correlation(dbmf_returns, gold_returns):
    aligned = align_net_returns(dbmf_returns, gold_returns)
    return float(aligned["DBMF"].corr(aligned["GLD"]))
