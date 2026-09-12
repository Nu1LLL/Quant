"""固定50/50等权组合GLD和CWB两个已独立评估过的候选的净收益
序列，覆盖两者共同的约17.4年历史（比第39轮DBMF+GLD组合的7.3年
窗口长得多，真正包含黄金2011-2019"失落十年"在内）。不做相关性
驱动的权重优化。规则见
reports/mini_medallion_gld_cwb_combo_oos/PREREGISTRATION.md。

结构和dbmf_gold_combo_oos.py完全一致，单独成文件只是为了避免
复用那个模块里硬编码的"DBMF"/"GLD"列名（这里的两条输入是GLD和
CWB，不是DBMF和GLD），保持本项目"一个假说一个文件"的既有约定。
"""
import pandas as pd


def align_net_returns(gold_returns, cwb_returns):
    """按日期内连接对齐两条净收益序列，只保留两者都有数据的日期。"""
    gold_returns = pd.Series(gold_returns).dropna().astype(float).sort_index()
    cwb_returns = pd.Series(cwb_returns).dropna().astype(float).sort_index()
    aligned = pd.concat(
        [gold_returns.rename("GLD"), cwb_returns.rename("CWB")],
        axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("GLD and CWB have no overlapping return dates")
    return aligned


def combine_equal_weight(gold_returns, cwb_returns):
    """组合净收益 = 0.5*GLD净收益 + 0.5*CWB净收益，逐日固定权重，
    不做相关性驱动的动态调整——两条输入序列本身已经是各自实验
    扣费后的净收益，这里只是加权平均，不额外收费。
    """
    aligned = align_net_returns(gold_returns, cwb_returns)
    combined = 0.5 * aligned["GLD"] + 0.5 * aligned["CWB"]
    combined.name = "net_return"
    return combined, aligned


def pearson_correlation(gold_returns, cwb_returns):
    aligned = align_net_returns(gold_returns, cwb_returns)
    return float(aligned["GLD"].corr(aligned["CWB"]))
