"""真正跨资产类别的组合：把已经验证过、不做任何参数调整的
50日/200日均线交叉规则（golden_cross_oos.py）独立应用到四个
经济机制完全不同的资产类别（美股QQQ、外汇FXE、商品SLV、加密
货币ETHUSDT），固定各25%权重组合。规则、成本、样本区间见
reports/mini_medallion_multi_asset_golden_cross_oos/PREREGISTRATION.md。
"""
import pandas as pd

import golden_cross_oos as gc

SLEEVE_LEG_COSTS = {
    "QQQ": 0.0005,
    "FXE": 0.0005,
    "SLV": 0.0005,
    "ETHUSDT": 0.0015,
}


def align_sleeves(prices_by_symbol):
    """按日历日期（不是精确时间戳）内连接对齐四条价格序列。Yahoo
    的股票/外汇/商品数据时间戳带交易所开盘时间（如13:30 UTC），
    Binance的加密货币数据时间戳是00:00 UTC——两者代表同一个交易
    日，但精确时间戳不同，必须先按日历日期归一化再对齐，否则会
    因为时间戳不匹配而找不到任何重叠日期（这正是本模块早期版本
    实测踩到的真实bug，不是假设的边界情况）。样本区间由历史最短
    的ETHUSDT决定。
    """
    normalized = {}
    for symbol, series in prices_by_symbol.items():
        series = series.dropna().astype(float).sort_index()
        calendar_index = pd.DatetimeIndex(series.index).tz_convert("UTC").normalize()
        if calendar_index.has_duplicates:
            raise ValueError(f"Duplicate calendar-date price for {symbol}")
        normalized[symbol] = pd.Series(series.to_numpy(), index=calendar_index, name=symbol)

    frame = pd.concat(normalized.values(), axis=1, join="inner").dropna()
    if frame.empty:
        raise ValueError("Multi-asset sleeves have no overlapping dates")
    return frame


def run_sleeve_returns(aligned_prices, leverage):
    """对每个资产独立跑50/200日均线交叉规则(各自1x基础上乘以
    组合层面的leverage)，返回{symbol: net_return序列}和明细。
    """
    sleeve_returns = {}
    sleeve_details = {}
    for symbol in aligned_prices.columns:
        leg_cost = SLEEVE_LEG_COSTS[symbol]
        net, detail = gc.run_scenario(
            aligned_prices[symbol], leverage=leverage, leg_cost=leg_cost
        )
        sleeve_returns[symbol] = net
        sleeve_details[symbol] = detail
    return sleeve_returns, sleeve_details


def combine_equal_weight(sleeve_returns):
    """组合净收益 = 各sleeve净收益的等权(各25%)平均，不额外收费——
    每个sleeve自己的run_scenario已经扣过成本，这里只是加权求和。
    """
    combined = pd.DataFrame(sleeve_returns).sum(axis=1) * 0.25
    combined.name = "net_return"
    return combined
