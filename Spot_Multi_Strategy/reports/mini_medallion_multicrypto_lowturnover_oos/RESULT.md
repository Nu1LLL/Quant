# 六币低换手carry＋趋势组合：首次读取OOS结果

## 裁决

**失败，不进入实盘。** 数据覆盖Gate通过，但1x、2x、3x全部未通过CAGR、Sharpe、
最大回撤、PF、walk-forward和Monte Carlo联合目标。降低换手解决了上一轮的主要交易
成本问题，却暴露出新币组的180日趋势袖套本身是负alpha。不得在本历史上事后改窗口、
改更新日、删掉趋势或重新分配30/35/35权重。

## 数据覆盖

- 固定币种：VETUSDT、THETAUSDT、ALGOUSDT、ATOMUSDT、FILUSDT、UNIUSDT。
- 共同面板：6,473个8小时观测，2020-10-16 08:00 UTC至2026-09-12 16:00 UTC。
- 覆盖率100%，最大缺口8小时，无重复，跨度5.906年；预注册Gate通过。
- 样本不足十年，任何结果都不能证明“近十年稳定”。

## 成本后正式结果

| 杠杆 | 期末资金 | CAGR | Sharpe | 最大回撤 | PF | 完整年度通过 | 严格通过 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1x | $6,454.66 | -7.15% | -0.155 | -55.61% | 0.976 | 0/6 | 否 |
| 2x | $2,182.00 | -22.72% | -0.234 | -86.50% | 0.965 | 0/6 | 否 |
| 3x | $486.61 | -40.06% | -0.262 | -97.27% | 0.960 | 0/6 | 否 |

1x的六个完整年度中只有2021和2022盈利，且每一个年度折都因联合Gate失败。

## 收益来源与成本

1x累计简单毛收益为-20.61%，交易/滑点为-2.79%，净简单收益为-23.40%。低换手使成本
显著低于上一轮五币组合，但毛收益已经为负：

| 袖套 | 成本前简单收益 | 年化波动 | 方差贡献 |
|---|---:|---:|---:|
| 30% basis carry | +18.03% | 1.31% | 0.27% |
| 35%平滑资金费率carry | +23.89% | 10.86% | 16.35% |
| 35% 180日趋势 | -62.53% | 24.05% | 83.39% |

三袖套相关性仍很低：basis/funding -0.010、basis/trend +0.008、funding/trend
-0.020。但趋势袖套贡献83.4%的组合方差且期望为负，低相关不能挽救组合。

## 市场状态与稳健性

- 2020–2022：CAGR +5.17%、Sharpe 0.318、最大回撤-32.06%。
- 2023以后：CAGR -13.80%、Sharpe -0.524、最大回撤-52.64%。
- 2021：CAGR +5.32%；2022：+7.29%；2024：-10.83%；其余日期合计CAGR -7.26%。
- 5,000条21日循环区块Monte Carlo：中位CAGR -5.18%、中位Sharpe -0.135、
  5%分位最大回撤-83.45%，联合目标概率0%。

## 数据来源与可复现性

数据来自Binance官方公开现货K线、USDⓈ-M永续mark Kline与资金费率历史接口。官方说明：

- https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
- https://developers.binance.com/en/docs/derivatives-trading-usd-s-futures/Introduction
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History

冻结输入SHA-256：

- ALGO: `0c18da1c6e0d9ae2adab041cddbc32e4efbba5b2f5bf74186ccacfd79bc41cae`
- ATOM: `92d8e7076da7140ee08bda3b345103ef69f1ed74404dbdb405b53e120effe0ca`
- FIL: `a0b7e1268369573cbeea7c0d645b068aa3cd5cb51d7d315de302cf9fdadd1f6b`
- THETA: `67bbfc0304bfdc27a3089f8d11dfefcfa4b4078c7632aa651193809454ab61e3`
- UNI: `80a8778abe8c7b21ea869f609bedcb8150ab01a67300ad160d4e8bc37b7b2c3a`
- VET: `42533ffc1ec9e55ba7a096d2620c609cd76c21596c13a5af54882f95e74029b3`
- common panel: `30a8d92184f5aaaaaf16101de73293fe363adb3c31030731a51e448adf450abe`

## 限制

公开API历史仍不是逐笔可成交订单簿；5bp单边成本未覆盖极端时段冲击、强平、稳定币、
交易所、保证金层级和税务风险。基差袖套也假设现货与永续能按目标小数权重执行。上述
限制只会降低可实现性，不会把已失败的统计结果变成通过。
