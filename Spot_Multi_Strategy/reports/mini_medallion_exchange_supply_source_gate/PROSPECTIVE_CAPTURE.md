# BTC 交易所库存：前瞻采集部署记录

核查日：2026-11-08。此记录是数据来源审计，**不是策略预注册、回测结果或交易信号**。

- 本机采集器：`~/Library/Application Support/MiniMedallion/minimedallion_shadow/coinmetrics_exchange_supply_journal.py`，与仓库内 `research/crypto_blackbox/coinmetrics_exchange_supply_journal.py` 的 SHA-256 相同（`2c84d7b07ea81c4238be2c7a7b185811d3c3b931e6839d23128ba1e2c8face6e`）。
- 本机日志：`~/Library/Application Support/MiniMedallion/coinmetrics_exchange_supply.sqlite`，权限 `0600`。原始指标数值没有进入公开仓库。
- macOS LaunchAgent：`~/Library/LaunchAgents/com.nu1lll.minimedallion.coinmetrics-exchange-supply.plist`；本机每天 10:15 运行一次，登录时也运行一次。`plutil -lint` 通过；`launchctl print` 显示已注册、运行一次后退出码为 0、日历触发器有效。
- 首次独立抓取：2026-11-07 17:40:58 UTC，返回 63 个日度观测，首次见到版本 63 个；后台首次运行后共有 2 次抓取、63 个版本，最近抓取时间为 2026-11-07 17:41:25 UTC。这里仅报告记录数量和时间，不报告指标值或收益。

每个历史日期的首次见到时间都是采集时间，不能因为指标日期较早就认为该数值曾在过去可用。标准免费序列可能因后来识别交易所地址而修订；本机日志只能提供**从开始采集后**的本机观测历史，不能替代供应商的 `SplyExPitNtv` 历史点时序列，也不满足五年独立 OOS。获取有授权的点时历史前，本假说不进入收益实验；后续抓取故障或数据空缺必须按实际记录处理，不能回填后冒充点时数据。
