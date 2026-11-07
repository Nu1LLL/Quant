# BTC 交易所库存收缩假说：五年来源门槛未通过

核查日：2026-11-08。**没有预注册收益实验，没有读取本假说的历史信号、价格、盈亏，也没有策略结论。** 候选机制是 BTC 存于交易所的原生币数量下降可能缩小即期可售库存，使随后币圈价格上涨；反向也可能只是资金流出或托管迁移。其经济来源与已测的期货未平仓量、资金费、链上手续费、活跃地址及 MVRV 不同，统计独立性尚未证明。

[Coin Metrics 官方定义](https://docs.coinmetrics.io/network-data/network-data-overview/exchange/exchange-supply)说明 `SplyExNtv` 汇总被识别为交易所控制的地址余额，并明确有 `SplyExPitNtv` 对应指标。[官方点时方法](https://docs.coinmetrics.io/network-data/network-data-overview/exchange/point-in-time-flows)明确：标准指标会把后来才识别的地址回填到过去；`Pit` 指标只用当时已识别的地址，正是历史回测避免前视所需。2026-11-08 免费 API 目录显示 `btc/SplyExNtv/1d` 为 `community=true`，覆盖 2011-04-24 至 2026-11-06；`btc/SplyExPitNtv/1d` 覆盖 2009-01-03 至 2026-11-06，但不标 `community`。匿名请求后者返回 HTTP **403**：`Requested metric 'SplyExPitNtv' with frequency '1d' for asset 'btc' is not available with supplied credentials.` 仅查目录/授权状态，未读历史数值。

**决定：当前不拿免费标准序列伪装成独立 OOS，也不在同一历史上先试有利规则。** 如果未来取得 `SplyExPitNtv` 合法访问权，须先独立预注册资产、窗口、方向、成本、样本及 Gate，再首次读该规则结果。申请入口为 [Coin Metrics 官方联系页面](https://coinmetrics.io/contact/)；费用和可用许可未获确认，不自动购买。

为开始积累真正按本机首次见到时刻保存的未来样本，新增只读、非交易的 `coinmetrics_exchange_supply_journal`：每天抓取免费 `SplyExNtv` 最近 63 日，记响应哈希、抓取时刻、每日数值的首次见到版本及之后的变更；按决策时刻读取只允许访问该时刻之前已抓到的版本。这不会使 2021–2025 旧历史变为点时数据，也不同于供应商的 `Pit` 地址归属方法。日志只存用户本机应用支持目录，不推送原始数值到公开 GitHub；单日一次、无仓位/订单。未来前瞻样本仍须积累、核对数据权利和执行成本，才能形成严格 OOS。
