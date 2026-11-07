# BTC MVRV 实验终态审计（2026-11-08）

状态：`CODE_FAILURE`，**无可报告收益、无 Alpha 结论、完整 Gate 未通过**。预注册 `46d10cb`，读数值前时间/编码修正 `65cc374`，独立代码提交 `6c05785`；三者均先于唯一一次正式运行推送。正式运行生成不可覆盖的 [`CODE_FAILURE.json`](CODE_FAILURE.json)，没有 `RESULT.json`、`DATA_FAILURE.json`，锁已释放。

失败原因为 `TypeError: report object keys must be strings`。静态复核发现复用的 `research/run_btc_fee_crypto.py::_json_safe` 拒绝非字符串字典键，而复用的 `research/wiki_attention_metrics.py::_ledger_metadata` 在 `conservative_max_drawdown_by_year` 中使用整数年份键。此前 BTC 手续费实验也以相同错误封存，预运行检查没有覆盖完整报告序列化，这是本次工程流程的失误。运行走到报告写入阶段时内存中已有计算对象，但进程退出后没有可复核的净值、成本或 Gate 数值，不能推测结果。禁止改写该终态或在同一五年历史上修补并重跑本假说。

下一条**经济机制独立、另行预注册**的实验应在提交代码前用合成账本完整序列化其拟输出的 `RESULT`，并检查所有嵌套键、非有限数、日期和 numpy 类型；这属于工程验收，不改变本实验的终态，也不能用本次未公开的内存结果选择下一假说。

数据来源限制仍在：Coin Metrics MVRV 当前修订版没有核实的每日历史发布版本，Binance Vision 档案不能证明真实盘口成交和保证金阶梯；本实验原本也不能构成独立可执行 OOS。币安官方[归档问题 #334](https://github.com/binance/binance-public-data/issues/334)称 USD-M 历史 `bookTicker` 自 2024 年停更，不能拿它替代连续五年 L1 盘口。
