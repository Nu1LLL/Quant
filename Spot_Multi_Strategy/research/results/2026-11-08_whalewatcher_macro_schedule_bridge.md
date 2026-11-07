# WhaleWatcher 宏观事件日程前瞻接入（非交易结果）

Quant 已在独立研究模块 `crypto_blackbox/whalewatcher_macro_journal.py` 中接入本机 WhaleWatcher 的 `event_state/macro.json`。现有小时采集命令在原新闻与方向性声明采集之后，额外记录 CPI、非农、PCE、FOMC 的事件身份哈希、已公布的日程时间、核验时刻、确认状态、官方来源网址哈希和本机首次采集时刻。原始网址、标题、AI 预测、Polymarket 赔率、公布值、复盘和价格均不复制到 Quant 日志。BLS 的静态官方快照来源与 BEA/Fed 的实时来源保留不同 provenance；未经确认的日程不能进入 `upcoming_confirmed`。

本机首次实际采集返回 `fresh; items=8; changes=8`，数据库仅 24,576 字节。四类事件均有元数据，其中未来七天有一个**已确认**事件。以首次采集时刻之前查询，结果为空，验证不会把今日已知的日程倒填到过去。原有 WhaleWatcher 新闻与 BTC/ETH 声明采集保持 `unchanged; items=164/0/0`。现有 LaunchAgent 仍按每 3,600 秒运行，上次退出码 0。相关 23 个合成测试通过，包括更正、撤回、数据源中断、重复 JSON 键、未确认事件和不可变数据库记录。

这份日程日志从今天开始积累，不具备五年历史、更没有预测收益。`poll_coverage` 把超过两小时的采集间断和任何不可用快照计为连续性失败；即使未来连续性通过，也仍需单独预注册信号、5 年真实独立 OOS、交易成本和完整 Gate。当前 `execution_eligible=false`，不生成交易指令。
