# 分钟信号成交与 Alpha 反馈：预注册

状态：实现前冻结。本模块只是市价单压力测试，不模拟未知队列位置的 maker 成交。

## 时序与成交

- `t` 时点的信号只使用到 `t` 的已完成数据。
- BUY 在 `t` 的 ask 开仓，在 `t+1` 的 bid 平仓；SELL 在 `t` 的 bid 开仓，在 `t+1` 的 ask 平仓。
- FLAT 不记为交易；最后一行因无 `t+1` 报价不可成交。
- 报价路径已包含 spread，净收益只额外扣除成本门槛中 `spread` 以外的佣金、冲击和融资，禁止重复扣 spread。
- 缺报价、信号过期、非法成本或时间间隔超过信号有效期时拒绝成交，不向前填充。

## 输出

每笔包含 `signal_timestamp/exit_timestamp/market/venue/symbol/alpha_id/action`、开平价、报价毛收益、额外成本、净收益、模型版本。该净收益是 `alpha_lifecycle.py` 的唯一准入输入。

## 验收

- 平价不变时 BUY 和 SELL 都因 spread+额外成本为负；
- 上涨路径的 BUY 优于 SELL，下跌路径反之；
- 更高的成本只能降低净收益；
- 未来报价变化不能改变之前已成交记录。
