# 缠论第三类买卖点四市场实验：数据Gate结果

## 裁决

**数据失败，绩效未揭晓，不进入实盘。** 固定顺序取得ITOT后，Yahoo chart接口对
CYB返回0条时间戳且没有可用调整收盘价。程序随即在OHLC复权、结构识别、仓位和收益
计算之前终止；DBB与BIV未再请求。

不能在看见该结果后用裸收盘代替调整价、删除外汇袖套、把CYB换成另一只货币ETF，
或只回测已成功下载的ITOT。CAGR、Sharpe、最大回撤、PF、walk-forward及Monte
Carlo均为**未计算**，而不是0。

## 外部产品状态核对

WisdomTree 2023年基金关闭公告把Chinese Yuan Strategy Fund（CYB）列入关闭清单，
最后交易日为2023-10-20，随后按NAV自动赎回。官方文件：

- <https://www.wisdomtree.com/-/media/us-media-files/documents/about/pdf/2023/wisdomtree-announces-changes-to-etf-registered-products.pdf>
- <https://www.wisdomtree.com/-/media/us-media-files/documents/about/pdf/2023/fund-closure-faq-september-2023.pdf>

因此即使另一个供应商仍保存2018–2023历史，该固定组合也不满足预注册要求的
2026-08-31当前可交易终点。Yahoo当前无序列既是数据可得性失败，也与产品已清算一致。

## 已保存证据

`data_rejection.csv`记录固定下载顺序和停止位置。唯一在停止前保存的ITOT文件有4,955
条记录，SHA-256为
`6f47dfa58c87950127f56d785fde596d4715143a628dd62d26d600bdce8f935c`；该文件没有用于
生成任何绩效。CYB响应审计为0条时间戳，指标容器存在但无可用调整价数组。

## 方法与受保护范围

第三类买卖点的无未来函数实现及7项定向测试仍保留，供未来**新的预注册资产实验**复用；
本轮不得用同一预注册补选资产继续。提交`d9861c9`中的成熟策略核心和正式配置、既有
A26实现与报告，以及用户未提交的futures/hybrid/rotation/paper_state均未修改。
