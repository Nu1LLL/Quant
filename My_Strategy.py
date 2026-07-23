import requests
import pandas as pd
from Indicators.Get_Candlesticks import get_klines
# 获取K线数据的函数
from Strategies.MA_Signal import generate_signals
# 导入MA策略信号函数


df = get_klines(
    symbol="BTCUSDT",
    interval="15m",
    days=15,
    end_time="2026-07-22 00:00:00+00:00"
)
# 获取截至指定UTC时间之前15天的K线



# -----------------上面是获取K线数据的代码，下面是计算MA20的代码----------------------


df = generate_signals(
    original_df=df
)
# 使用MA策略模块生成金叉死叉信号


signals = df.loc[
    df["golden_cross"] | df["death_cross"],
    ["open_time", "close", "MA20", "MA60", "golden_cross", "death_cross"]
]
# 筛选金叉和死叉记录


position = 0
# 0空仓，1持仓

initial_capital = 5000
# 初始资金

cash = initial_capital
# 剩余资金

fee_rate = 0.00075
# 手续费率

total_fees = 0
# 总手续费

btc = 0
# 持有BTC数量


for i in range( len(df) -1 ):

    now_row = df.iloc[i]

    next_row = df.iloc[i + 1]

    if now_row["long_entry"] and position == 0:

        position = 1

        buy_fee = cash * fee_rate
        # 计算买入手续费

        total_fees += buy_fee
        # 累计总手续费

        btc = (cash - buy_fee) / next_row["open"]
        # 计算买入后持有的BTC数量

        cash = 0

        # print(f"{next_row['open_time']} - 买入信号，价格: {next_row['open']}, 持有BTC数量: {btc}, 剩余资金: {cash}")
    
    


    elif now_row["long_exit"] and position == 1:

        position = 0

        sell_fee = btc * next_row["open"] * fee_rate
        # 计算卖出手续费

        total_fees += sell_fee
        # 累计总手续费

        cash = btc * next_row["open"] - sell_fee
        # 计算卖出后剩余资金

        btc = 0

        # print(f"{next_row['open_time']} - 卖出信号，价格: {next_row['open']}, 剩余资金: {cash}")



last_price = df.iloc[-1]["close"]
# 获取最后一根K线的收盘价

if btc > 0:
    final_sell_fee = btc * last_price * fee_rate
    final_value = cash + btc * last_price - final_sell_fee
    total_fees += final_sell_fee
    # 最后如果持有BTC，计算最终资产价值

else:
    final_value = cash
    # 最后如果没有持有BTC，最终资产就是剩余资金
   

profit = final_value - initial_capital
# 盈亏 

return_rate = profit / initial_capital * 100
# 收益率 = 盈亏 / 初始资金 * 100

print("\n回测结果：")
print(f"初始资金：{initial_capital:.2f} USDT")
print(f"最终资产：{final_value:.2f} USDT")
print(f"盈亏：{profit:.2f} USDT")
print(f"收益率：{return_rate:.2f}%")
print(f"总手续费：{total_fees:.2f} USDT")       