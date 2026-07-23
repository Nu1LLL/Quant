def run_backtest(
    signal_df,
    initial_capital=5000,
    fee_rate=0.00075
):
    # 定义通用的只做多现货回测函数


    df = signal_df.copy()
    # 复制包含交易信号的K线数据