def generate_signals(
    original_df
):
    # 定义MA策略信号函数


    df = original_df.copy()
    # 复制原始K线，防止策略修改外面的原始数据


    df["MA20"] = (
        df["close"]
        .rolling(window=20)
        .mean()
    )
    # 计算MA20


    df["MA60"] = (
        df["close"]
        .rolling(window=60)
        .mean()
    )
    # 计算MA60


    df["golden_cross"] = (
        (df["MA20"].shift(1) <= df["MA60"].shift(1))
        &
        (df["MA20"] > df["MA60"])
    )
    # 判断金叉


    df["death_cross"] = (
        (df["MA20"].shift(1) >= df["MA60"].shift(1))
        &
        (df["MA20"] < df["MA60"])
    )
    # 判断死叉


    df["long_entry"] = df["golden_cross"]
    # 把金叉转换成统一的做多进场信号


    df["long_exit"] = df["death_cross"]
    # 把死叉转换成统一的做多离场信号


    return df
    # 返回包含指标和统一交易信号的数据
    # 返回包含均线和交易信号的数据