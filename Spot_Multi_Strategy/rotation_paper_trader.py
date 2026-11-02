import argparse
import fcntl
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data import INTERVAL_TO_TIMEDELTA
from paper_trader import (
    PaperDataGapError,
    append_records,
    atomic_write_json,
    fetch_current_price,
    fetch_recent_closed_klines,
    utc_now_text
)
from rotation_strategy import (
    calculate_rotation_exposure,
    generate_rotation_targets,
    load_rotation_config
)


RESEARCH_STATUS = "research_candidate_not_release_qualified"


def calculate_rotation_config_hash(
    strategy_config,
    capital,
    fee_rate,
    slippage_rate
):
    # 把所有会影响模拟结果的参数变成固定指纹
    parameter_data = {
        "strategy": asdict(strategy_config),
        "capital": float(capital),
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate)
    }
    serialized = json.dumps(
        parameter_data,
        sort_keys=True,
        ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def create_rotation_paper_state(
    symbols,
    initial_capital,
    interval,
    config_hash
):
    # 新候选策略使用独立状态，不接触原模拟盘
    return {
        "version": 1,
        "strategy_status": RESEARCH_STATUS,
        "created_at": utc_now_text(),
        "last_run_at": None,
        "last_signal_time": None,
        "interval": interval,
        "config_hash": config_hash,
        "symbols": [symbol.upper() for symbol in symbols],
        "run_count": 0,
        "duplicate_run_count": 0,
        "data_gap_count": 0,
        "gap_acknowledgement_count": 0,
        "portfolio": {
            "initial_capital": float(initial_capital),
            "cash": float(initial_capital),
            "quantity": 0.0,
            "held_symbol": None,
            "entry_price": 0.0,
            "entry_time": None,
            "entry_fee": 0.0,
            "entry_total_cost": 0.0,
            "entry_account_value": 0.0,
            "total_fees": 0.0,
            "closed_trades": 0
        }
    }


def _sell_rotation_position(
    portfolio,
    raw_price,
    event_time,
    reason,
    fee_rate,
    slippage_rate
):
    # 卖出时把手续费和不利滑点都计入模拟结果
    symbol = portfolio["held_symbol"]
    exit_price = float(raw_price) * (1 - slippage_rate)
    gross_value = portfolio["quantity"] * exit_price
    exit_fee = gross_value * fee_rate
    net_value = gross_value - exit_fee
    net_profit = net_value - portfolio["entry_total_cost"]

    event = {
        "strategy": "volatility_sized_spot_rotation_candidate",
        "strategy_status": RESEARCH_STATUS,
        "symbol": symbol,
        "event_time": str(event_time),
        "event": "sell",
        "reason": reason,
        "price": exit_price,
        "quantity": portfolio["quantity"],
        "fee": exit_fee,
        "net_profit": net_profit
    }

    portfolio["cash"] += net_value
    portfolio["quantity"] = 0.0
    portfolio["held_symbol"] = None
    portfolio["entry_price"] = 0.0
    portfolio["entry_time"] = None
    portfolio["entry_fee"] = 0.0
    portfolio["entry_total_cost"] = 0.0
    portfolio["entry_account_value"] = 0.0
    portfolio["total_fees"] += exit_fee
    portfolio["closed_trades"] += 1
    return event


def _buy_rotation_position(
    portfolio,
    symbol,
    raw_price,
    event_time,
    exposure,
    fee_rate,
    slippage_rate
):
    # 按波动率算出的资金比例买入候选品种
    entry_price = float(raw_price) * (1 + slippage_rate)
    account_value = portfolio["cash"]
    cash_budget = min(
        portfolio["cash"],
        account_value * float(exposure)
    )
    quantity = cash_budget / (entry_price * (1 + fee_rate))
    gross_cost = quantity * entry_price
    entry_fee = gross_cost * fee_rate
    total_cost = gross_cost + entry_fee

    portfolio["cash"] -= total_cost
    portfolio["quantity"] = quantity
    portfolio["held_symbol"] = symbol
    portfolio["entry_price"] = entry_price
    portfolio["entry_time"] = str(event_time)
    portfolio["entry_fee"] = entry_fee
    portfolio["entry_total_cost"] = total_cost
    portfolio["entry_account_value"] = account_value
    portfolio["total_fees"] += entry_fee

    return {
        "strategy": "volatility_sized_spot_rotation_candidate",
        "strategy_status": RESEARCH_STATUS,
        "symbol": symbol,
        "event_time": str(event_time),
        "event": "buy",
        "reason": "target_changed",
        "price": entry_price,
        "quantity": quantity,
        "fee": entry_fee,
        "exposure": float(exposure)
    }


def process_rotation_state(
    original_state,
    original_frames,
    current_prices,
    strategy_config,
    interval,
    fee_rate,
    slippage_rate,
    event_time=None,
    acknowledge_gap=False
):
    # 纯计算函数，不访问网络、不写文件，方便自动测试
    state = deepcopy(original_state)
    frames, targets = generate_rotation_targets(
        original_frames,
        strategy_config
    )
    latest_signal_time = pd.Timestamp(targets.index[-1])
    desired_value = targets.iloc[-1]
    desired_symbol = (
        str(desired_value)
        if pd.notna(desired_value)
        else None
    )
    event_time = event_time or utc_now_text()
    events = []

    if state["last_signal_time"] is None:
        state["last_signal_time"] = latest_signal_time.isoformat()
        return state, events, "initialized", frames, desired_symbol

    previous_signal_time = pd.Timestamp(state["last_signal_time"])

    if latest_signal_time < previous_signal_time:
        raise PaperDataGapError("最新K线时间早于轮动模拟盘状态")

    if latest_signal_time == previous_signal_time:
        return state, events, "duplicate", frames, desired_symbol

    if interval not in INTERVAL_TO_TIMEDELTA:
        raise ValueError(f"暂不支持K线周期：{interval}")

    expected_interval = INTERVAL_TO_TIMEDELTA[interval]
    signal_time_distance = latest_signal_time - previous_signal_time

    if signal_time_distance != expected_interval:

        if not acknowledge_gap:
            raise PaperDataGapError(
                "轮动模拟盘漏掉了K线，拒绝用历史价格追补实时交易"
            )

        missed_bars = max(
            int(signal_time_distance / expected_interval) - 1,
            1
        )
        state["last_signal_time"] = latest_signal_time.isoformat()
        state["gap_acknowledgement_count"] = (
            state.get("gap_acknowledgement_count", 0) + 1
        )
        events.append(
            {
                "strategy": "volatility_sized_spot_rotation_candidate",
                "strategy_status": RESEARCH_STATUS,
                "event_time": str(event_time),
                "event": "data_gap_acknowledged",
                "reason": "skip_historical_bars_without_trading",
                "previous_signal_time": previous_signal_time.isoformat(),
                "latest_signal_time": latest_signal_time.isoformat(),
                "missed_bars": missed_bars,
                "target_symbol": desired_symbol
            }
        )
        return (
            state,
            events,
            "gap_acknowledged_no_trade",
            frames,
            desired_symbol
        )

    portfolio = state["portfolio"]

    # 目标改变时先卖旧仓，再决定是否买入新目标
    if (
        portfolio["held_symbol"] is not None
        and
        portfolio["held_symbol"] != desired_symbol
    ):
        held_symbol = portfolio["held_symbol"]
        events.append(
            _sell_rotation_position(
                portfolio=portfolio,
                raw_price=current_prices[held_symbol],
                event_time=event_time,
                reason="target_changed",
                fee_rate=fee_rate,
                slippage_rate=slippage_rate
            )
        )

    if (
        portfolio["held_symbol"] is None
        and
        desired_symbol is not None
    ):
        exposure = calculate_rotation_exposure(
            frames=frames,
            desired_symbol=desired_symbol,
            signal_position=-1,
            config=strategy_config
        )
        events.append(
            _buy_rotation_position(
                portfolio=portfolio,
                symbol=desired_symbol,
                raw_price=current_prices[desired_symbol],
                event_time=event_time,
                exposure=exposure,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate
            )
        )

    state["last_signal_time"] = latest_signal_time.isoformat()
    return state, events, "processed", frames, desired_symbol


def calculate_portfolio_equity(portfolio, current_prices):
    # 当前总权益等于现金加上持仓的实时模拟市值
    equity = float(portfolio["cash"])

    if portfolio["held_symbol"] is not None:
        equity += (
            portfolio["quantity"]
            * float(current_prices[portfolio["held_symbol"]])
        )

    return equity


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="公开行情只读轮动候选策略模拟盘"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument(
        "--strategy-config",
        default="configs/rotation_volatility_candidate.json"
    )
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument(
        "--state-folder",
        default="paper_state_rotation_candidate"
    )
    parser.add_argument(
        "--acknowledge-gap",
        action="store_true",
        help="确认断档并跳过历史K线，不产生模拟交易"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    state_folder = Path(args.state_folder)

    if not state_folder.is_absolute():
        state_folder = project_folder / state_folder

    state_folder.mkdir(parents=True, exist_ok=True)
    lock_handle = (state_folder / ".rotation_paper.lock").open("w")

    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError as error:
        raise RuntimeError("另一个轮动模拟盘进程仍在运行") from error

    if (state_folder / "EMERGENCY_STOP").exists():
        print("轮动模拟盘紧急停止开关已启用，本次不读取行情。")
        return

    config_path = Path(args.strategy_config)

    if not config_path.is_absolute():
        config_path = project_folder / config_path

    strategy_config = load_rotation_config(config_path)
    config_hash = calculate_rotation_config_hash(
        strategy_config=strategy_config,
        capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage
    )
    requested_symbols = [symbol.upper() for symbol in args.symbols]
    state_file = state_folder / "state.json"

    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as file:
            state = json.load(file)

        if state["config_hash"] != config_hash:
            raise RuntimeError(
                "候选策略参数已变化，拒绝继续使用旧候选状态"
            )

        if state["interval"] != args.interval:
            raise RuntimeError("K线周期已变化，拒绝继续使用旧候选状态")

        if set(state["symbols"]) != set(requested_symbols):
            raise RuntimeError("交易品种已变化，拒绝继续使用旧候选状态")
    else:
        state = create_rotation_paper_state(
            symbols=requested_symbols,
            initial_capital=args.capital,
            interval=args.interval,
            config_hash=config_hash
        )

    frames = {
        symbol: fetch_recent_closed_klines(
            symbol=symbol,
            interval=args.interval
        )
        for symbol in state["symbols"]
    }
    current_prices = {
        symbol: fetch_current_price(symbol)
        for symbol in state["symbols"]
    }
    run_time = utc_now_text()

    try:
        new_state, events, status, _, desired_symbol = (
            process_rotation_state(
                original_state=state,
                original_frames=frames,
                current_prices=current_prices,
                strategy_config=strategy_config,
                interval=args.interval,
                fee_rate=args.fee,
                slippage_rate=args.slippage,
                event_time=run_time,
                acknowledge_gap=args.acknowledge_gap
            )
        )
    except PaperDataGapError:
        state["data_gap_count"] += 1
        atomic_write_json(state_file, state)
        raise

    if status == "duplicate":
        new_state["duplicate_run_count"] += 1

    new_state["last_run_at"] = utc_now_text()
    new_state["run_count"] += 1
    equity = calculate_portfolio_equity(
        new_state["portfolio"],
        current_prices
    )
    equity_record = {
        "run_time": run_time,
        "status": status,
        "target_symbol": desired_symbol,
        "held_symbol": new_state["portfolio"]["held_symbol"],
        "cash": new_state["portfolio"]["cash"],
        "quantity": new_state["portfolio"]["quantity"],
        "equity": equity
    }

    atomic_write_json(state_file, new_state)
    append_records(state_folder / "events.csv", events)
    append_records(state_folder / "equity.csv", [equity_record])

    print("\n研究候选轮动模拟运行完成")
    print(f"当前目标：{desired_symbol or '现金'}")
    print(f"当前持仓：{new_state['portfolio']['held_symbol'] or '现金'}")
    print(f"模拟权益：{equity:.2f} USDT")
    print(f"本次事件数：{len(events)}")
    print(f"状态位置：{state_folder}")
    print("候选尚未通过近年压力测试，禁止实盘。")
    print("程序没有API密钥、账户连接或下单接口。")


if __name__ == "__main__":
    main()
