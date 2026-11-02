import argparse
import fcntl
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import BacktestConfig, validate_config
from config_io import load_strategy_config
from data import (
    BINANCE_KLINES_URL,
    INTERVAL_TO_TIMEDELTA,
    download_klines,
    validate_ohlcv
)
from strategies import generate_signals


BINANCE_TICKER_PRICE_URL = (
    "https://data-api.binance.vision/api/v3/ticker/price"
)


class PaperDataGapError(RuntimeError):
    pass


def utc_now_text():
    return datetime.now(timezone.utc).isoformat()


def calculate_config_hash(strategy_config, backtest_config):
    parameter_data = {
        "strategy": strategy_config.__dict__,
        "backtest": backtest_config.__dict__
    }
    serialized = json.dumps(
        parameter_data,
        sort_keys=True,
        ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def fetch_recent_closed_klines(
    symbol,
    interval,
    limit=1000,
    timeout=15,
    session=None,
    now=None
):
    # 只调用公开行情接口，不需要API密钥
    request_session = session or requests.Session()
    response = request_session.get(
        BINANCE_KLINES_URL,
        params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        },
        timeout=timeout
    )
    response.raise_for_status()
    raw_data = response.json()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]
    df = pd.DataFrame(raw_data, columns=columns)

    if df.empty:
        raise ValueError("公开行情接口没有返回K线")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )
    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )
    current_time = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else pd.Timestamp(now)
    )

    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")

    closed_df = df[df["close_time"] < current_time].copy()

    return validate_ohlcv(
        closed_df,
        interval=interval,
        strict_continuity=True
    )


def fetch_current_price(
    symbol,
    timeout=15,
    session=None
):
    request_session = session or requests.Session()
    response = request_session.get(
        BINANCE_TICKER_PRICE_URL,
        params={"symbol": symbol.upper()},
        timeout=timeout
    )
    response.raise_for_status()
    price = float(response.json()["price"])

    if price <= 0:
        raise ValueError("当前价格必须大于0")

    return price


def update_paper_market_cache(
    symbol,
    interval,
    cache_folder,
    lookback_days=260,
    now=None
):
    """Maintain enough 1m history for closed 1D/4H/1H/15m context."""
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    delta = INTERVAL_TO_TIMEDELTA[interval]
    seconds = int(delta.total_seconds())
    boundary = pd.Timestamp(
        int(current.timestamp()) // seconds * seconds,
        unit="s",
        tz="UTC"
    )
    desired_start = boundary - pd.Timedelta(days=lookback_days)
    folder = Path(cache_folder)
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / f"{symbol.upper()}_{interval}.csv"

    if cache_path.exists():
        cached = validate_ohlcv(
            pd.read_csv(cache_path), interval=interval, strict_continuity=True
        )
        download_start = cached.iloc[-1]["open_time"] + delta
    else:
        cached = pd.DataFrame()
        download_start = desired_start

    if download_start < boundary:
        fresh = download_klines(
            symbol=symbol,
            interval=interval,
            start_time=download_start,
            end_time=boundary
        )
        combined = pd.concat([cached, fresh], ignore_index=True)
    else:
        combined = cached
    combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
    combined = (
        combined.drop_duplicates("open_time", keep="last")
        .sort_values("open_time")
    )
    combined = combined[combined["open_time"] >= desired_start]
    combined = validate_ohlcv(
        combined, interval=interval, strict_continuity=True
    )
    combined.to_csv(cache_path, index=False)
    return combined


def create_account(initial_capital):
    return {
        "initial_capital": float(initial_capital),
        "cash": float(initial_capital),
        "quantity": 0.0,
        "side": "flat",
        "entry_price": 0.0,
        "entry_time": None,
        "entry_fee": 0.0,
        "entry_account_value": 0.0,
        "stop_price": 0.0,
        "target_price": 0.0,
        "locked_margin": 0.0,
        "highest_close": 0.0,
        "lowest_close": 0.0,
        "last_signal_time": None,
        "total_fees": 0.0,
        "total_trading_fees": 0.0,
        "total_funding_fees": 0.0,
        "open_trade_funding_fee": 0.0,
        "closed_trades": 0
    }


def create_paper_state(
    symbols,
    total_capital,
    interval,
    config_hash
):
    capital_per_symbol = total_capital / len(symbols)

    return {
        "version": 2,
        "created_at": utc_now_text(),
        "last_run_at": None,
        "interval": interval,
        "config_hash": config_hash,
        "symbols": [symbol.upper() for symbol in symbols],
        "run_count": 0,
        "duplicate_run_count": 0,
        "data_gap_count": 0,
        "accounts": {
            symbol.upper(): create_account(capital_per_symbol)
            for symbol in symbols
        }
    }


def _paper_sell(
    account,
    symbol,
    event_time,
    raw_price,
    reason,
    backtest_config
):
    side = account.get("side", "long")
    if side == "flat" and account["quantity"] > 0:
        side = "long"
    exit_price = raw_price * (
        1 - backtest_config.slippage_rate
        if side == "long"
        else 1 + backtest_config.slippage_rate
    )
    gross_value = account["quantity"] * exit_price
    fee_rate = (
        backtest_config.short_fee_rate
        if side == "short" and backtest_config.short_fee_rate is not None
        else backtest_config.fee_rate
    )
    exit_fee = gross_value * fee_rate
    gross_profit = (
        (exit_price - account["entry_price"]) * account["quantity"]
        if side == "long"
        else (account["entry_price"] - exit_price) * account["quantity"]
    )
    funding_fee = account.get("open_trade_funding_fee", 0.0)
    net_profit = gross_profit - account["entry_fee"] - exit_fee - funding_fee
    trade = {
        "symbol": symbol,
        "entry_time": account["entry_time"],
        "exit_time": str(event_time),
        "entry_price": account["entry_price"],
        "exit_price": exit_price,
        "side": side,
        "quantity": account["quantity"],
        "entry_fee": account["entry_fee"],
        "exit_fee": exit_fee,
        "trading_fee": account["entry_fee"] + exit_fee,
        "funding_fee": funding_fee,
        "entry_account_value": account["entry_account_value"],
        "net_profit": net_profit,
        "exit_reason": reason
    }
    if side == "long":
        account["cash"] += gross_value - exit_fee
    else:
        account["cash"] += (
            account.get("locked_margin", 0.0)
            + gross_profit
            - exit_fee
        )
    account["quantity"] = 0.0
    account["side"] = "flat"
    account["entry_price"] = 0.0
    account["entry_time"] = None
    account["entry_fee"] = 0.0
    account["entry_account_value"] = 0.0
    account["stop_price"] = 0.0
    account["target_price"] = 0.0
    account["locked_margin"] = 0.0
    account["highest_close"] = 0.0
    account["lowest_close"] = 0.0
    account["total_fees"] += exit_fee
    account["total_trading_fees"] = (
        account.get("total_trading_fees", 0.0) + exit_fee
    )
    account["open_trade_funding_fee"] = 0.0
    account["closed_trades"] += 1
    return trade


def process_account(
    original_account,
    symbol,
    signal_df,
    sleeve,
    current_price,
    interval,
    backtest_config,
    event_time=None
):
    # 纯计算函数，便于自动测试；不会访问网络或写文件
    account = deepcopy(original_account)
    latest_row = signal_df.iloc[-1]
    latest_signal_time = pd.Timestamp(latest_row["open_time"])
    interval_delta = INTERVAL_TO_TIMEDELTA[interval]
    event_time = event_time or utc_now_text()
    events = []

    if account["last_signal_time"] is None:
        account["last_signal_time"] = latest_signal_time.isoformat()
        return account, events, "initialized"

    previous_signal_time = pd.Timestamp(account["last_signal_time"])

    if latest_signal_time < previous_signal_time:
        raise PaperDataGapError("最新K线时间早于模拟盘状态")

    if latest_signal_time == previous_signal_time:
        position_side = account.get("side", "flat")
        if position_side == "flat" and account["quantity"] > 0:
            position_side = "long"
        observed_stop = (
            account["quantity"] > 0
            and
            (
                (
                    position_side == "long"
                    and current_price <= account["stop_price"]
                )
                or
                (
                    position_side == "short"
                    and current_price >= account["stop_price"]
                )
            )
        )
        observed_target = (
            account["quantity"] > 0
            and account.get("target_price", 0.0) > 0
            and
            (
                (
                    position_side == "long"
                    and current_price >= account["target_price"]
                )
                or
                (
                    position_side == "short"
                    and current_price <= account["target_price"]
                )
            )
        )
        if (
            observed_stop or observed_target
        ):
            events.append(
                _paper_sell(
                    account,
                    symbol,
                    event_time,
                    current_price,
                    (
                        "observed_price_stop"
                        if observed_stop
                        else "observed_price_take_profit"
                    ),
                    backtest_config
                )
            )
            return account, events, (
                "stop" if observed_stop else "take_profit"
            )

        return account, events, "duplicate"

    if latest_signal_time - previous_signal_time != interval_delta:
        raise PaperDataGapError(
            "模拟盘漏掉了K线，拒绝使用历史数据追补实时交易"
        )

    exited_this_run = False
    entered_this_run = False

    if account.get("side") == "short":
        funding_rate = float(latest_row.get("funding_rate", 0.0))
        if pd.notna(funding_rate) and funding_rate != 0:
            funding_fee = (
                -account["quantity"] * current_price * funding_rate
            )
            account["cash"] -= funding_fee
            account["total_funding_fees"] = (
                account.get("total_funding_fees", 0.0) + funding_fee
            )
            account["open_trade_funding_fee"] = (
                account.get("open_trade_funding_fee", 0.0) + funding_fee
            )
            events.append(
                {
                    "symbol": symbol,
                    "event_time": str(event_time),
                    "event": "funding",
                    "side": "short",
                    "funding_rate": funding_rate,
                    "funding_fee": funding_fee
                }
            )

    if account["quantity"] > 0:
        position_side = account.get("side", "flat")
        if position_side == "flat":
            position_side = "long"
        bar_open = float(latest_row["open"])
        bar_high = float(latest_row["high"])
        bar_low = float(latest_row["low"])
        target_price = account.get("target_price", 0.0)
        stop_hit = (
            (position_side == "long" and bar_low <= account["stop_price"])
            or
            (position_side == "short" and bar_high >= account["stop_price"])
        )
        target_hit = (
            target_price > 0
            and
            (
                (position_side == "long" and bar_high >= target_price)
                or
                (position_side == "short" and bar_low <= target_price)
            )
        )
        exit_column = (
            sleeve.exit_column
            if position_side == "long"
            else sleeve.short_exit_column
        )

        if stop_hit:
            if position_side == "long":
                raw_exit_price = (
                    bar_open if bar_open <= account["stop_price"]
                    else account["stop_price"]
                )
            else:
                raw_exit_price = (
                    bar_open if bar_open >= account["stop_price"]
                    else account["stop_price"]
                )
            events.append(
                _paper_sell(
                    account,
                    symbol,
                    event_time,
                    raw_exit_price,
                    "bar_stop",
                    backtest_config
                )
            )
            exited_this_run = True
        elif target_hit:
            if position_side == "long":
                raw_exit_price = (
                    bar_open if bar_open >= target_price else target_price
                )
            else:
                raw_exit_price = (
                    bar_open if bar_open <= target_price else target_price
                )
            events.append(
                _paper_sell(
                    account,
                    symbol,
                    event_time,
                    raw_exit_price,
                    "bar_take_profit",
                    backtest_config
                )
            )
            exited_this_run = True
        elif exit_column is not None and bool(latest_row[exit_column]):
            events.append(
                _paper_sell(
                    account,
                    symbol,
                    event_time,
                    current_price,
                    "signal",
                    backtest_config
                )
            )
            exited_this_run = True

    if (
        account["quantity"] == 0
        and
        not exited_this_run
    ):
        long_entry = bool(latest_row[sleeve.entry_column])
        short_entry = (
            sleeve.short_entry_column is not None
            and bool(latest_row[sleeve.short_entry_column])
        )
        side = None
        if long_entry and not short_entry:
            side = "long"
        elif short_entry and not long_entry:
            side = "short"

        entry_price = current_price * (
            1 + backtest_config.slippage_rate
            if side == "long"
            else 1 - backtest_config.slippage_rate
        ) if side is not None else 0.0
        structural_stop_column = (
            sleeve.long_stop_column
            if side == "long"
            else sleeve.short_stop_column
            if side == "short"
            else None
        )
        if structural_stop_column is not None:
            raw_stop = latest_row[structural_stop_column]
            stop_price = float(raw_stop) if pd.notna(raw_stop) else 0.0
        elif (
            side is not None
            and sleeve.stop_atr_multiple is not None
            and pd.notna(latest_row.get("atr"))
        ):
            stop_distance = (
                float(latest_row["atr"]) * sleeve.stop_atr_multiple
            )
            stop_price = (
                entry_price - stop_distance
                if side == "long"
                else entry_price + stop_distance
            )
        else:
            stop_price = 0.0

        valid_stop = (
            0 < stop_price < entry_price
            if side == "long"
            else stop_price > entry_price
            if side == "short"
            else False
        )
        stop_distance = abs(entry_price - stop_price)

        if side is not None and valid_stop:
            estimated_stop_fill = stop_price * (
                1 - backtest_config.slippage_rate
                if side == "long"
                else 1 + backtest_config.slippage_rate
            )
            fee_rate = (
                backtest_config.short_fee_rate
                if side == "short" and backtest_config.short_fee_rate is not None
                else backtest_config.fee_rate
            )
            risk_per_unit = (
                abs(entry_price - estimated_stop_fill)
                +
                entry_price * fee_rate
                +
                estimated_stop_fill * fee_rate
            )
            risk_budget = (
                account["cash"]
                *
                backtest_config.risk_per_trade
            )
            quantity_by_risk = risk_budget / risk_per_unit
            maximum_gross_cost = (
                account["cash"]
                *
                backtest_config.max_sleeve_exposure
                /
                (1 + fee_rate)
            )
            maximum_gross_cost = min(
                maximum_gross_cost,
                backtest_config.max_long_notional
                if side == "long"
                else backtest_config.max_short_notional
            )
            quantity_by_cash = maximum_gross_cost / entry_price
            quantity = min(quantity_by_risk, quantity_by_cash)

            if quantity > 0:
                account_value = account["cash"]
                gross_cost = quantity * entry_price
                entry_fee = gross_cost * fee_rate
                account["cash"] -= gross_cost + entry_fee
                account["quantity"] = quantity
                account["side"] = side
                account["entry_price"] = entry_price
                account["entry_time"] = str(event_time)
                account["entry_fee"] = entry_fee
                account["entry_account_value"] = account_value
                account["stop_price"] = stop_price
                account["locked_margin"] = (
                    gross_cost if side == "short" else 0.0
                )
                account["target_price"] = (
                    entry_price
                    + (
                        1 if side == "long" else -1
                    )
                    * stop_distance
                    * sleeve.take_profit_r_multiple
                    if sleeve.take_profit_r_multiple is not None
                    else 0.0
                )
                account["highest_close"] = entry_price
                account["lowest_close"] = entry_price
                account["total_fees"] += entry_fee
                account["total_trading_fees"] = (
                    account.get("total_trading_fees", 0.0) + entry_fee
                )
                account["open_trade_funding_fee"] = 0.0
                entered_this_run = True
                events.append(
                    {
                        "symbol": symbol,
                        "event_time": str(event_time),
                        "event": "buy" if side == "long" else "short_sell",
                        "side": side,
                        "price": entry_price,
                        "quantity": quantity,
                        "fee": entry_fee,
                        "stop_price": stop_price,
                        "target_price": account["target_price"]
                    }
                )

    if (
        account["quantity"] > 0
        and
        not entered_this_run
        and
        sleeve.trailing_atr_multiple is not None
        and
        pd.notna(latest_row["atr"])
    ):
        if account.get("side", "long") == "long":
            account["highest_close"] = max(
                account["highest_close"], float(latest_row["close"])
            )
            trailing_stop = (
                account["highest_close"]
                - float(latest_row["atr"]) * sleeve.trailing_atr_multiple
            )
            account["stop_price"] = max(account["stop_price"], trailing_stop)
        else:
            account["lowest_close"] = min(
                account["lowest_close"], float(latest_row["close"])
            )
            trailing_stop = (
                account["lowest_close"]
                + float(latest_row["atr"]) * sleeve.trailing_atr_multiple
            )
            account["stop_price"] = min(account["stop_price"], trailing_stop)

    account["last_signal_time"] = latest_signal_time.isoformat()
    return account, events, "processed"


def atomic_write_json(file_path, data):
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(output_path)


def append_records(file_path, records):
    if not records:
        return

    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if output_path.exists():
        old_df = pd.read_csv(output_path)
        output_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        output_df = new_df

    temporary_path = output_path.with_suffix(".tmp")
    output_df.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="公开行情只读现货模拟盘"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument(
        "--strategy-config",
        default="configs/profit_first_simplified.json"
    )
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--long-cap", type=float, default=10_000.0)
    parser.add_argument("--state-folder", default="paper_state")
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    state_folder = Path(args.state_folder)

    if not state_folder.is_absolute():
        state_folder = project_folder / state_folder

    state_folder.mkdir(parents=True, exist_ok=True)
    # 防止定时任务重叠运行，造成同一信号被处理两次
    lock_handle = (state_folder / ".paper_trader.lock").open("w")

    try:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError as error:
        raise RuntimeError("另一个模拟盘进程仍在运行") from error

    emergency_stop_file = state_folder / "EMERGENCY_STOP"

    if emergency_stop_file.exists():
        print("模拟盘紧急停止开关已启用，本次不读取行情、不交易。")
        return

    config_path = Path(args.strategy_config)

    if not config_path.is_absolute():
        config_path = project_folder / config_path

    strategy_config = load_strategy_config(config_path)

    symbol_count = len(args.symbols)
    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk,
        # The paper ledger uses fixed per-symbol accounts. Divide the total
        # spot cap so simultaneous BTC/ETH positions cannot exceed it.
        max_long_notional=args.long_cap / symbol_count
    )
    validate_config(strategy_config, backtest_config)
    config_hash = calculate_config_hash(
        strategy_config,
        backtest_config
    )
    state_file = state_folder / "state.json"

    if state_file.exists():
        with state_file.open("r", encoding="utf-8") as file:
            state = json.load(file)

        if state["config_hash"] != config_hash:
            raise RuntimeError(
                "策略或风险参数已经变化，拒绝继续使用旧模拟盘状态"
            )

        if state["interval"] != args.interval:
            raise RuntimeError("K线周期已经变化，拒绝继续使用旧状态")

        requested_symbols = {
            symbol.upper() for symbol in args.symbols
        }

        if set(state["symbols"]) != requested_symbols:
            raise RuntimeError("交易品种已经变化，拒绝继续使用旧状态")
    else:
        state = create_paper_state(
            symbols=args.symbols,
            total_capital=args.capital,
            interval=args.interval,
            config_hash=config_hash
        )

    all_events = []
    equity_records = []
    run_time = utc_now_text()

    for symbol in state["symbols"]:
        closed_df = fetch_recent_closed_klines(
            symbol=symbol,
            interval=args.interval
        )
        signal_df, sleeves = generate_signals(
            closed_df,
            strategy_config
        )

        signal_df["funding_rate"] = 0.0
        if len(sleeves) != 1 or sleeves[0].weight != 1.0:
            raise ValueError(
                "当前模拟盘只接受一个权重100%的已验收策略"
            )

        current_price = fetch_current_price(symbol)

        try:
            account, events, status = process_account(
                original_account=state["accounts"][symbol],
                symbol=symbol,
                signal_df=signal_df,
                sleeve=sleeves[0],
                current_price=current_price,
                interval=args.interval,
                backtest_config=backtest_config,
                event_time=run_time
            )
        except PaperDataGapError:
            state["data_gap_count"] += 1
            atomic_write_json(state_file, state)
            raise

        state["accounts"][symbol] = account
        all_events.extend(events)

        if status == "duplicate":
            state["duplicate_run_count"] += 1

        if account.get("side", "flat") == "short":
            equity = (
                account["cash"]
                + account.get("locked_margin", 0.0)
                + (account["entry_price"] - current_price)
                * account["quantity"]
            )
        else:
            equity = account["cash"] + account["quantity"] * current_price
        equity_records.append(
            {
                "run_time": run_time,
                "symbol": symbol,
                "current_price": current_price,
                "cash": account["cash"],
                "quantity": account["quantity"],
                "equity": equity,
                "trading_fees": account.get("total_trading_fees", 0.0),
                "funding_fees": account.get("total_funding_fees", 0.0),
                "status": status
            }
        )

    state["last_run_at"] = utc_now_text()
    state["run_count"] += 1
    atomic_write_json(state_file, state)
    append_records(state_folder / "events.csv", all_events)
    append_records(state_folder / "equity.csv", equity_records)

    total_equity = sum(record["equity"] for record in equity_records)
    print("\n只读模拟盘运行完成")
    print(f"总模拟资产：{total_equity:.2f} USDT")
    print(f"本次事件数：{len(all_events)}")
    print(f"状态位置：{state_folder}")
    print("该程序没有下单代码，也不会连接真实账户。")


if __name__ == "__main__":
    main()
