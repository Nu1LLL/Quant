"""Generate auditable BUY/SELL/FLAT files from standardized minute books."""
import argparse
from pathlib import Path

import pandas as pd

import alpha_lifecycle
import market_execution_costs
import microstructure_blackbox as blackbox
import microstructure_fill_simulator


def load_standardized_csv(path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "market", "venue", "symbol"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks metadata columns {sorted(required - set(frame.columns))}")
    if frame[["market", "venue", "symbol"]].nunique().max() != 1:
        raise ValueError(f"{path} must contain exactly one market/venue/symbol")
    return frame.set_index("timestamp").sort_index()


def run_symbol_frame(frame: pd.DataFrame, train_end,
                     model_version="microstructure-v2", feature_version="v2",
                     alpha_id="hmm-rbf-minute-v2") -> pd.DataFrame:
    train_end = pd.Timestamp(train_end)
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    else:
        train_end = train_end.tz_convert("UTC")
    market, venue, symbol = (
        str(frame[column].iloc[0]) for column in ("market", "venue", "symbol")
    )
    train = frame.loc[frame.index <= train_end]
    test = frame.loc[frame.index > train_end]
    if len(train) < 1000 or len(test) < 100:
        raise ValueError(f"{symbol} requires >=1000 train and >=100 test minutes")
    model, prediction = blackbox.train_and_predict(
        train, test, model_version=f"{model_version}:{market}:{venue}:{symbol}",
        feature_version=feature_version,
    )
    orders = blackbox.generate_orders(
        prediction, frame, model, venue=venue, symbol=symbol,
        explicit_cost_bps=market_execution_costs.round_trip_cost_bps(frame, market),
    )
    orders.insert(0, "market", market)
    orders.insert(3, "alpha_id", alpha_id)
    orders["raw_action"] = orders["action"]
    orders["lifecycle_status"] = "UNASSESSED"
    orders["lifecycle_weight"] = 0.0
    orders["deployable"] = False
    return orders


def run_directory(input_dir, train_end, output_file,
                  model_version="microstructure-v2", feature_version="v2",
                  alpha_id="hmm-rbf-minute-v2",
                  realized_returns_file=None, lifecycle_output=None,
                  fills_output=None) -> pd.DataFrame:
    files = sorted(Path(input_dir).glob("*.csv"))
    if not files:
        raise ValueError("No standardized minute-book CSV files found")
    outputs = []
    simulated_fills = []
    for path in files:
        frame = load_standardized_csv(path)
        symbol_orders = run_symbol_frame(
            frame, train_end, model_version=model_version,
            feature_version=feature_version,
            alpha_id=alpha_id,
        )
        outputs.append(symbol_orders)
        if fills_output is not None:
            simulated_fills.append(
                microstructure_fill_simulator.simulate_taker_fills(symbol_orders, frame)
            )
    result = pd.concat(outputs).sort_index()
    if realized_returns_file is not None:
        realized = pd.read_csv(realized_returns_file, parse_dates=["timestamp"])
        evaluated = alpha_lifecycle.evaluate_lifecycle(realized)
        evaluated = alpha_lifecycle.allocate_active_weights(evaluated, realized)
        result = gate_orders_with_lifecycle(result, evaluated)
        if lifecycle_output is not None:
            lifecycle_output = Path(lifecycle_output)
            lifecycle_output.parent.mkdir(parents=True, exist_ok=True)
            evaluated.to_csv(lifecycle_output, index=False)
    if fills_output is not None:
        fills_output = Path(fills_output)
        fills_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(simulated_fills, ignore_index=True).to_csv(fills_output, index=False)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.rename_axis("timestamp").to_csv(output_file)
    latest = result.groupby(["market", "venue", "symbol"], sort=True).tail(1)
    latest.rename_axis("timestamp").to_csv(
        output_file.with_name(f"{output_file.stem}_latest{output_file.suffix}")
    )
    return result


def gate_orders_with_lifecycle(orders: pd.DataFrame,
                               lifecycle: pd.DataFrame) -> pd.DataFrame:
    """Make only independently validated ACTIVE signals deployable."""
    keys = list(alpha_lifecycle.KEY_COLUMNS)
    required = {*keys, "status", "weight"}
    if not required.issubset(lifecycle.columns):
        raise ValueError(f"Lifecycle table missing {sorted(required - set(lifecycle.columns))}")
    result = orders.copy()
    status = lifecycle[keys + ["status", "weight"]].drop_duplicates(keys)
    timestamp_name = result.index.name
    result = result.reset_index().merge(status, how="left", on=keys).set_index(
        timestamp_name or "index"
    )
    result["lifecycle_status"] = result.pop("status").fillna("SHADOW")
    result["lifecycle_weight"] = result.pop("weight").fillna(0.0)
    result["deployable"] = result["lifecycle_status"].eq("ACTIVE")
    result.loc[~result["deployable"], "action"] = "FLAT"
    result.loc[~result["deployable"], "kelly_position"] = 0.0
    result.loc[result["deployable"], "kelly_position"] *= result.loc[
        result["deployable"], "lifecycle_weight"
    ]
    return result.sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", default="microstructure-v2")
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--alpha-id", default="hmm-rbf-minute-v2")
    parser.add_argument("--realized-returns")
    parser.add_argument("--lifecycle-output")
    parser.add_argument("--fills-output")
    arguments = parser.parse_args()
    result = run_directory(
        arguments.input_dir, arguments.train_end, arguments.output,
        arguments.model_version, arguments.feature_version, arguments.alpha_id,
        arguments.realized_returns,
        arguments.lifecycle_output, arguments.fills_output,
    )
    print(result.groupby("action").size().to_string())


if __name__ == "__main__":
    main()
