"""One facade for the existing discrete and continuous backtest paths."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import pandas as pd

import portfolio_metrics
from engine import run_backtest
from risk_overlay import apply_risk_overlay

from .contracts import ResearchRunConfig


def _json_safe(value):
    """Convert numpy/pandas scalars and non-finite metrics to strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not pd.notna(value):
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


class ResearchRunner:
    def __init__(self, registry=None):
        self.registry = registry

    def run_exposure(self, data, target_exposure, config=None, experiment_id=None):
        config = config or ResearchRunConfig()
        if config.allow_short:
            raise ValueError("long/short execution is not implemented by risk_overlay")
        if experiment_id is not None:
            if self.registry is None or self.registry.experiment(experiment_id) is None:
                raise ValueError("experiment must be preregistered before backtest")
        exposure = pd.Series(target_exposure, index=data.index, dtype=float)
        if ((exposure < 0) | (exposure > 1)).any():
            raise ValueError("long-only target exposure must stay in [0, 1]")
        simulation = apply_risk_overlay(
            data,
            exposure,
            config=config.risk,
            delay=config.execution_delay_bars,
            fee_rate=config.cost.fee_rate,
            slippage_rate=config.cost.slippage_rate,
        )
        metrics = portfolio_metrics.calculate_extended_metrics(
            simulation, initial_capital=config.initial_capital
        )
        result = {
            "engine": "continuous_exposure",
            "metrics": metrics,
            "equity_curve": simulation,
            "config": config.snapshot(),
        }
        if experiment_id is not None:
            self.registry.record_result(
                experiment_id,
                recorded_us=time.time_ns() // 1000,
                metrics=metrics,
                artifacts={"engine": result["engine"]},
            )
        return result

    def run_discrete(self, signal_df, sleeves, backtest_config):
        result = run_backtest(signal_df, sleeves, config=backtest_config)
        return {
            "engine": "legacy_discrete",
            "metrics": {
                "initial_capital": result.initial_capital,
                "final_value": result.final_value,
                "total_return": result.final_value / result.initial_capital - 1.0,
                "total_fees": result.total_fees,
                "trade_count": len(result.trades),
            },
            "equity_curve": result.equity_curve,
            "trades": result.trades,
            "config": asdict(backtest_config),
        }

    @staticmethod
    def write_report(result, output_dir, run_id, provenance=None):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        equity_path = destination / f"{run_id}_equity.csv"
        metrics_path = destination / f"{run_id}_metrics.json"
        manifest_path = destination / f"{run_id}_manifest.json"
        result["equity_curve"].to_csv(equity_path, index=False)
        metrics_path.write_text(
            json.dumps(
                _json_safe(result["metrics"]), indent=2,
                sort_keys=True, allow_nan=False
            ),
            encoding="utf-8",
        )
        manifest = {
            "run_id": run_id,
            "engine": result["engine"],
            "config": result["config"],
            "provenance": provenance or {},
            "artifacts": {
                "equity": equity_path.name,
                "metrics": metrics_path.name,
            },
        }
        if "trades" in result:
            trades_path = destination / f"{run_id}_trades.csv"
            result["trades"].to_csv(trades_path, index=False)
            manifest["artifacts"]["trades"] = trades_path.name
        manifest_path.write_text(
            json.dumps(
                _json_safe(manifest), indent=2,
                sort_keys=True, allow_nan=False
            ),
            encoding="utf-8",
        )
        return manifest
