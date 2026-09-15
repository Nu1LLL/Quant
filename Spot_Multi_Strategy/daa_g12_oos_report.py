"""Run the preregistered published DAA-G12 replication and temporal OOS test."""
from pathlib import Path
import hashlib

import pandas as pd

from cross_asset_trend import load_adjusted_close
import daa_g12_oos as strategy
import strict_validation


START, END = "2005-01-01", "2026-09-15"


def evaluate(returns, independent_oos, minimum_years):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=independent_oos,
        costs_included=True, minimum_years=minimum_years,
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 5 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    result["checks"]["worst_rolling_3y_sharpe_at_least_1"] = bool(
        result["metrics"]["worst_rolling_3y_sharpe"] >= 1.0
    )
    result["checks"]["solvent"] = not pd.Series(returns).le(-1.0).any()
    result["passed"] = all(result["checks"].values())
    return result


def run_experiment(prices, simulations=5000):
    audit = strategy.coverage_audit(prices)
    if not audit["passed"]:
        raise ValueError(f"DAA-G12 coverage failed: {audit}")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, positions, month_end, scores, detail = strategy.run_backtest(
            prices, leverage=leverage
        )
        samples = {
            "full_replication": returns,
            "paper_era": returns.loc[:"2018-12-31"],
            "post_publication_oos": returns.loc[strategy.POST_PUBLICATION_START:],
        }
        validations = {
            "full_replication": evaluate(samples["full_replication"], False, 18.5),
            "paper_era": evaluate(samples["paper_era"], False, 10.0),
            "post_publication_oos": evaluate(
                samples["post_publication_oos"], True, 7.5
            ),
        }
        results[leverage] = {
            "returns": returns, "positions": positions, "month_end": month_end,
            "scores": scores, "detail": detail, "samples": samples,
            "validations": validations,
        }
    primary_oos = results[1.0]["samples"]["post_publication_oos"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary_oos, simulations=simulations, block_length=63, seed=20181231
    )
    full = results[1.0]["returns"]
    events = pd.Series(
        [str(year) if year in (2008, 2020, 2022) else "other_days"
         for year in full.index.year], index=full.index,
    )
    eras = pd.Series("paper_era", index=full.index)
    eras.loc[full.index >= strategy.POST_PUBLICATION_START] = "post_publication_oos"
    regimes = pd.concat([
        strict_validation.regime_metrics(full, events).assign(family="event"),
        strict_validation.regime_metrics(full, eras).assign(family="evidence_era"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, audit


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(refresh=False):
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_daa_g12_replication"
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    raw = {
        symbol: load_adjusted_close(symbol, START, END, inputs, refresh=refresh)
        for symbol in strategy.SYMBOLS
    }
    prices = strategy.align_prices(raw)
    audit = strategy.coverage_audit(prices)
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    if not audit["passed"]:
        raise ValueError(f"DAA-G12 coverage failed before performance: {audit}")
    results, monte_carlo, regimes, _ = run_experiment(prices)
    prices.rename_axis("date").to_csv(output / "aligned_prices.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        pd.DataFrame({
            "net_return": result["returns"],
            "equity": 10000.0 * (1.0 + result["returns"]).cumprod(),
        }).to_csv(output / f"daily_{suffix}.csv")
        result["positions"].rename_axis("date").to_csv(output / f"positions_{suffix}.csv")
        result["month_end"].rename_axis("signal_date").to_csv(
            output / f"month_end_targets_{suffix}.csv"
        )
        result["detail"].to_csv(output / f"cost_detail_{suffix}.csv")
        for sample, validation in result["validations"].items():
            validation["walk_forward_folds"].to_csv(
                output / f"walk_forward_{sample}_{suffix}.csv", index=False
            )
            rows.append({
                "leverage": leverage, "sample": sample,
                **{f"oos__{key}": value for key, value in validation["metrics"].items()},
                **{f"check__{key}": value for key, value in validation["checks"].items()},
                "strict_oos_passed": validation["passed"],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_post_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    costs = results[1.0]["detail"][["turnover", "trading_cost", "financing_cost"]].sum()
    costs.to_frame("total_fraction").to_csv(output / "cost_totals_1x.csv")
    frequency = results[1.0]["month_end"].gt(0).mean().rename("positive_weight_fraction")
    frequency.to_csv(output / "holding_frequency_1x.csv")
    hashes = {symbol: _sha256(inputs / f"{symbol}_{START}_{END}.csv")
              for symbol in strategy.SYMBOLS}
    hashes["aligned_panel"] = _sha256(output / "aligned_prices.csv")
    pd.Series(hashes, name="sha256").to_csv(output / "data_hashes.csv")

    columns = ["leverage", "sample", "oos__final_value", "oos__cagr",
               "oos__sharpe_ratio", "oos__max_drawdown", "oos__profit_factor",
               "oos__positive_complete_year_ratio", "oos__worst_rolling_3y_sharpe",
               "check__walk_forward_passed", "strict_oos_passed"]
    print("COVERAGE", audit)
    print(summary[columns].to_string(index=False))
    print("MONTE_CARLO_POST_1X", monte_carlo)
    print("COST_TOTALS_1X", costs.to_dict())
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
