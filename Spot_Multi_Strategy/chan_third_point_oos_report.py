"""Run the preregistered causal Chan third-point four-market study."""
from pathlib import Path
import hashlib

import pandas as pd
import requests

import chan_third_point_oos as strategy
import strict_validation


START = pd.Timestamp("2007-01-01", tz="UTC")
END = pd.Timestamp("2026-09-15", tz="UTC")
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def download_ohlc(symbol, session=None):
    client = session or requests.Session()
    response = client.get(
        YAHOO_URL.format(symbol=symbol),
        params={"period1": int(START.timestamp()), "period2": int(END.timestamp()),
                "interval": "1d", "events": "div,splits"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
    )
    response.raise_for_status()
    result = response.json().get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo returned no history for {symbol}")
    result = result[0]
    quote = result["indicators"]["quote"][0]
    adjusted = result.get("indicators", {}).get("adjclose")
    if not adjusted or "adjclose" not in adjusted[0]:
        raise ValueError(f"Yahoo returned no adjusted close for {symbol}")
    frame = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize(),
        "raw_open": quote["open"], "raw_high": quote["high"],
        "raw_low": quote["low"], "raw_close": quote["close"],
        "adjusted_close": adjusted[0]["adjclose"],
    })
    return frame.dropna().drop_duplicates("date", keep="last").set_index("date")


def load_ohlc(symbol, folder, refresh=False):
    path = Path(folder) / f"{symbol}_2007-01-01_2026-09-15_ohlc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame = download_ohlc(symbol)
    frame.rename_axis("date").to_csv(path)
    return frame


def evaluate(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=18.0
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 14 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    result["checks"]["worst_rolling_3y_sharpe_at_least_1"] = bool(
        result["metrics"]["worst_rolling_3y_sharpe"] >= 1.0
    )
    result["checks"]["solvent"] = not pd.Series(returns).le(-1.0).any()
    result["passed"] = all(result["checks"].values())
    return result


def run_experiment(markets, simulations=5000):
    audit = strategy.coverage_audit(markets)
    if not audit["passed"]:
        raise ValueError(f"Chan third-point coverage failed: {audit}")
    results = {}
    for leverage in (1.0, 2.0, 3.0, 4.0, 5.0):
        returns, positions, targets, boundaries, events, detail = strategy.run_portfolio(
            markets, leverage
        )
        results[leverage] = {
            "returns": returns, "positions": positions, "targets": targets,
            "boundaries": boundaries, "events": events, "detail": detail,
            "validation": evaluate(returns),
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=63, seed=20070120
    )
    periods = pd.Series(index=primary.index, dtype="object")
    periods.loc[primary.index.year <= 2012] = "2008_2012"
    periods.loc[(primary.index.year >= 2013) & (primary.index.year <= 2019)] = "2013_2019"
    periods.loc[primary.index.year >= 2020] = "2020_plus"
    events = pd.Series([str(y) if y in (2008, 2020, 2022) else "other_days"
                        for y in primary.index.year], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, periods).assign(family="period"),
        strict_validation.regime_metrics(primary, events).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, audit


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(refresh=False):
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_chan_third_point_oos"
    inputs = output / "inputs"
    raw = {symbol: load_ohlc(symbol, inputs, refresh=refresh)
           for symbol in strategy.SYMBOLS}
    markets = strategy.align_markets(raw)
    audit = strategy.coverage_audit(markets)
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    if not audit["passed"]:
        raise ValueError(f"Chan third-point coverage failed before performance: {audit}")
    results, monte_carlo, regimes, _ = run_experiment(markets)
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        pd.DataFrame({"net_return": result["returns"],
                      "equity": 10000.0 * (1.0 + result["returns"]).cumprod()}).to_csv(
            output / f"daily_{suffix}.csv"
        )
        result["positions"].to_csv(output / f"positions_{suffix}.csv")
        result["detail"].to_csv(output / f"cost_detail_{suffix}.csv")
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{suffix}.csv", index=False
        )
        rows.append({
            "leverage": leverage,
            **{f"oos__{key}": value for key, value in result["validation"]["metrics"].items()},
            **{f"check__{key}": value for key, value in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    results[1.0]["positions"].corr().to_csv(output / "positioned_return_correlation_1x.csv")
    event_rows, diagnostics = [], []
    for symbol in strategy.SYMBOLS:
        event = results[1.0]["events"][symbol].copy()
        event.insert(0, "symbol", symbol)
        event_rows.append(event)
        position = results[1.0]["positions"][symbol] * len(strategy.SYMBOLS)
        diagnostics.append({
            "symbol": symbol, "event_count": len(event),
            "buy_count": int((event["signal"] > 0).sum()) if len(event) else 0,
            "sell_count": int((event["signal"] < 0).sum()) if len(event) else 0,
            "long_day_fraction": float((position > 0).mean()),
            "short_day_fraction": float((position < 0).mean()),
            "flat_day_fraction": float((position == 0).mean()),
        })
    pd.concat(event_rows, ignore_index=True).to_csv(output / "events.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "signal_diagnostics.csv", index=False)
    hashes = {symbol: _sha256(inputs / f"{symbol}_2007-01-01_2026-09-15_ohlc.csv")
              for symbol in strategy.SYMBOLS}
    pd.Series(hashes, name="sha256").to_csv(output / "data_hashes.csv")

    columns = ["leverage", "oos__final_value", "oos__cagr", "oos__sharpe_ratio",
               "oos__max_drawdown", "oos__profit_factor",
               "oos__positive_complete_year_ratio", "oos__worst_rolling_3y_sharpe",
               "check__walk_forward_passed", "strict_oos_passed"]
    print("COVERAGE", audit)
    print(summary[columns].to_string(index=False))
    print("SIGNALS", pd.DataFrame(diagnostics).to_dict("records"))
    print("MONTE_CARLO", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
