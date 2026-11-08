"""Synthetic-only stablecoin source, no-lookahead decision and spot account checks."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import zipfile

import pandas as pd

from research.stablecoin_liquidity_source import (
    FIRST, LAST, SYMBOLS, LiquiditySourceFailure, parse_stablecoin,
)
import research.stablecoin_liquidity_source as source
from research.stablecoin_liquidity_sim import (
    ReplayFailure, evaluate, simulate, weekly_targets,
)


def _stablecoins():
    days = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    return pd.DataFrame({"USDT": [1000. + 10 * i for i in range(len(days))],
                         "USDC": [1000.] * len(days)}, index=days)


def _flat_prices(*, quote_volume=1e8):
    days = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    reference = {"BNBUSDT": 10., "BTCUSDT": 100., "ETHUSDT": 20.}
    frames = {symbol: pd.DataFrame({"open": [price] * len(days),
                                    "high": [price] * len(days),
                                    "low": [price] * len(days),
                                    "close": [price] * len(days),
                                    "quote_volume": [quote_volume] * len(days)},
                                   index=days)
              for symbol, price in reference.items()}
    return frames, reference


def test_sunday_growth_controls_thursday_and_ignores_monday():
    stable = _stablecoins()
    assert weekly_targets(stable)[date(2021, 1, 7)] is True
    sunday = pd.Timestamp("2021-01-03", tz="UTC")
    stable.loc[sunday, "USDT"] = stable.loc[pd.Timestamp("2020-12-06", tz="UTC"), "USDT"]
    assert weekly_targets(stable)[date(2021, 1, 7)] is False
    stable.loc[pd.Timestamp("2021-01-04", tz="UTC"), "USDT"] = 1e12
    assert weekly_targets(stable)[date(2021, 1, 7)] is False


def test_stablecoin_pin_and_exact_utc_day():
    rows = []
    for day in pd.date_range(FIRST, LAST, tz="UTC", freq="D"):
        rows.append({"date": str(int(day.timestamp())),
                     "totalCirculating": {"peggedUSD": 1000},
                     "totalCirculatingUSD": {"peggedUSD": 1000}})
    subset = [[int(row["date"]), 1000] for row in rows]
    digest = sha256(json.dumps(subset, separators=(",", ":"), ensure_ascii=False,
                               allow_nan=False).encode()).hexdigest()
    coin = (1, "USDT", digest)
    values, audit = parse_stablecoin(json.dumps(rows).encode(), coin)
    assert len(values) == 1857 and audit["selected_subset_sha256"] == digest
    rows[0]["totalCirculating"]["peggedUSD"] = 1001
    try:
        parse_stablecoin(json.dumps(rows).encode(), coin)
    except LiquiditySourceFailure:
        pass
    else:
        raise AssertionError("changed stablecoin history accepted")


def test_flat_five_year_spot_cost_and_latency():
    prices, terminal = _flat_prices()
    stable = _stablecoins()
    net = simulate(stable, prices, terminal)
    gross = simulate(stable, prices, terminal, slip=0, fee_rate=0)
    stress = simulate(stable, prices, terminal, slip=.002)
    late = simulate(stable, prices, terminal, latency_days=1)
    benchmark = simulate(stable, prices, terminal, benchmark=True)
    assert len(net.days) == 1826 and net.risk_on_entries == 1
    assert net.trades[0]["day"] == "2021-01-07"
    assert late.trades[0]["day"] == "2021-01-08"
    assert net.trades[-1]["day"] == "2026-01-01"
    assert math.isclose(gross.final, 10000., abs_tol=1e-7)
    assert net.final < gross.final and net.fees > 0
    assert benchmark.trades[0]["day"] == "2021-01-01"
    result = evaluate(net, gross, stress, late, benchmark, prices)
    assert result["monthly"]["months"] == 60
    assert result["monte_carlo"]["paths"] == 5000
    assert result["checks"]["independent_executable_oos"] is False
    assert result["all_gate_passed"] is False
    json.dumps(result, allow_nan=False)


def test_capacity_failure_is_not_silently_skipped():
    prices, terminal = _flat_prices(quote_volume=100.)
    try:
        simulate(_stablecoins(), prices, terminal)
    except ReplayFailure as exc:
        assert "quote-volume cap" in str(exc)
    else:
        raise AssertionError("over-capacity entry was accepted")


def test_spot_archive_microsecond_grid_and_terminal_first_open():
    for month in ("2025-01", "2026-01"):
        name = f"BTCUSDT-1d-{month}.zip"
        first = pd.Timestamp(f"{month}-01", tz="UTC")
        lines = []
        for i in range(31):
            stamp = int((first + pd.Timedelta(days=i)).timestamp()) * 1_000_000
            lines.append(f"{stamp},100,101,99,100,10,{stamp+86400000000-1},1000,10,5,500,0")
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr(name[:-4] + ".csv", "\n".join(lines))
        payload = buf.getvalue()
        digest = sha256(payload).hexdigest()
        row = {"month": month, "symbol": "BTCUSDT", "sha256": digest}
        actual = source._fetch
        try:
            source._fetch = lambda url, *, limit, params=None: (
                f"{digest}  {name}".encode() if url.endswith(".CHECKSUM") else payload)
            _, _, parsed, _ = source._load_month(row)
            if month == "2025-01":
                assert isinstance(parsed, pd.DataFrame) and len(parsed) == 31
                assert parsed.iloc[0]["open"] == 100.
            else:
                assert parsed == 100.
        finally:
            source._fetch = actual
