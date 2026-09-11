"""Pre-registered inverse-volatility blend of PPUT and BTC trend sleeves."""
import numpy as np
import pandas as pd

from cboe_options_benchmark import summarize_returns


def align_component_returns(pput_level, btc_simulation):
    pput_level = pput_level.sort_index()
    btc = btc_simulation.copy()
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc_equity = btc.set_index("open_time")["equity"].sort_index()
    common_dates = pput_level.index.intersection(btc_equity.index)
    pput = pput_level.reindex(common_dates).pct_change(fill_method=None)
    # Equity-to-equity change over Cboe dates compounds all intervening weekends.
    btc_return = btc_equity.reindex(common_dates).pct_change(fill_method=None)
    return pd.DataFrame({
        "PPUT": pput.fillna(0.0),
        "BTC_TSMOM30_DD": btc_return.fillna(0.0),
    }, index=common_dates)


def build_positions(component_returns, vol_window=60, rebalance_every=21):
    annualized_vol = component_returns.rolling(
        vol_window, min_periods=vol_window
    ).std(ddof=0) * np.sqrt(252.0)
    inverse_vol = 1.0 / annualized_vol.replace(0, np.nan)
    targets = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)
    mask = pd.Series(False, index=component_returns.index)
    mask.iloc[vol_window::rebalance_every] = True
    scheduled = targets.where(mask, axis=0).ffill().fillna(0.0)
    return scheduled.shift(1).fillna(0.0)


def run_base_blend(component_returns, transaction_cost=0.0005):
    positions = build_positions(component_returns)
    gross_return = (positions * component_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    net_return = gross_return - turnover * transaction_cost
    return net_return, positions, turnover


def run_scenario(
    component_returns, leverage, annual_financing=0.04,
    transaction_cost=0.0005
):
    base_return, base_positions, base_turnover = run_base_blend(
        component_returns, transaction_cost=transaction_cost
    )
    active_gross = base_positions.abs().sum(axis=1)
    financing = (
        (active_gross * leverage - 1.0).clip(lower=0.0)
        * annual_financing / 252.0
    )
    scenario_return = base_return * leverage - financing
    simulation, yearly, metrics = summarize_returns(scenario_return)
    simulation["gross_exposure"] = (active_gross * leverage).values
    simulation["turnover"] = (base_turnover * leverage).values
    simulation["financing_cost"] = financing.values
    simulation["equity"] = (1.0 + scenario_return).cumprod().values
    return simulation, base_positions, yearly, metrics
