"""Monthly SPY/BTAL tactical sleeve blended with fixed BTC trend."""
import pandas as pd


def build_tactical_sleeve(spy_level, btal_level, transaction_cost=0.0005):
    levels = pd.concat(
        [spy_level.rename("SPY"), btal_level.rename("BTAL")],
        axis=1, join="inner"
    ).dropna()
    asset_returns = levels.pct_change(fill_method=None).fillna(0.0)
    moving_average = levels["SPY"].rolling(200, min_periods=200).mean()
    valid = moving_average.notna()
    target_spy = (levels["SPY"] > moving_average).astype(float).where(valid)
    targets = pd.DataFrame({
        "SPY": target_spy,
        "BTAL": (1.0 - target_spy).where(valid),
    }, index=levels.index)
    month_keys = pd.Series(
        levels.index.year * 100 + levels.index.month, index=levels.index
    )
    month_end = month_keys.ne(month_keys.shift(-1))
    scheduled = targets.where(month_end, axis=0).ffill().fillna(0.0)
    positions = scheduled.shift(1).fillna(0.0)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    net_return = (
        (positions * asset_returns).sum(axis=1)
        - turnover * transaction_cost
    )
    return net_return.rename("TACTICAL_ANTIBETA"), positions, turnover


def align_with_btc(tactical_return, btc_simulation):
    btc = btc_simulation.copy()
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc_equity = btc.set_index("open_time")["equity"].sort_index()
    common = tactical_return.index.intersection(btc_equity.index)
    btc_return = btc_equity.reindex(common).pct_change(fill_method=None)
    return pd.DataFrame({
        "TACTICAL_ANTIBETA": tactical_return.reindex(common).fillna(0.0),
        "BTC_TSMOM30_DD": btc_return.fillna(0.0),
    }, index=common)
