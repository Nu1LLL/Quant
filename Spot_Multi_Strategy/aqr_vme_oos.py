"""AQR post-publication Value and Momentum Everywhere helpers."""
import pandas as pd

import aqr_tsmom_oos as monthly


def load_factor_csv(path):
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "VAL", "MOM"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing AQR VME columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame["date"].dt.to_period("M").duplicated().any():
        raise ValueError("Duplicate AQR VME month")
    return frame.set_index("date")


def equal_value_momentum(frame):
    frame = frame[["VAL", "MOM"]].dropna().astype(float)
    result = 0.5 * frame["VAL"] + 0.5 * frame["MOM"]
    result.name = "gross_equal_value_momentum"
    return result


def apply_scenario(gross_returns, leverage):
    return monthly.apply_scenario(
        gross_returns, leverage, annual_cost=0.03, annual_financing=0.04
    )
