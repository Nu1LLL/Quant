"""AQR post-publication global Quality Minus Junk helpers."""
import pandas as pd

import aqr_tsmom_oos as monthly


def load_factor_csv(path):
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "QMJ_GLOBAL"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing AQR QMJ columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame["date"].dt.to_period("M").duplicated().any():
        raise ValueError("Duplicate global QMJ month")
    return frame.set_index("date")


def apply_scenario(raw_returns, leverage):
    return monthly.apply_scenario(
        raw_returns, leverage, annual_cost=0.03, annual_financing=0.04
    )
