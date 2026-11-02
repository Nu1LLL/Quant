"""Auditable HMM + RBF-kernel core for minute order-book signals."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import hmm_regime


REQUIRED_COLUMNS = (
    "best_bid", "best_ask", "bid_size_1", "ask_size_1",
    "bid_depth_5", "ask_depth_5", "bid_depth_20", "ask_depth_20",
    "buy_notional", "sell_notional", "bid_cancel_notional",
    "ask_cancel_notional",
)
V1_FEATURE_COLUMNS = (
    "return_1m", "return_5m", "return_15m", "spread_bps",
    "imbalance_l1", "imbalance_l5", "imbalance_l20",
    "microprice_deviation_bps", "trade_imbalance", "cancel_imbalance",
    "log_depth", "volume_surprise", "realized_vol_30m",
)
OFI_FEATURE_COLUMNS = ("ofi_l1", "ofi_l1_depth_scaled", "ofi_5m_depth_scaled")
FEATURE_COLUMNS = V1_FEATURE_COLUMNS + OFI_FEATURE_COLUMNS


@dataclass(frozen=True)
class KernelModel:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    random_weights: np.ndarray
    random_bias: np.ndarray
    return_beta: np.ndarray
    class_beta: np.ndarray
    residual_variance: float


@dataclass(frozen=True)
class BlackBoxModel:
    hmm: hmm_regime.HMMParameters
    kernel: KernelModel
    hmm_feature_columns: tuple
    kernel_feature_columns: tuple
    final_hmm_probability: np.ndarray
    model_version: str


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(frame).copy().sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("Microstructure frame needs a unique DatetimeIndex")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing order-book columns: {missing}")
    values = frame[list(REQUIRED_COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Order-book values must be finite")
    if (frame["best_bid"] <= 0).any() or (frame["best_ask"] <= frame["best_bid"]).any():
        raise ValueError("Crossed, locked or non-positive quotes are rejected")
    nonnegative = [column for column in REQUIRED_COLUMNS if column not in ("best_bid", "best_ask")]
    if (frame[nonnegative] < 0).any().any():
        raise ValueError("Sizes and notionals must be non-negative")
    return frame


def _imbalance(left, right):
    total = left + right
    return (left - right).div(total.replace(0.0, np.nan))


def order_flow_imbalance(frame: pd.DataFrame) -> pd.DataFrame:
    """Cont-Kukanov-Stoikov L1 OFI and causal depth-scaled variants."""
    frame = validate_frame(frame)
    previous_bid = frame["best_bid"].shift(1)
    previous_ask = frame["best_ask"].shift(1)
    previous_bid_size = frame["bid_size_1"].shift(1)
    previous_ask_size = frame["ask_size_1"].shift(1)
    ofi = (
        frame["bid_size_1"].where(frame["best_bid"].ge(previous_bid), 0.0)
        - previous_bid_size.where(frame["best_bid"].le(previous_bid), 0.0)
        - frame["ask_size_1"].where(frame["best_ask"].le(previous_ask), 0.0)
        + previous_ask_size.where(frame["best_ask"].ge(previous_ask), 0.0)
    )
    total_depth = frame["bid_size_1"] + frame["ask_size_1"]
    adjacent_mean_depth = (total_depth + total_depth.shift(1)) / 2
    return pd.DataFrame({
        "ofi_l1": ofi,
        "ofi_l1_depth_scaled": ofi.div(adjacent_mean_depth.replace(0.0, np.nan)),
        "ofi_5m_depth_scaled": ofi.rolling(5, min_periods=5).sum().div(
            total_depth.rolling(5, min_periods=5).mean().replace(0.0, np.nan)
        ),
    }, index=frame.index)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_frame(frame)
    mid = (frame["best_bid"] + frame["best_ask"]) / 2.0
    log_mid = np.log(mid)
    l1_total = frame["bid_size_1"] + frame["ask_size_1"]
    microprice = (
        frame["best_ask"] * frame["bid_size_1"]
        + frame["best_bid"] * frame["ask_size_1"]
    ).div(l1_total.replace(0.0, np.nan))
    total_notional = frame["buy_notional"] + frame["sell_notional"]
    log_volume = np.log1p(total_notional)
    prior_volume_mean = log_volume.shift(1).rolling(60, min_periods=30).mean()
    prior_volume_std = log_volume.shift(1).rolling(60, min_periods=30).std(ddof=1)
    return_1m = log_mid.diff()
    features = pd.DataFrame({
        "return_1m": return_1m,
        "return_5m": log_mid.diff(5),
        "return_15m": log_mid.diff(15),
        "spread_bps": (frame["best_ask"] - frame["best_bid"]) / mid * 1e4,
        "imbalance_l1": _imbalance(frame["bid_size_1"], frame["ask_size_1"]),
        "imbalance_l5": _imbalance(frame["bid_depth_5"], frame["ask_depth_5"]),
        "imbalance_l20": _imbalance(frame["bid_depth_20"], frame["ask_depth_20"]),
        "microprice_deviation_bps": (microprice - mid) / mid * 1e4,
        "trade_imbalance": _imbalance(frame["buy_notional"], frame["sell_notional"]),
        "cancel_imbalance": _imbalance(
            frame["ask_cancel_notional"], frame["bid_cancel_notional"]
        ),
        "log_depth": np.log1p(frame["bid_depth_20"] + frame["ask_depth_20"]),
        "volume_surprise": (log_volume - prior_volume_mean).div(
            prior_volume_std.replace(0.0, np.nan)
        ),
        "realized_vol_30m": return_1m.rolling(30, min_periods=30).std(ddof=0),
    })
    features = features.join(order_flow_imbalance(frame))
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def _rff_transform(values, mean, std, weights, bias):
    z = np.clip((values - mean) / std, -6.0, 6.0)
    return np.sqrt(2.0 / weights.shape[1]) * np.cos(z @ weights + bias)


def fit_kernel_model(features, next_returns, components=128, ridge=10.0,
                     seed=20260916) -> KernelModel:
    x = np.asarray(features, dtype=float)
    y = np.asarray(next_returns, dtype=float)
    if x.ndim != 2 or len(x) < 200 or y.shape != (len(x),):
        raise ValueError("Kernel training requires at least 200 aligned rows")
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=1)
    std = np.where(std > 1e-12, std, 1.0)
    rng = np.random.default_rng(seed)
    gamma = 1.0 / x.shape[1]
    weights = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(x.shape[1], components))
    bias = rng.uniform(0.0, 2.0 * np.pi, size=components)
    phi = _rff_transform(x, mean, std, weights, bias)
    design = np.column_stack([np.ones(len(phi)), phi])
    penalty = np.diag(np.r_[0.0, np.full(components, ridge)])
    return_beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    target = y.gt(0).astype(float) if isinstance(y, pd.Series) else (y > 0).astype(float)
    class_beta = np.zeros(design.shape[1])
    for _ in range(50):
        linear = np.clip(design @ class_beta, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(probability * (1 - probability), 1e-8, None)
        gradient = design.T @ (probability - target) + penalty @ class_beta
        hessian = design.T @ (design * weight[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        class_beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    residual = y - design @ return_beta
    return KernelModel(mean, std, weights, bias, return_beta, class_beta,
                       float(max(residual.var(ddof=1), 1e-12)))


def predict_kernel(features, model: KernelModel) -> tuple[np.ndarray, np.ndarray]:
    phi = _rff_transform(np.asarray(features, dtype=float), model.feature_mean,
                         model.feature_std, model.random_weights, model.random_bias)
    design = np.column_stack([np.ones(len(phi)), phi])
    expected = design @ model.return_beta
    score = np.clip(design @ model.class_beta, -35.0, 35.0)
    probability = 1.0 / (1.0 + np.exp(-score))
    return expected, probability


def train_and_predict(train_frame: pd.DataFrame, test_frame: pd.DataFrame,
                      model_version="microstructure-v2", feature_version="v2"
                      ) -> tuple[BlackBoxModel, pd.DataFrame]:
    combined = pd.concat([validate_frame(train_frame), validate_frame(test_frame)]).sort_index()
    features = build_features(combined)
    train_end = train_frame.index.max()
    # Exclude the last training timestamp because its t+1 label would be the
    # first quote in the held-out test period.
    train_features = features.loc[features.index < train_end]
    test_features = features.loc[features.index > train_end]
    next_return = np.log(
        ((combined["best_bid"] + combined["best_ask"]) / 2.0).shift(-1)
        / ((combined["best_bid"] + combined["best_ask"]) / 2.0)
    ).reindex(train_features.index)
    valid = next_return.notna()
    train_features, next_return = train_features.loc[valid], next_return.loc[valid]
    hmm_columns = ("return_1m", "spread_bps", "log_depth", "realized_vol_30m")
    hmm = hmm_regime.fit_gaussian_hmm(train_features[list(hmm_columns)].to_numpy(), states=2)
    train_probability = hmm_regime.filter_probabilities(
        train_features[list(hmm_columns)].to_numpy(), hmm
    )
    initial = train_probability[-1] @ hmm.transition
    test_probability = hmm_regime.filter_probabilities(
        test_features[list(hmm_columns)].to_numpy(), hmm,
        initial_probability=initial,
    )
    if feature_version not in {"v1", "v2"}:
        raise ValueError("feature_version must be v1 or v2")
    base_columns = V1_FEATURE_COLUMNS if feature_version == "v1" else FEATURE_COLUMNS
    kernel_columns = tuple(base_columns) + ("hmm_stress_probability",)
    train_kernel = train_features.copy()
    train_kernel["hmm_stress_probability"] = train_probability[:, hmm.stress_state]
    test_kernel = test_features.copy()
    test_kernel["hmm_stress_probability"] = test_probability[:, hmm.stress_state]
    kernel = fit_kernel_model(train_kernel[list(kernel_columns)], next_return)
    expected, probability = predict_kernel(test_kernel[list(kernel_columns)], kernel)
    prediction = pd.DataFrame({
        "expected_return": expected,
        "probability_up": probability,
        "hmm_stress_probability": test_probability[:, hmm.stress_state],
    }, index=test_kernel.index)
    model = BlackBoxModel(hmm, kernel, hmm_columns, kernel_columns,
                          train_probability[-1], model_version)
    return model, prediction


def generate_orders(prediction: pd.DataFrame, market_frame: pd.DataFrame,
                    model: BlackBoxModel, venue: str, symbol: str,
                    fee_bps_each_side=2.0, impact_bps=1.0,
                    explicit_cost_bps=None,
                    probability_buy=0.58, probability_sell=0.42,
                    kelly_fraction=0.25, position_cap=0.25) -> pd.DataFrame:
    frame = validate_frame(market_frame).reindex(prediction.index)
    spread_bps = (frame["best_ask"] - frame["best_bid"]) / (
        (frame["best_ask"] + frame["best_bid"]) / 2.0
    ) * 1e4
    if explicit_cost_bps is None:
        cost_bps = spread_bps + 2 * fee_bps_each_side + impact_bps
    else:
        cost_bps = pd.Series(
            explicit_cost_bps, index=getattr(explicit_cost_bps, "index", None),
            dtype=float,
        ).reindex(prediction.index)
        if cost_bps.isna().any() or (cost_bps < spread_bps).any():
            raise ValueError("Explicit cost gate must be finite and include quoted spread")
    expected_bps = prediction["expected_return"] * 1e4
    action = pd.Series("FLAT", index=prediction.index)
    buy = prediction["probability_up"].ge(probability_buy) & expected_bps.gt(cost_bps)
    sell = prediction["probability_up"].le(probability_sell) & expected_bps.lt(-cost_bps)
    action.loc[buy], action.loc[sell] = "BUY", "SELL"
    raw_kelly = kelly_fraction * prediction["expected_return"] / model.kernel.residual_variance
    size = raw_kelly.clip(-position_cap, position_cap)
    size = size.where(action.ne("FLAT"), 0.0)
    size *= 1.0 - 0.5 * prediction["hmm_stress_probability"]
    return pd.DataFrame({
        "venue": venue, "symbol": symbol, "action": action,
        "probability_up": prediction["probability_up"],
        "expected_return_bps": expected_bps, "cost_gate_bps": cost_bps,
        "hmm_stress_probability": prediction["hmm_stress_probability"],
        "kelly_position": size, "valid_for_seconds": 60,
        "model_version": model.model_version,
    }, index=prediction.index)


def alpha_health(net_returns: pd.Series, minimum_observations=200,
                 window=1000) -> dict:
    values = pd.Series(net_returns).dropna().iloc[-window:]
    if len(values) < minimum_observations:
        return {"status": "SHADOW", "observations": len(values),
                "mean": np.nan, "lower_confidence": np.nan}
    mean, std = float(values.mean()), float(values.std(ddof=1))
    lower = mean - 1.96 * std / np.sqrt(len(values))
    return {"status": "ACTIVE" if lower > 0 else "DISABLED",
            "observations": len(values), "mean": mean,
            "lower_confidence": float(lower)}


def point_in_time_universe(snapshot: pd.DataFrame,
                           market_thresholds: dict) -> pd.DataFrame:
    required = {"market", "symbol", "notional_30d", "spread_bps_30d",
                "depth_usd_30d", "history_days", "data_complete", "shortable"}
    if not required.issubset(snapshot.columns):
        raise ValueError(f"Universe snapshot missing {sorted(required - set(snapshot.columns))}")
    keep = []
    for _, row in snapshot.iterrows():
        threshold = market_thresholds.get(row["market"])
        if threshold is None:
            keep.append(False)
            continue
        keep.append(
            row["notional_30d"] >= threshold["min_notional"]
            and row["spread_bps_30d"] <= threshold["max_spread_bps"]
            and row["depth_usd_30d"] >= threshold["min_depth"]
            and row["history_days"] >= threshold.get("min_history_days", 90)
            and bool(row["data_complete"]) and bool(row["shortable"])
        )
    return snapshot.loc[keep].copy().reset_index(drop=True)
