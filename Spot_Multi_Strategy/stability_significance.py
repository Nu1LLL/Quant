"""Dependence-aware significance helpers for experimental alpha validation.

This module does not replace the historical PnL-concentration Gate. It supplies
an independently named diagnostic so old results remain reproducible.
"""
import math

import numpy as np
import pandas as pd


def circular_block_bootstrap_mean_pvalue(
    values, block_length=42, repetitions=4999, seed=1729
):
    """One-sided p-value for H0: mean <= 0 using a circular block bootstrap."""
    sample = np.asarray(pd.Series(values).dropna(), dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    if block_length < 1 or repetitions < 1:
        raise ValueError("block_length and repetitions must be positive")

    observed_mean = float(sample.mean())
    centered = sample - observed_mean
    n = len(centered)
    block_length = min(int(block_length), n)
    block_count = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)
    exceedances = 0

    completed = 0
    while completed < repetitions:
        chunk_size = min(128, repetitions - completed)
        starts = rng.integers(0, n, size=(chunk_size, block_count))
        indices = (starts[:, :, None] + offsets) % n
        resampled = centered[indices].reshape(chunk_size, -1)[:, :n]
        null_means = resampled.mean(axis=1)
        exceedances += int(np.count_nonzero(null_means >= observed_mean))
        completed += chunk_size

    return float((exceedances + 1) / (repetitions + 1))


def benjamini_hochberg(pvalues):
    """Return Benjamini-Hochberg adjusted q-values, preserving NaNs."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return adjusted

    valid = values[valid_positions]
    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    raw_adjusted = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adjusted_valid = np.empty(m, dtype=float)
    adjusted_valid[order] = monotone
    adjusted[valid_positions] = adjusted_valid
    return adjusted


def cross_sectional_ic_series(signal_matrix, forward_return_matrix):
    """Pearson IC at each timestamp, keeping the asset cross-section intact."""
    signals, forwards = signal_matrix.align(
        forward_return_matrix, join="inner", axis=0
    )
    signals, forwards = signals.align(forwards, join="inner", axis=1)
    records = []
    for signal_row, forward_row in zip(
        signals.to_numpy(dtype=float), forwards.to_numpy(dtype=float)
    ):
        valid = np.isfinite(signal_row) & np.isfinite(forward_row)
        if valid.sum() < 3:
            records.append(np.nan)
            continue
        left = signal_row[valid]
        right = forward_row[valid]
        if left.std(ddof=0) == 0 or right.std(ddof=0) == 0:
            records.append(np.nan)
        else:
            records.append(float(np.corrcoef(left, right)[0, 1]))
    return pd.Series(records, index=signals.index, name="cross_sectional_ic")
