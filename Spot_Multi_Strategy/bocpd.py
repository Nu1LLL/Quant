"""Causal Bayesian online changepoint detector for Task 3 experiment 2."""
from dataclasses import dataclass
from math import lgamma, pi

import numpy as np


@dataclass(frozen=True)
class StudentTPrior:
    mu: float
    kappa: float
    alpha: float
    beta: float


def prior_from_training(values) -> StudentTPrior:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all():
        raise ValueError("Training data must be a finite vector with at least 30 rows")
    variance = max(float(values.var(ddof=1)), 1e-8)
    return StudentTPrior(float(values.mean()), 1.0, 2.0, variance)


def _student_logpdf(value, mu, kappa, alpha, beta):
    degrees = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    gammaln = np.vectorize(lgamma, otypes=[float])
    return (gammaln((degrees + 1.0) / 2.0) - gammaln(degrees / 2.0)
            - 0.5 * np.log(degrees * pi * scale2)
            - (degrees + 1.0) / 2.0
            * np.log1p((value - mu) ** 2 / (degrees * scale2)))


def _update(value, mu, kappa, alpha, beta):
    next_kappa = kappa + 1.0
    next_mu = (kappa * mu + value) / next_kappa
    next_alpha = alpha + 0.5
    next_beta = beta + kappa * (value - mu) ** 2 / (2.0 * next_kappa)
    return next_mu, next_kappa, next_alpha, next_beta


def changepoint_probabilities(values, prior: StudentTPrior,
                              hazard=1 / 52, max_run_length=260) -> np.ndarray:
    """Return P(run length == 0) using only observations through each row."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Observations must be a finite vector")
    if not 0 < hazard < 1 or max_run_length < 2:
        raise ValueError("Invalid hazard or run-length cap")
    probability = np.array([1.0])
    mu = np.array([prior.mu])
    kappa = np.array([prior.kappa])
    alpha = np.array([prior.alpha])
    beta = np.array([prior.beta])
    result = np.empty(len(values))
    for t, value in enumerate(values):
        log_predictive = _student_logpdf(value, mu, kappa, alpha, beta)
        predictive = np.exp(log_predictive - np.max(log_predictive))
        prior_log_predictive = _student_logpdf(
            value, prior.mu, prior.kappa, prior.alpha, prior.beta
        )
        prior_predictive = float(np.exp(prior_log_predictive - np.max(log_predictive)))
        changepoint = probability.sum() * hazard * prior_predictive
        growth = probability * (1.0 - hazard) * predictive
        next_probability = np.r_[changepoint, growth]
        if len(next_probability) > max_run_length + 1:
            next_probability = next_probability[:max_run_length + 1]
        next_probability /= next_probability.sum()

        reset = _update(value, prior.mu, prior.kappa, prior.alpha, prior.beta)
        grown = _update(value, mu, kappa, alpha, beta)
        next_mu = np.r_[reset[0], grown[0]][:len(next_probability)]
        next_kappa = np.r_[reset[1], grown[1]][:len(next_probability)]
        next_alpha = np.r_[reset[2], grown[2]][:len(next_probability)]
        next_beta = np.r_[reset[3], grown[3]][:len(next_probability)]
        probability = next_probability
        mu, kappa, alpha, beta = next_mu, next_kappa, next_alpha, next_beta
        result[t] = probability[0]
    return result


def combine_probabilities(probability_matrix) -> np.ndarray:
    values = np.asarray(probability_matrix, dtype=float)
    if values.ndim != 2 or ((values < 0) | (values > 1)).any():
        raise ValueError("Probability matrix must be 2-D and within [0, 1]")
    return 1.0 - np.prod(1.0 - values, axis=1)
