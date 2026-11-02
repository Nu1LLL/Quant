"""Small deterministic diagonal-Gaussian HMM with causal filtering.

The implementation exists to make Task 3 auditable without a hidden dependency.
Trading code must call ``filter_probabilities``; smoothed/Viterbi states are not
provided on purpose.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HMMParameters:
    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    training_mean: np.ndarray
    training_std: np.ndarray
    stress_state: int
    log_likelihood: float
    iterations: int


def _logsumexp(values, axis=None):
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    total = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    if axis is None:
        return float(total.ravel()[0])
    return np.squeeze(total, axis=axis)


def _log_emissions(x, means, variances):
    difference = x[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.log(2.0 * np.pi * variances)[None, :, :]
        + difference * difference / variances[None, :, :]
    ).sum(axis=2)


def _forward_backward(log_emission, initial, transition):
    n, states = log_emission.shape
    log_initial = np.log(np.clip(initial, 1e-300, None))
    log_transition = np.log(np.clip(transition, 1e-300, None))
    alpha = np.empty((n, states))
    alpha[0] = log_initial + log_emission[0]
    for t in range(1, n):
        alpha[t] = log_emission[t] + _logsumexp(
            alpha[t - 1][:, None] + log_transition, axis=0
        )
    likelihood = _logsumexp(alpha[-1])
    beta = np.zeros((n, states))
    for t in range(n - 2, -1, -1):
        beta[t] = _logsumexp(
            log_transition + log_emission[t + 1][None, :] + beta[t + 1][None, :],
            axis=1,
        )
    log_gamma = alpha + beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    xi_sum = np.zeros((states, states))
    for t in range(n - 1):
        log_xi = (alpha[t][:, None] + log_transition
                  + log_emission[t + 1][None, :] + beta[t + 1][None, :])
        log_xi -= _logsumexp(log_xi)
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, likelihood


def fit_gaussian_hmm(values, states=2, max_iter=100, tolerance=1e-6,
                     variance_floor=1e-4) -> HMMParameters:
    """Fit an HMM on one training window; only 2 or 3 states are allowed."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 30 or not np.isfinite(x).all():
        raise ValueError("HMM training values must be a finite 2-D array with >=30 rows")
    if states not in (2, 3):
        raise ValueError("Only the preregistered 2 or 3 states are allowed")
    training_mean = x.mean(axis=0)
    training_std = x.std(axis=0)
    training_std = np.where(training_std > 1e-12, training_std, 1.0)
    z = (x - training_mean) / training_std

    energy = np.mean(z * z, axis=1)
    cuts = np.quantile(energy, np.arange(1, states) / states)
    labels = np.digitize(energy, cuts)
    means = np.vstack([
        z[labels == state].mean(axis=0) if np.any(labels == state)
        else z[state::states].mean(axis=0)
        for state in range(states)
    ])
    variances = np.vstack([
        z[labels == state].var(axis=0) if np.any(labels == state)
        else np.ones(z.shape[1])
        for state in range(states)
    ])
    variances = np.maximum(variances, variance_floor)
    transition = np.full((states, states), 1.0)
    for left, right in zip(labels, labels[1:]):
        transition[left, right] += 1.0
    transition /= transition.sum(axis=1, keepdims=True)
    initial = np.full(states, 1.0 / states)

    previous = -np.inf
    likelihood = -np.inf
    iteration = 0
    for iteration in range(1, max_iter + 1):
        emission = _log_emissions(z, means, variances)
        gamma, xi_sum, likelihood = _forward_backward(emission, initial, transition)
        weights = np.maximum(gamma.sum(axis=0), 1e-12)
        initial = np.maximum(gamma[0], 1e-8)
        initial /= initial.sum()
        transition = xi_sum + 1e-6
        transition /= transition.sum(axis=1, keepdims=True)
        means = (gamma.T @ z) / weights[:, None]
        difference = z[:, None, :] - means[None, :, :]
        variances = (gamma[:, :, None] * difference * difference).sum(axis=0)
        variances /= weights[:, None]
        variances = np.maximum(variances, variance_floor)
        if likelihood - previous < tolerance and likelihood >= previous:
            break
        previous = likelihood

    # Feature 0 is preregistered as SPY weekly return.  The state with the
    # largest conditional variance is the stress state; reorder for stability.
    order = np.argsort(variances[:, 0])
    initial = initial[order]
    means = means[order]
    variances = variances[order]
    transition = transition[np.ix_(order, order)]
    stress_state = int(np.argmax(variances[:, 0]))
    return HMMParameters(initial, transition, means, variances,
                         training_mean, training_std, stress_state,
                         float(likelihood), iteration)


def filter_probabilities(values, parameters: HMMParameters,
                         initial_probability=None) -> np.ndarray:
    """Forward-only probabilities; appending future rows cannot alter the past."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Filtering values must be a finite 2-D array")
    z = (x - parameters.training_mean) / parameters.training_std
    emission = _log_emissions(z, parameters.means, parameters.variances)
    probability = (parameters.initial.copy() if initial_probability is None
                   else np.asarray(initial_probability, dtype=float).copy())
    probability /= probability.sum()
    result = np.empty((len(z), len(probability)))
    for t in range(len(z)):
        if t:
            probability = probability @ parameters.transition
        log_probability = np.log(np.clip(probability, 1e-300, None)) + emission[t]
        log_probability -= _logsumexp(log_probability)
        probability = np.exp(log_probability)
        result[t] = probability
    return result
