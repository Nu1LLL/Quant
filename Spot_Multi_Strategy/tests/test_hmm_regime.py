import numpy as np

import hmm_regime as hmm


def sample(seed=1):
    rng = np.random.default_rng(seed)
    calm = rng.normal([0.2, -0.3, -0.2], [0.4, 0.3, 0.3], size=(100, 3))
    stress = rng.normal([-0.4, 1.2, 1.0], [1.8, 0.7, 0.8], size=(80, 3))
    return np.vstack([calm, stress, calm])


def test_fit_identifies_high_first_feature_variance_as_stress():
    parameters = hmm.fit_gaussian_hmm(sample(), states=2)
    assert parameters.stress_state == int(np.argmax(parameters.variances[:, 0]))
    assert np.allclose(parameters.transition.sum(axis=1), 1.0)
    assert np.isfinite(parameters.log_likelihood)


def test_filter_probabilities_are_normalized():
    values = sample()
    parameters = hmm.fit_gaussian_hmm(values[:180])
    probabilities = hmm.filter_probabilities(values[180:], parameters)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_appending_future_cannot_rewrite_filtered_history():
    values = sample()
    parameters = hmm.fit_gaussian_hmm(values[:180])
    prefix = hmm.filter_probabilities(values[180:230], parameters)
    extended = hmm.filter_probabilities(values[180:], parameters)
    np.testing.assert_allclose(prefix, extended[:len(prefix)])


def test_three_states_allowed_but_four_rejected():
    parameters = hmm.fit_gaussian_hmm(sample(), states=3, max_iter=10)
    assert len(parameters.initial) == 3
    try:
        hmm.fit_gaussian_hmm(sample(), states=4)
    except ValueError as error:
        assert "2 or 3" in str(error)
    else:
        raise AssertionError("Unregistered state count was accepted")
