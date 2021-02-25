"""
Main test module. For ease of testing each test makes use of parameterized fixtures defined in conftest.py. Fixtures are passed in as args to each test function which automatically generates a completed grid of parameter combinations for each test.

Tests are split up by model for ease of modular testing and to avoid creating uneccesary parameter combinations, thereby reducing the total number of tests.

For running tests in parallel `pip install pytest-xdist` and for nicer testing output `pip install pytest-sugar`.

With these two packages installed, you can run pytest locally using pytest -rs -n auto, to see skip messages at the end of the test session and visually confirm that only intended skipped tests are being skipped. To aid in this, all pytest.skip() messages end with 'OK' for intentionally skipped tests.
"""

from emotioncf import Mean, KNN, NNMF_mult, NNMF_sgd
import pytest
import pandas as pd
import numpy as np


def verify_fit(fit_kwargs):
    """Helper function to test fit call"""
    model = fit_kwargs.pop("model")
    model.fit(**fit_kwargs)
    assert model.is_fit
    assert model.predictions is not None
    return model.summary(), model


def verify_results(results, model, true_scores=None):
    """Helper function to test results object"""
    assert isinstance(results, pd.DataFrame)
    assert model.results is not None
    assert results.shape == (3, 7)
    long_results = model.summary(long_df=True, verbose=False)
    assert isinstance(long_results, pd.DataFrame)
    assert long_results.shape == (12, 4)
    if model.is_dense:
        assert not long_results.isnull().any().any()
    else:
        assert long_results.isnull().sum().sum() == 4

    # For regression testing our results should be relatively consistent, this just ensures that. We expect performances to be in the same ballpark (+/- 2), so ignore MSE because that fluctuates much more than RMSE, MAE, or corr
    if true_scores is not None:
        assert np.allclose(
            long_results.query("metric != 'mse'").score.to_numpy(), true_scores, atol=2
        )


def test_init(init, mask, n_mask_items):
    """Test model initialization and initial masking"""

    print(init.__class__.__name__)
    # Test that we calculate a mask
    if mask is not None or n_mask_items is not None:
        assert init.is_masked
        assert init.masked_data.isnull().any().any()

    # Test the mask is the right shape
    if n_mask_items is not None:
        total_items = init.data.shape[1]
        if isinstance(n_mask_items, (float, np.floating)):
            n_false_items = int(total_items * n_mask_items)
        else:
            n_false_items = n_mask_items
        calculated_n_false_items = init.masked_data.isnull().sum(1)[0]
        assert n_false_items == calculated_n_false_items

    # Test no accidental masking
    if mask is None and n_mask_items is None:
        assert not init.is_masked
        assert not init.masked_data.isnull().any().any()

        # Test fit failure when not masked
        with pytest.raises(ValueError):
            init.fit()


def test_mean(model, dilate_by_nsamples):
    """Test Mean model"""
    if not isinstance(model, Mean):
        pytest.skip("Skip non Mean - OK")
    results, model = verify_fit(locals())
    if model.n_mask_items == 0.5 and not model.is_mask_dilated:
        true_scores = np.array(
            [
                8.12277126e-01,
                9.75982274e00,
                1.75006327e01,
                5.61000920e-01,
                1.95196455e01,
                2.47496320e01,
                1.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )
    else:
        true_scores = None
    verify_results(results, model, true_scores)


def test_knn(model, dilate_by_nsamples, n_mask_items, k, metric):
    """Test KNN model"""
    if not isinstance(model, KNN):
        pytest.skip("Skip non KNN - OK")
    results, model = verify_fit(locals())
    if model.n_mask_items == 0.5 and not model.is_mask_dilated and k == 3:
        true_scores = np.array(
            [
                0.85812186,
                13.46414568,
                16.26052896,
                0.84304036,
                13.99007607,
                16.78521663,
                0.87251641,
                12.9382153,
                15.71833665,
            ]
        )
    else:
        true_scores = None
    verify_results(results, model, true_scores)


def test_nmf_mult(model, dilate_by_nsamples, n_mask_items, n_factors, n_iterations):
    """Test NNMF_mult model"""
    if not isinstance(model, NNMF_mult):
        pytest.skip("Skip non NNMF_mult - OK")
    results, model = verify_fit(locals())
    if (
        model.n_mask_items == 0.5
        and not model.is_mask_dilated
        and n_iterations == 100
        and n_factors is None
        and model.converged is True
    ):
        true_scores = np.array(
            [
                5.03636830e-01,
                1.89621113e01,
                3.39885854e01,
                -2.78597027e-02,
                3.69112079e01,
                4.80280757e01,
                9.97852743e-01,
                1.01301480e00,
                1.93696252e00,
            ]
        )
    else:
        true_scores = None
    verify_results(results, model, true_scores)


def test_nmf_sgd(model, dilate_by_nsamples, n_mask_items, n_factors, n_iterations):
    """Test NNMF_sgd model"""
    if not isinstance(model, NNMF_sgd):
        pytest.skip("Skip non NNMF_sgd - OK")
    results, model = verify_fit(locals())
    if (
        model.n_mask_items == 0.5
        and not model.is_mask_dilated
        and n_iterations == 100
        and n_factors is None
        and model.converged is True
    ):
        true_scores = np.array(
            [
                0.89757001,
                7.96887004,
                13.33854933,
                0.78658875,
                15.44602272,
                18.85269805,
                0.99977349,
                0.49171737,
                0.63997854,
            ]
        )
    else:
        true_scores = None
    verify_results(results, model, true_scores)
