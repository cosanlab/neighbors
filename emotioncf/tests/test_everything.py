"""
Main test module. For ease of testing each test makes use of parameterized fixtures defined in conftest.py. Fixtures are passed in as args to each test function which automatically generates a completed grid of parameter combinations for each test.

Tests are split up by model for ease of modular testing and to avoid creating uneccesary parameter combinations, thereby reducing the total number of tests.

For running tests in parallel `pip install pytest-xdist` and for nicer testing output `pip install pytest-sugar`.

With these two packages installed, you can run pytest locally using pytest -rs -n auto, to see skip messages at the end of the test session and visually confirm that only intended skipped tests are being skipped. To aid in this, all pytest.skip() messages end with 'OK' for intentionally skipped tests.
"""

from emotioncf import Mean, KNN, NNMF_mult, NNMF_sgd, Results
import pytest
import pandas as pd


def verify_fit(fit_kwargs):
    """Helper function to test fit call"""
    model = fit_kwargs.pop("model")
    model.fit(**fit_kwargs)
    assert model.is_fit
    assert model.predictions is not None
    return model.summary(), model


def verify_results(results, model):
    """Helper function to test results object"""
    assert isinstance(results, Results)
    assert isinstance(results.predictions, pd.DataFrame)
    assert results.predictions.shape[0] == (
        model.data.shape[0] * model.data.shape[1] * 2
    )
    if model.is_dense:
        expected_scores = 3
    else:
        expected_scores = 1
    assert len(results.rmse.keys()) == expected_scores
    assert len(results.mse.keys()) == expected_scores
    assert len(results.mae.keys()) == expected_scores
    assert len(results.correlation.keys()) == expected_scores


def test_init(init, mask, n_mask_items):
    """Test model initialization and initial masking"""

    print(init.__class__.__name__)
    if mask is not None or n_mask_items is not None:
        assert init.is_masked
        assert init.masked_data.isnull().any().any()
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
    verify_results(results, model)


def test_knn(model, dilate_by_nsamples, n_mask_items, k, metric):
    """Test KNN model"""
    if not isinstance(model, KNN):
        pytest.skip("Skip non KNN - OK")
    results, model = verify_fit(locals())
    verify_results(results, model)


def test_nmf_mult(model, dilate_by_nsamples, n_mask_items, n_factors, n_iterations):
    """Test NNMF_mult model"""
    if not isinstance(model, NNMF_mult):
        pytest.skip("Skip non NNMF_mult - OK")
    results, model = verify_fit(locals())
    verify_results(results, model)


def test_nmf_sgd(model, dilate_by_nsamples, n_mask_items, n_factors, n_iterations):
    """Test NNMF_sgd model"""
    if not isinstance(model, NNMF_sgd):
        pytest.skip("Skip non NNMF_sgd - OK")
    results, model = verify_fit(locals())
    verify_results(results, model)
