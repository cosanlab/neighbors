"""
Main test module. For ease of testing each test makes use of parameterized fixtures defined in conftest.py. Fixtures are passed in as args to each test function which automatically generates a completed grid of parameter combinations for each test.

Tests are split up by model for ease of modular testing and to avoid creating uneccesary parameter combinations, thereby reducing the total number of tests.

For running tests in parallel `pip install pytest-xdist` and for nicer testing output `pip install pytest-sugar`.

With these two packages installed, you can run pytest locally using pytest -rs -n auto, to see skip messages at the end of the test session and visually confirm that only intended skipped tests are being skipped. To aid in this, all pytest.skip() messages end with 'OK' for intentionally skipped tests.
"""

from emotioncf.models import Mean, KNN, NNMF_mult, NNMF_sgd
import pytest


# TODO: Extend this by returning the model so we can pipe that into subsequent helpers
def fit_model(fit_kwargs):
    """Helper function for test_*_fit tests. Fit a model with its respective kwargs and ensure that the fit was successful"""
    model = fit_kwargs.pop("model")
    model.fit(**fit_kwargs)
    assert model.is_fit
    assert model.predictions is not None


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
    fit_model(locals())


def test_knn(model, dilate_by_nsamples, n_mask_items, k, metric):
    """Test KNN model"""
    if not isinstance(model, KNN):
        pytest.skip("Skip non KNN - OK")
    fit_model(locals())


def test_nmf_mult(model, dilate_by_nsamples, n_mask_items, n_factors, n_iterations):
    """Test NNMF_mult model"""
    if not isinstance(model, NNMF_mult):
        pytest.skip("Skip non NNMF_mult - OK")
    fit_model(locals())


def test_nmf_sgd(model, dilate_by_nsamples, n_mask_items, n_factors, n_iterations):
    """Test NNMF_sgd model"""
    if not isinstance(model, NNMF_sgd):
        pytest.skip("Skip non NNMF_sgd - OK")
    fit_model(locals())
