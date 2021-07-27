"""
Modular testing for core algorithms. For ease of testing each test makes use of parameterized fixtures defined in conftest.py. Fixtures are passed in as args to each test function which automatically generates a complete grid of parameter combinations for each test.

Tests are split up by model for ease of modular testing (i.e. using pytest -k 'test_name') and to avoid creating uneccesary parameter combinations, thereby reducing the total number of tests.

For running tests in parallel `pip install pytest-xdist` and for nicer testing output `pip install pytest-sugar`.

Then you can run pytest locally using `pytest -rs -n auto`, to see skip messages at the end of the test session and visually confirm that only intended skipped tests are being skipped. To aid in this, all pytest.skip() messages end with 'OK' for intentionally skipped tests.
"""

from neighbors import Mean, KNN, NNMF_mult, NNMF_sgd, Base
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    assert model.overall_results is not None
    assert model.user_results is not None
    assert model.user_results.shape == (model.data.shape[0], 4 * 2)
    # 4 metrics, 2 datasets (default .summary returns observed and missing only), 2 "groups" (all, subject)
    assert results.shape == (4 * 2 * 2, 5)
    if model.is_dense:
        assert not results.isnull().any().any()
    else:
        assert results.isnull().sum().sum() == 8

    # For regression testing our results should be relatively consistent, this just ensures that. We expect performances to be in the same ballpark (+/- 2), so ignore MSE because that fluctuates much more than RMSE, MAE, or corr
    if true_scores is not None:
        assert np.allclose(
            results.query("metric != 'mse' and group == 'all'").score.to_numpy(),
            true_scores,
            atol=2,
        )


def verify_transform(model):
    out = model.transform()
    # Handle edge case where masking leads to an entire column of NaNs i.e. no ratings at all for a single item
    if not len(
        model.masked_data.columns[model.masked_data.isnull().all()].tolist()
    ) and not len(model.predictions.columns[model.predictions.isnull().all()].tolist()):
        # Aside from that edge case there should be no missing values in the output of .transform()
        assert not out.isnull().any().any()
    else:
        pytest.skip("Skip masking edgecase - OK")

    out = model.transform(return_only_predictions=True)
    assert out.equals(model.predictions)


def verify_plotting(model):
    for dataset in ["full", "observed", "missing"]:
        out = model.plot_predictions(dataset=dataset)
        if dataset != "missing":
            assert isinstance(out, tuple)
        plt.close("all")


def test_downsample(simulate_wide_data):
    n_users, n_items = simulate_wide_data.shape
    cf = Base(simulate_wide_data)
    assert cf.data.shape == (n_users, n_items)

    # Test sampling_freq has no effect if target_type = 'samples'
    sampling_freq, target = 10, 2
    expected_items = int(n_items * (1 / target))
    cf.downsample(sampling_freq=sampling_freq, n_samples=target, target_type="samples")
    assert cf.data.shape == (n_users, expected_items)

    # Test each target_type
    sampling_freq, target = 10, 2
    cf = Base(simulate_wide_data)
    expected_items = int(n_items * (1 / (target * sampling_freq)))
    cf.downsample(sampling_freq=sampling_freq, n_samples=target, target_type="seconds")
    assert cf.data.shape == (n_users, expected_items)

    sampling_freq, target = 10, 5
    cf = Base(simulate_wide_data)
    expected_items = int(n_items * (1 / (sampling_freq / target)))
    cf.downsample(sampling_freq=sampling_freq, n_samples=target, target_type="hz")
    assert cf.data.shape == (n_users, expected_items)

    # Make sure downsampling affects fitted model artifacts
    cf = Mean(simulate_wide_data)
    cf.create_masked_data(n_mask_items=0.5)
    cf.fit()
    cf.downsample(sampling_freq=sampling_freq, n_samples=target, target_type="hz")
    assert cf.data.shape == (n_users, expected_items)
    assert cf.mask.shape == (n_users, expected_items)
    assert cf.masked_data.shape == (n_users, expected_items)
    assert cf.predictions.shape == (n_users, expected_items)

    # Make sure downsampling affects fitted model artifactsa including dilation
    cf = Mean(simulate_wide_data)
    cf.create_masked_data(n_mask_items=0.5)
    cf.fit(dilate_by_nsamples=5)
    cf.downsample(sampling_freq=sampling_freq, n_samples=target, target_type="hz")
    assert cf.data.shape == (n_users, expected_items)
    assert cf.mask.shape == (n_users, expected_items)
    assert cf.dilated_mask.shape == (n_users, expected_items)
    assert cf.masked_data.shape == (n_users, expected_items)
    assert cf.predictions.shape == (n_users, expected_items)


def test_init_and_dilate(init, mask, n_mask_items):
    """Test model initialization, initial masking, and dilation"""

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
    if mask is not None or n_mask_items is not None:
        # Test dilation
        n_masked = init.masked_data.isnull().sum().sum()
        init.dilate_mask(n_samples=5)
        assert init.dilated_mask is not None
        assert init.is_mask_dilated is True
        # More values when we dilate the mask
        assert init.dilated_mask.sum().sum() > init.mask.sum().sum()
        # Fewer masked values after we dilate the mask
        assert n_masked > init.masked_data.isnull().sum().sum()


def test_mean(model, dilate_by_nsamples, n_mask_items):
    """Test Mean model"""
    if not isinstance(model, Mean):
        pytest.skip("Skip non Mean - OK")
    results, model = verify_fit(locals())
    if model.n_mask_items == 0.5 and not model.is_mask_dilated:
        true_scores = np.array(
            [
                0.55695942,
                19.73822567,
                25.1216878,
                0.61205836,
                18.62724324,
                23.57214148,
            ]
        )
    else:
        true_scores = None
    verify_results(results, model, true_scores)
    verify_plotting(model)
    verify_transform(model)


def test_knn(model, dilate_by_nsamples, n_mask_items, k, metric):
    """Test KNN model"""
    if not isinstance(model, KNN):
        pytest.skip("Skip non KNN - OK")
    results, model = verify_fit(locals())
    if (
        model.n_mask_items == 0.5
        and metric == "correlation"
        and not model.is_mask_dilated
        and k == 3
    ):
        true_scores = np.array(
            [
                0.84244861,
                14.40186486,
                17.49846264,
                0.86440771,
                13.20914251,
                15.95942564,
            ]
        )
    else:
        true_scores = None
    verify_results(results, model, true_scores)
    verify_plotting(model)
    verify_transform(model)


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
    verify_plotting(model)
    verify_transform(model)
    # Smoke test for plotting learning curves
    model.plot_learning()
    plt.close("all")


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
    verify_plotting(model)
    verify_transform(model)
    # Smoke test for plotting learning curves
    model.plot_learning()
    plt.close("all")
