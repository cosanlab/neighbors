"""
Modular testing for core algorithms. For ease of testing each test makes use of parameterized fixtures defined in conftest.py. Fixtures are passed in as args to each test function which automatically generates a complete grid of parameter combinations for each test.

Tests are split up by model for ease of modular testing (i.e. using pytest -k 'test_name') and to avoid creating uneccesary parameter combinations, thereby reducing the total number of tests.

For running tests in parallel `pip install pytest-xdist` and for nicer testing output `pip install pytest-sugar`.

Then you can run pytest locally using `pytest -rs -n auto`, to see skip messages at the end of the test session and visually confirm that only intended skipped tests are being skipped. To aid in this, all pytest.skip() messages end with 'OK' for intentionally skipped tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from neighbors import KNN, Base, Mean, NNMF_mult, NNMF_sgd


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
    cf = Mean(simulate_wide_data, random_state=2)
    cf.create_masked_data(n_mask_items=0.5)
    cf.fit()
    cf.downsample(sampling_freq=sampling_freq, n_samples=target, target_type="hz")
    assert cf.data.shape == (n_users, expected_items)
    assert cf.mask.shape == (n_users, expected_items)
    assert cf.masked_data.shape == (n_users, expected_items)
    assert cf.predictions.shape == (n_users, expected_items)

    # Make sure downsampling affects fitted model artifactsa including dilation
    cf = Mean(simulate_wide_data, random_state=2)
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
        calculated_n_false_items = init.masked_data.isnull().sum(1).iloc[0]
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


def test_dilation_centers_odd_width_kernel_on_observation():
    ratings = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, 50, np.nan, np.nan, np.nan, np.nan]
    )

    dilated = Base._conv_ts_mean_overlap(ratings, n_samples=5)

    expected = np.array([np.nan, np.nan, 50, 50, 50, 50, 50, np.nan, np.nan])
    np.testing.assert_equal(dilated, expected)


def test_dilation_uses_documented_half_sample_alignment_for_even_width_kernel():
    ratings = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, 50, np.nan, np.nan, np.nan, np.nan]
    )

    dilated = Base._conv_ts_mean_overlap(ratings, n_samples=4)

    expected = np.array([np.nan, np.nan, np.nan, 50, 50, 50, 50, np.nan, np.nan])
    np.testing.assert_equal(dilated, expected)


def test_dilation_averages_overlapping_centered_kernels():
    ratings = pd.Series(
        [np.nan, np.nan, np.nan, 20, np.nan, 80, np.nan, np.nan, np.nan]
    )

    dilated = Base._conv_ts_mean_overlap(ratings, n_samples=5)

    expected = np.array([np.nan, 20, 20, 50, 50, 50, 80, 80, np.nan])
    np.testing.assert_equal(dilated, expected)


def test_dilation_does_not_mutate_input_ratings():
    ratings = pd.Series([np.nan, 25, np.nan])
    original = ratings.copy()

    Base._conv_ts_mean_overlap(ratings, n_samples=3)

    pd.testing.assert_series_equal(ratings, original)


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
    ):
        true_scores = np.array(
            [
                0.6645,
                18.7424,
                24.1340,
                0.9973,
                1.6227,
                2.3367,
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


def test_nmf_mult_ignores_missing_entries(simulate_wide_data):
    """Missing entries must not act as observed zeros during multiplicative updating. Before the update denominators were masked, held-out predictions were dragged toward zero (mean ~14 vs a true mean ~38 on this fixture)"""
    model = NNMF_mult(simulate_wide_data, n_mask_items=0.5, random_state=2)
    model.fit(n_iterations=500, n_factors=10)
    missing = ~model.mask.to_numpy()
    truth = simulate_wide_data.to_numpy()[missing]
    pred = model.predictions.to_numpy()[missing]
    # Held-out predictions should be centered near the held-out truth, not near zero
    assert np.isclose(pred.mean(), truth.mean(), rtol=0.25)
    heldout_rmse = np.sqrt(np.mean((truth - pred) ** 2))
    assert heldout_rmse < 30
    # Training error is computed over observed entries only and should be lower than held-out error
    assert model.score(metric="rmse", dataset="observed", by_user=False) < heldout_rmse


def test_nmf_predictions_clipped_to_observed_range(simulate_wide_data):
    """By default NNMF predictions are clipped to the observed rating range, since unconstrained bias terms can otherwise push predictions outside it (issue #47)"""
    for cls in [NNMF_mult, NNMF_sgd]:
        clipped = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        clipped.fit(n_iterations=50)
        unclipped = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        unclipped.fit(n_iterations=50, clip_predictions=False)
        vmin = unclipped.masked_data.min().min()
        vmax = unclipped.masked_data.max().max()
        assert (clipped.predictions >= vmin).all().all()
        assert (clipped.predictions <= vmax).all().all()
        # Clipping should be the only difference between the two fits
        np.testing.assert_allclose(
            clipped.predictions.to_numpy(),
            unclipped.predictions.clip(vmin, vmax).to_numpy(),
        )


def test_nmf_clip_bounds_ignore_dilation(simulate_wide_data):
    """Clip bounds come from the raw observed ratings, not the dilated training data. Dilation replaces masked_data with a moving average whose range is narrower than the observed ratings, so clipping to it would truncate legitimate predictions"""
    for cls in [NNMF_mult, NNMF_sgd]:
        clipped = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        clipped.fit(n_iterations=50, dilate_by_nsamples=5)
        observed = clipped.data[clipped.mask]
        vmin, vmax = observed.min().min(), observed.max().max()
        # Premise: dilation shrinks the range of masked_data
        assert clipped.masked_data.min().min() > vmin
        assert clipped.masked_data.max().max() < vmax
        # Predictions are bounded by the observed range...
        assert (clipped.predictions >= vmin).all().all()
        assert (clipped.predictions <= vmax).all().all()
        # ...and not by the narrower dilated range
        unclipped = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        unclipped.fit(n_iterations=50, dilate_by_nsamples=5, clip_predictions=False)
        np.testing.assert_allclose(
            clipped.predictions.to_numpy(),
            unclipped.predictions.clip(vmin, vmax).to_numpy(),
        )


def test_nmf_clip_range(simulate_wide_data):
    """An explicit clip_range overrides the observed min/max, and is validated"""
    for cls in [NNMF_mult, NNMF_sgd]:
        unclipped = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        unclipped.fit(n_iterations=50, clip_predictions=False)
        # Pick a range strictly inside the observed range so clipping is guaranteed to bite
        lo, hi = 20.0, 100.0
        assert unclipped.predictions.min().min() < lo
        assert unclipped.predictions.max().max() > hi

        clipped = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        clipped.fit(n_iterations=50, clip_range=(lo, hi))
        assert clipped.clip_range == (lo, hi)
        assert (clipped.predictions >= lo).all().all()
        assert (clipped.predictions <= hi).all().all()
        np.testing.assert_allclose(
            clipped.predictions.to_numpy(),
            unclipped.predictions.clip(lo, hi).to_numpy(),
        )
        # Default is the observed range
        assert unclipped.clip_range is None

        # Validation
        model = cls(simulate_wide_data, n_mask_items=0.5, random_state=2)
        with pytest.raises(ValueError, match="clip_predictions=False"):
            model.fit(n_iterations=5, clip_predictions=False, clip_range=(lo, hi))
        with pytest.raises(TypeError, match="tuple"):
            model.fit(n_iterations=5, clip_range=5)
        with pytest.raises(ValueError, match="min < max"):
            model.fit(n_iterations=5, clip_range=(hi, lo))
        with pytest.raises(ValueError, match="min < max"):
            model.fit(n_iterations=5, clip_range=(lo, np.nan))


def test_nmf_sgd_nan_divergence(simulate_wide_data):
    """A degenerate learning rate should make SGD diverge to NaN errors, which are caught and flagged rather than silently propagated or raised"""
    model = NNMF_sgd(simulate_wide_data, n_mask_items=0.5, random_state=2)
    model.fit(n_iterations=100, learning_rate=100)
    assert model.error_is_nan is True
    assert model.converged is False


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
