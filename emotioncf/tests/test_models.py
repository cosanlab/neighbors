"""
Test core algorithms
"""
import numpy as np
import pandas as pd
from emotioncf import (
    Base,
    Mean,
    KNN,
    NNMF_mult,
    NNMF_sgd,
    create_train_test_mask,
    create_sub_by_item_matrix,
)
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize(["cf"], [(Mean,), (KNN,), (NNMF_mult,), (NNMF_sgd,)])
def test_init(cf, simulate_simple_dataframe, capsys):

    mat = create_sub_by_item_matrix(simulate_simple_dataframe)

    # Dense data warning
    _ = cf(mat)
    captured = capsys.readouterr()
    assert "Model initialized with dense data" in captured.out

    # Existing nans warning
    df = simulate_simple_dataframe.copy()
    df.iloc[-1, -1] = np.nan
    mat = create_sub_by_item_matrix(df)
    _ = cf(mat)
    captured = capsys.readouterr()
    assert "data contains NaNs" in captured.out


@pytest.mark.parametrize(["cf"], [(Mean,), (KNN,), (NNMF_mult,), (NNMF_sgd,)])
@pytest.mark.xfail(raises=ValueError)
def test_bad_init(cf, simulate_simple_dataframe):

    mat = create_sub_by_item_matrix(simulate_simple_dataframe)
    mask = np.array(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )
    _ = cf(mat, mask=mask, n_mask_items=0.5)
    df = simulate_simple_dataframe.copy()
    df.iloc[-1, -1] = np.nan
    mat = create_sub_by_item_matrix(df)
    _ = cf(mat, n_mask_items=0.5, verbose=False)


def basecf_method_test(cf=None, dataset=None):
    """Test methods and attributes common to all models"""

    print(f"\n\nEval on: {dataset}")

    # Check masking
    if cf.is_masked:
        print("model has mask")
        assert cf.mask.shape == (50, 100)
    else:
        print("model has NO mask")

    # Check dilation
    if cf.is_mask_dilated:
        assert cf.dilated_mask.sum(axis=1).sum() > cf.mask.sum(axis=1).sum()

    # Check fit
    if cf.is_fit:
        print("model has been fit")
        # Check predictions
        assert cf.predictions.shape == (50, 100)

        # Check long format structure
        df = cf.to_long_df()
        assert isinstance(df, pd.DataFrame)
        assert "Condition" in df.columns
        assert "Observed" in df["Condition"].unique()
        assert "Predicted" in df["Condition"].unique()
        assert df.shape[0] == cf.data.shape[0] * cf.data.shape[1] * 2
        if cf.is_masked:
            assert "Mask" in df.columns

        # if dataset == "all" or (cf.is_mask and dataset in ["train", "test"]):
        #     mse = cf.get_mse(dataset=dataset)
        #     r = cf.get_corr(dataset=dataset)
        #     sub_r = cf.get_sub_corr(dataset=dataset)
        #     sub_mse = cf.get_sub_mse(dataset=dataset)
        #     assert isinstance(mse, float)
        #     assert isinstance(r, float)
        #     assert isinstance(sub_r, np.ndarray)
        #     assert len(sub_r) == cf.data.shape[0]
        #     assert isinstance(sub_mse, np.ndarray)
        #     assert len(sub_mse) == cf.data.shape[0]
        #     assert mse > 0
        #     assert np.abs(r) > 0
        #     assert np.abs(np.nanmean(sub_r)) > 0

        #     cf.plot_predictions(dataset=dataset, verbose=True)
        #     plt.close()
        #     print(f"MSE: {mse}")
    else:
        print("model NOT been fit")


def basecf_method_all_tests(cf=None):
    basecf_method_test(cf=cf, dataset="all")
    basecf_method_test(cf=cf, dataset="train")
    basecf_method_test(cf=cf, dataset="test")


@pytest.mark.parametrize(
    ["mask", "n_mask_items", "dilate_by_nsamples"],
    [(None, None, None), (None, 20, None), (None, 0.5, 2)],
)
def test_cf_mean(mask, n_mask_items, dilate_by_nsamples, simulate_wide_data):
    disp_dict = {
        "mask": mask,
        "n_mask_items": n_mask_items,
        "dilate_by_nsamples": dilate_by_nsamples,
    }
    cf = Mean(simulate_wide_data, mask=mask, n_mask_items=n_mask_items)
    print(f"\nMODEL: {cf}\nTEST PARAMS: {disp_dict}")

    cf.fit(dilate_by_nsamples=dilate_by_nsamples)
    cf.create_masked_data()
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train"]]
    cf.fit(dilate_by_nsamples=dilate_by_nsamples)
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train", "test"]]


@pytest.mark.parametrize(
    ["metric", "k", "n_mask_items", "dilate_by_nsamples"],
    [("pearson", None, 50, 2), ("correlation", 10, 0.5, 2), ("cosine", 10, 0.95, None)],
)
def test_cf_knn(metric, k, n_mask_items, dilate_by_nsamples, simulate_wide_data):
    disp_dict = {
        "metric": metric,
        "k": k,
        "n_mask_items": n_mask_items,
        "dilate_by_nsamples": dilate_by_nsamples,
    }
    cf = KNN(simulate_wide_data)
    print(f"\nMODEL: {cf}\nTEST PARAMS: {disp_dict}")

    cf.fit(metric=metric, k=k)
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train"]]
    k = k + 1 if k is not None else k
    cf.fit(metric=metric, k=k, skip_refit=True)
    cf.create_masked_data(n_mask_items=n_mask_items)
    cf.fit(metric=metric, k=k, dilate_by_nsamples=dilate_by_nsamples)
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train", "test"]]


@pytest.mark.parametrize(
    ["n_iterations", "n_mask_items", "tol", "dilate_by_nsamples", "n_factors"],
    [
        (100, 0.1, 0.001, None, 10),
        (30, 50, 0.0001, None, None),
        (10, 0.5, 0.001, 2, 5),
        (5000, 0.5, 1e-12, 2, 5),
    ],
)
def test_cf_nnmf_mult(
    n_iterations, n_mask_items, tol, dilate_by_nsamples, n_factors, simulate_wide_data
):
    disp_dict = {
        "n_iterations": n_iterations,
        "n_mask_items": n_mask_items,
        "tol": tol,
        "dilate_by_nsamples": dilate_by_nsamples,
        "n_factors": n_factors,
    }

    cf = NNMF_mult(simulate_wide_data)
    print(f"\nMODEL: {cf}\nTEST PARAMS: {disp_dict}")

    # Initial fitting on all data
    cf.fit(
        n_iterations=n_iterations,
        tol=tol,
        verbose=True,
    )
    cf.plot_learning()
    plt.close()
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train"]]

    # Now split
    cf.create_masked_data(n_mask_items=n_mask_items)
    cf.fit(
        n_iterations=n_iterations,
        tol=tol,
        dilate_by_nsamples=dilate_by_nsamples,
        verbose=True,
    )
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train", "test"]]


@pytest.mark.parametrize(
    ["n_iterations", "n_mask_items", "tol", "dilate_by_nsamples", "n_factors"],
    [
        (100, 0.1, 0.001, None, 10),
        (30, 50, 0.0001, None, None),
        (10, 0.5, 0.001, 2, 5),
        (5000, 0.5, 1e-12, 2, 5),
    ],
)
def test_cf_nnmf_sgd(
    n_iterations, n_mask_items, tol, dilate_by_nsamples, n_factors, simulate_wide_data
):
    disp_dict = {
        "n_iterations": n_iterations,
        "n_mask_items": n_mask_items,
        "tol": tol,
        "dilate_by_nsamples": dilate_by_nsamples,
        "n_factors": n_factors,
    }

    cf = NNMF_sgd(simulate_wide_data, n_mask_items=n_mask_items)
    print(f"\nMODEL: {cf}\nTEST PARAMS: {disp_dict}")

    # Initial fitting on all data
    cf.fit(
        n_iterations=n_iterations,
        tol=tol,
        verbose=True,
    )
    cf.plot_learning()
    plt.close()
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train"]]

    # Now split
    cf.create_masked_data(n_mask_items=n_mask_items)

    # Rerurn fit
    cf.fit(
        n_iterations=n_iterations,
        tol=tol,
        dilate_by_nsamples=dilate_by_nsamples,
        verbose=True,
    )
    _ = [basecf_method_test(cf, dataset) for dataset in ["all", "train", "test"]]


def test_downsample(simulate_wide_data):
    cf = Mean(simulate_wide_data)
    cf.downsample(sampling_freq=10, target=2, target_type="samples")
    assert cf.data.shape == (50, 50)
    cf = Mean(simulate_wide_data)
    cf.downsample(sampling_freq=10, target=5, target_type="hz")
    assert cf.data.shape == (50, 50)
    cf = Mean(simulate_wide_data)
    cf.downsample(sampling_freq=10, target=2, target_type="seconds")
    assert cf.data.shape == (50, 5)
    cf = Mean(simulate_wide_data)
    cf.split_train_test(n_train_items=20)
    cf.fit()
    cf.predict()
    cf.downsample(sampling_freq=10, target=2, target_type="samples")
    assert cf.data.shape == (50, 50)
    assert cf.train_mask.shape == (50, 50)
    assert cf.predictions.shape == (50, 50)

    cf = Mean(simulate_wide_data)
    cf.split_train_test(n_train_items=20)
    cf.fit(dilate_ts_n_samples=2)
    cf.predict()
    cf.downsample(sampling_freq=10, target=2, target_type="samples")
    assert cf.dilated_mask.shape == (50, 50)
    assert cf.train_mask.shape == (50, 50)
    assert cf.predictions.shape == (50, 50)


def test_base_class(simulate_wide_data):

    # Init
    model = Base(simulate_wide_data)
    assert ~model.data.isnull().any().any()
    assert model.mask is None
    assert model.n_mask_items is None

    # with pytest.raises(ValueError):
    #     model.get_data("train")

    # Masking from training size
    model = Base(simulate_wide_data, n_mask_items=0.1)
    assert ~model.data.isnull().any().any()
    assert model.masked_data.isnull().any().any()
    assert model.is_masked is True
    assert model.mask is not None

    # Masking from input mask
    mask = create_train_test_mask(simulate_wide_data)
    model = Base(simulate_wide_data, mask=mask)
    assert ~model.data.isnull().any().any()
    assert model.masked_data.isnull().any().any()
    assert model.is_masked is True
    assert model.mask is not None

    # Ensure get_data respects mask
    # train = model.get_data("train")
    # test = model.get_data("test")
    # all_data = model.get_data()

    # assert train.isnull().sum().sum() > test.isnull().sum().sum()
    # assert ~all_data.isnull().any().any()

    with pytest.raises(ValueError):
        model = Base(simulate_wide_data, mask=mask, n_mask_items=0.1)

    # Mask dilation
    model.dilate_mask(n_samples=5)
    assert model.is_mask_dilated is True
    assert model.masked_data.isnull().sum().sum() > model.mask.sum().sum()

    # Make sure get_data respects dilation
    # assert train.isnull().sum().sum() > model.get_data("train").isnull().sum().sum()
