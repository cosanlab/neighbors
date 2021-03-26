"""
Test utility functions
"""

from emotioncf.utils import split_train_test
import numpy as np
import pandas as pd
from emotioncf import (
    create_sub_by_item_matrix,
    nanpdist,
    create_train_test_mask,
    estimate_performance,
    approximate_generalization,
    flatten_dataframe,
    unflatten_dataframe,
    Mean,
)


def test_estimate_performance(simulate_wide_data):
    out = estimate_performance(
        Mean, simulate_wide_data, model_kwargs={"n_mask_items": 0.2}
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (4 * 3 * 2, 6)
    out = estimate_performance(
        Mean,
        simulate_wide_data,
        model_kwargs={"n_mask_items": 0.2},
        agg_stats=["mean", "std", "var"],
    )
    assert out.shape == (4 * 3 * 2, 7)
    out = estimate_performance(
        Mean, simulate_wide_data, model_kwargs={"n_mask_items": 0.2}, return_agg=False
    )
    assert out.shape == (4 * 3 * 2 * 10, 6)


def test_approximate_generalization(simulate_wide_data):
    out = approximate_generalization(Mean, simulate_wide_data)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (4 * 3 * 2 * 2, 7)
    out = approximate_generalization(
        Mean,
        simulate_wide_data,
        agg_stats=["mean", "std", "var"],
    )
    assert out.shape == (4 * 3 * 2 * 2, 8)
    out = approximate_generalization(Mean, simulate_wide_data, return_agg=False)
    assert out.shape == (4 * 3 * 2 * 10, 7)


def test_create_sub_by_item_matrix(simulate_long_data):
    rating = create_sub_by_item_matrix(simulate_long_data)
    assert isinstance(rating, pd.DataFrame)
    assert rating.shape == (50, 100)

    renamed = simulate_long_data.rename(
        columns={"Subject": "A", "Item": "B", "Rating": "C"}
    )
    rating = create_sub_by_item_matrix(renamed, columns=["A", "B", "C"])
    assert isinstance(rating, pd.DataFrame)
    assert rating.shape == (50, 100)


def test_nanpdist(simulate_wide_data):
    # Non-nan data should behave like pdist
    out = nanpdist(simulate_wide_data.to_numpy())
    assert out.ndim == 2
    assert np.allclose(out, out.T, rtol=1e-05, atol=1e-08)
    out = nanpdist(simulate_wide_data.to_numpy(), return_square=False)
    assert out.ndim == 1

    # Now mask it
    mask = np.random.choice([0, 1], size=simulate_wide_data.shape)
    df = simulate_wide_data * mask
    df = df.replace({0: np.nan})

    out = nanpdist(df.to_numpy())
    assert out.ndim == 2
    assert np.allclose(out, out.T, rtol=1e-05, atol=1e-08)
    out = nanpdist(df.to_numpy(), return_square=False)
    assert out.ndim == 1

    calc_corr_mat = 1 - nanpdist(df.to_numpy(), metric="correlation")
    pd_corr_mat = df.T.corr(method="pearson").to_numpy()
    assert np.allclose(calc_corr_mat, pd_corr_mat)


def test_create_train_test_mask(simulate_wide_data):
    mask = create_train_test_mask(simulate_wide_data, n_mask_items=0.1)
    expected_items = int(simulate_wide_data.shape[1] * (1 - 0.10))
    assert mask.shape == simulate_wide_data.shape
    assert all(mask.sum(1) == expected_items)

    mask = create_train_test_mask(simulate_wide_data, n_mask_items=19)
    assert mask.shape == simulate_wide_data.shape
    expected_items = int(simulate_wide_data.shape[1] - 19)
    assert all(mask.sum(1) == expected_items)

    masked_data = simulate_wide_data[mask]
    assert isinstance(masked_data, pd.DataFrame)
    assert masked_data.shape == simulate_wide_data.shape
    assert ~simulate_wide_data.isnull().any().any()
    assert masked_data.isnull().any().any()


def test_flatten_dataframe(simulate_wide_data):
    out = flatten_dataframe(simulate_wide_data)
    assert isinstance(out, np.ndarray)
    assert len(out) == simulate_wide_data.shape[0] * simulate_wide_data.shape[1]
    assert all([len(elem) == 3 for elem in out])


def test_unflatten_dataframe(simulate_wide_data):
    out = flatten_dataframe(simulate_wide_data)
    new = unflatten_dataframe(
        out, index=simulate_wide_data.index, columns=simulate_wide_data.columns
    )
    assert new.equals(simulate_wide_data)
    new = unflatten_dataframe(out)
    assert new.equals(simulate_wide_data)
    new = unflatten_dataframe(
        out, num_rows=simulate_wide_data.shape[0], num_cols=simulate_wide_data.shape[1]
    )


def test_split_train_test(simulate_wide_data):
    # Dense data
    for train, test in split_train_test(simulate_wide_data, n_folds=5):
        assert train.shape == simulate_wide_data.shape
        assert test.shape == simulate_wide_data.shape
        # 1/5 of dense data should be sparse for training, i.e. 4/5 data folds
        assert train.notnull().sum().sum() == int(4 / 5 * simulate_wide_data.size)
        # 4/5 of dense data should be sparse for testing, i.e. 1/5 data folds
        assert test.notnull().sum().sum() == int(1 / 5 * simulate_wide_data.size)
        assert train.add(test, fill_value=0).equals(simulate_wide_data)

    # Sparse data
    mask = create_train_test_mask(simulate_wide_data, n_mask_items=0.1)
    masked_data = simulate_wide_data[mask]
    masked_not_null = masked_data.notnull().sum().sum()

    for train, test in split_train_test(masked_data, n_folds=5):
        assert train.shape == mask.shape
        assert test.shape == mask.shape
        # Train and test should be more sparse than original
        assert train.isnull().sum().sum() > masked_data.isnull().sum().sum()
        assert test.isnull().sum().sum() > masked_data.isnull().sum().sum()
        assert train.add(test, fill_value=0).equals(masked_data)
        train_not_null = train.notnull().sum().sum()
        test_not_null = test.notnull().sum().sum()
        # And adhere close the expected train/test split
        assert np.allclose(train_not_null / masked_not_null, 0.8, atol=0.12)
        assert np.allclose(test_not_null / masked_not_null, 0.2, atol=0.12)
