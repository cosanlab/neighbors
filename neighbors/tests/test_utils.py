"""
Test utility functions
"""

import numpy as np
import pandas as pd
from neighbors import (
    create_user_item_matrix,
    invert_user_item_matrix,
    nanpdist,
    create_sparse_mask,
    estimate_performance,
    flatten_dataframe,
    unflatten_dataframe,
    split_train_test,
    Mean,
)


def test_estimate_performance(simulate_wide_data):
    # DENSE DATA
    # 1) Only out-of-sample
    group_out, user_out = estimate_performance(Mean, simulate_wide_data, verbose=True)
    assert group_out.shape == (4 * 2, 6)
    assert user_out.shape == (50, 4)
    # 2) In and out-of-sample
    group_out, user_out = estimate_performance(
        Mean,
        simulate_wide_data,
        verbose=True,
        return_full_performance=True,
        parallelize=True,
    )
    assert group_out.shape == (4 * 2 * 2, 6)
    assert user_out.shape == (50, 8)
    missing = group_out.query("dataset == 'missing' and group =='all'")["mean"]
    observed = group_out.query("dataset == 'observed' and group =='all'")["mean"]
    # Make sure all missing scores are worse than observed
    for i, (o, m) in enumerate(zip(observed, missing)):
        if i == 0:
            assert o > m
        else:
            assert o < m
    # 3) Non-aggregated out-of-sample, with timing info
    group_out, user_out = estimate_performance(
        Mean, simulate_wide_data, return_agg=False, verbose=True
    )
    assert group_out.shape == (4 * 2 * 10, 7)
    assert user_out.shape == (50 * 10, 6)

    # SPARSE DATA
    mask = create_sparse_mask(simulate_wide_data)
    masked_data = simulate_wide_data[mask]
    # 1) Only out-of-sample
    group_out, user_out = estimate_performance(Mean, masked_data, verbose=True)
    assert group_out.shape == (4 * 2, 6)
    assert user_out.shape == (50, 4)
    # 2) In and out-of-sample
    group_out, user_out = estimate_performance(
        Mean,
        simulate_wide_data,
        verbose=True,
        return_full_performance=True,
        parallelize=True,
    )
    assert group_out.shape == (4 * 2 * 2, 6)
    assert user_out.shape == (50, 8)
    test = group_out.query("dataset == 'test' and group =='all'")["mean"]
    train = group_out.query("dataset == 'train' and group =='all'")["mean"]
    # Make sure all test scores are worse than train
    for i, (o, m) in enumerate(zip(train, test)):
        if i == 0:
            assert o > m
        else:
            assert o < m
    # 3) Non-aggregated out-of-sample, without timing info
    group_out, user_out = estimate_performance(
        Mean, masked_data, return_agg=False, verbose=True, timeit=False
    )
    assert group_out.shape == (4 * 2 * 10, 6)
    assert user_out.shape == (50 * 10, 6)


# TODO: Update this test to handle commented out lines. This is a pandas issue where going from long -> wide -> long leads pandas to sort columns rather than preserving the original column order
def test_create_and_invert_user_item_matrix(simulate_long_data, simulate_wide_data):
    rating = create_user_item_matrix(simulate_long_data)
    assert isinstance(rating, pd.DataFrame)
    assert rating.shape == (50, 100)
    # assert rating.equals(simulate_wide_data)

    inverted = invert_user_item_matrix(rating)
    assert inverted.shape == (50 * 100, 3)
    # assert inverted.equals(simulate_long_data)

    renamed = simulate_long_data.rename(
        columns={"User": "A", "Item": "B", "Rating": "C"}
    )
    rating = create_user_item_matrix(renamed, columns=["A", "B", "C"])
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


def test_create_sparse_mask(simulate_wide_data):
    mask = create_sparse_mask(simulate_wide_data, n_mask_items=0.1)
    expected_items = int(simulate_wide_data.shape[1] * (1 - 0.10))
    assert mask.shape == simulate_wide_data.shape
    assert all(mask.sum(1) == expected_items)

    mask = create_sparse_mask(simulate_wide_data, n_mask_items=19)
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
    mask = create_sparse_mask(simulate_wide_data, n_mask_items=0.1)
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
