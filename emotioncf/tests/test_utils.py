"""
Test utility functions
"""

import numpy as np
import pandas as pd
from emotioncf import create_sub_by_item_matrix, nanpdist, create_train_test_mask


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
