"""
Define pytest fixtures, i.e. reusable test initializations or parameters that can be used to automatically generated a grid of tests by test functions.
"""

import pytest
from pytest import fixture
import numpy as np
import pandas as pd
from emotioncf import (
    Mean,
    KNN,
    NNMF_sgd,
    NNMF_mult,
    create_sparse_mask,
)

## DATA FIXTURES
@fixture(scope="module")
def simulate_wide_data():
    """Generate user x item dataframe with 50 rows and 100 columns"""

    np.random.seed(0)
    i = 100
    s = 50
    rat = np.random.rand(s, i) * 50
    for x in np.arange(0, rat.shape[1], 5):
        rat[0 : int(s / 2), x] = rat[0 : int(s / 2), x] + x
    for x in np.arange(0, rat.shape[1], 3):
        rat[int(s / 2) : s, x] = rat[int(s / 2) : s, x] + x
    rat[int(s / 2) : s] = rat[int(s / 2) : s, ::-1]
    rat = pd.DataFrame(rat)
    rat.index.name = "User"
    rat.columns.name = "Item"
    return rat


@fixture(scope="module")
def simulate_long_data(simulate_wide_data):
    """Melt generated user x item dataframe to a (user * item, 3) shaped dataframe"""

    return simulate_wide_data.reset_index().melt(id_vars="User", value_name="Rating")


@fixture(scope="module")
def simulate_simple_dataframe():
    """Simple data frame with 5 users and 2 items"""
    ratings_dict = {
        "User": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        "Item": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "Rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3, 1],
    }
    return pd.DataFrame(ratings_dict)


## INIT FIXTURES
@fixture(scope="module", params=[Mean, KNN, NNMF_mult, NNMF_sgd])
def Model(request):
    """All model classes"""
    return request.param


@fixture(scope="module", params=[None, "masked"])
def mask(request, simulate_wide_data):
    """Masked or non masked input data"""
    if request.param == "masked":
        return create_sparse_mask(simulate_wide_data, n_mask_items=0.5)
    else:
        return request.param


@fixture(scope="module", params=[None, 0.1, 0.5, 0.9])
def n_mask_items(request):
    """Percentage of items to mask on init"""
    return request.param


@fixture(scope="module")
def init(Model, mask, n_mask_items, simulate_wide_data):
    """Make grid of all model init param combinations"""
    # Check that both a mask and n_mask_items together raises an error and skip making the fixture
    if mask is not None and n_mask_items is not None:
        with pytest.raises(ValueError):
            _ = Model(
                simulate_wide_data,
                mask=mask,
                n_mask_items=n_mask_items,
            )
        pytest.skip("Ambigious init fails properly - OK")
    else:
        # Otherwise put together each init param combination
        return Model(
            simulate_wide_data,
            mask=mask,
            n_mask_items=n_mask_items,
        )


## FIT FIXTURES
@fixture(scope="module")
def model(Model, simulate_wide_data, n_mask_items):
    """Initialized model with masking already performed"""
    if n_mask_items is None:
        pytest.skip("Skip testing model with dense data and no mask - OK")
    return Model(simulate_wide_data, n_mask_items=n_mask_items)


# General
@fixture(params=[None, 1, 2])
def dilate_by_nsamples(request):
    return request.param


# # KNN only models
@fixture(params=["pearson", "correlation", "cosine"])
def metric(request):
    return request.param


@fixture(params=[None, 10])
def k(request):
    return request.param


## NNMF only models
@fixture(params=[None, 10])
def n_factors(request):
    return request.param


@fixture(params=[10, 100])
def n_iterations(request):
    return request.param
