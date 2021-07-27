"""
Define pytest fixtures, i.e. reusable test initializations or parameters that can be used to automatically generated a grid of tests by test functions. Brief explanation on how they work and how to write new fixtures + tests:

Each function below is passed in as a argument to a test function in one of the test_*.py files. The value of that argument == whatever the function definition returns in this file. For example:

# From test_models.py
test_init_and_dilate(init, mask, n_mask_items) <- These arguments are functions defined in *this* file.

init() <- returns an initialized model or skips a test
So within test_init_and_dilate(), init == model instance

At the same time, fixtures in this file can make use of *other* fixtures passed in as arguments, e.g.
mask(request, simulate_wide_data) <- simulate_wide_data() is defined below and returns a dataframe available inside of mask()

Arguments to fixtures that are not other fixtures, such as `request` in mask(), are special arguments that can create test grids based on an iterable of parameter values. These values are defined using the @fixture decorator. So `request` in mask() is defined with 2 parameter values: None and "masked". This means that wherever mask() is invoked in other tests, it will be called 2x: once with None and once with "masked".

This is used for example in init() where for some tests a model instance is created with masking and for other tests without masking. Then whatever tests make use of init(), such as test_init_and_dilate() will be run at least twice because mask() takes a `request` with 2 parameter values.

While a bit complicated at first, this make it easy to create testing grids based on the dependencies between fixtures. Here's a simplified example dependency graph (ignoring other fixtures):

Test 1:
simulate_wide_data() -> returns df -> mask()
request.param == None -> mask()
mask() -> init()
init() -> test_init_and_dilate() -> tests non-masked model initialization

Test 2:
simulate_wide_data() -> returns df -> mask()
request.param == "masked" -> mask()
mask() -> init()
init() -> test_init_and_dilate() -> tests masked model initialization

In reality there is an entire grid of tests because init() accepts several other fixtures which also have their own parameterizations, e.g. n_mask_items() creates 4 tests with None, 0.1, 0.5, 0.9. Combining this with the 2 parameterizations for mask() results in *8 unique tests*.

Hopefully this provides a relatively clear example of how to do exhaustive testing by defining parameter grids that are automatically created by pytest based on these definitions.
"""

import pytest
from pytest import fixture
import numpy as np
import pandas as pd
from neighbors import (
    Mean,
    KNN,
    NNMF_sgd,
    NNMF_mult,
    create_sparse_mask,
)
from string import ascii_letters

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
    letters = list(ascii_letters)
    letters += [f"{elem}1" for elem in letters]
    rat.index = letters[: rat.shape[0]]
    rat.columns = letters[: rat.shape[1]]
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
@fixture(params=["pearson", "cosine"])
def metric(request):
    return request.param


@fixture(params=[None, 3])
def k(request):
    return request.param


## NNMF only models
@fixture(params=[None, 10])
def n_factors(request):
    return request.param


@fixture(params=[10, 100])
def n_iterations(request):
    return request.param
