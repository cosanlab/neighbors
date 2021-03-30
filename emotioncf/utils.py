"""
Utility functions and helpers
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
import numbers
from itertools import product, chain

__all__ = [
    "create_sub_by_item_matrix",
    "get_size_in_mb",
    "get_sparsity",
    "nanpdist",
    "create_train_test_mask",
    "estimate_performance",
    "approximate_generalization",
    "flatten_dataframe",
    "unflatten_dataframe",
    "split_train_test",
    "check_random_state",
]


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance, lifted directly from sklearn to avoid a heavy dependency for one function.

    Olivier Grisel, Andreas Mueller, Lars, Alexandre Gramfort, Gilles Louppe, Peter Prettenhofer, … Eustache. (2021, January 19). scikit-learn/scikit-learn: scikit-learn 0.24.1 (Version 0.24.1). Zenodo. http://doi.org/10.5281/zenodo.4450597

    Args:
        seed (None, int, RandomState): If seed is None, return the RandomState singleton used by np.random. If seed is an int, return a new RandomState instance seeded with seed. If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def create_sub_by_item_matrix(df, columns=None, force_float=True, errors="raise"):

    """Convert a pandas long data frame of a single rating into a subject by item matrix

    Args:
        df (Dataframe): input dataframe
        columns (list): list of length 3 with dataframe columns to use for reshaping. The first value should reflect unique individual identifier ("Subject"), the second a unique item identifier ("Item", "Timepoint"), and the last the rating made by the individual on that item ("Rating"). Defaults to ["Subject", "Item", "Rating"]
        force_float (bool): force the resulting output to be float data types with errors being set to NaN; Default True
        errors (string): how to handle errors in pd.to_numeric; Default 'raise'

    Return:
        pd.DataFrame: user x item rating Dataframe

    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be pandas instance")
    if columns is None:
        columns = ["Subject", "Item", "Rating"]
    if not all([x in df.columns for x in columns]):
        raise ValueError(
            f"df is missing some or all of the following columns: {columns}"
        )

    ratings = df[columns]
    ratings = ratings.pivot(index=columns[0], columns=columns[1], values=columns[2])
    try:
        if force_float:
            ratings = ratings.apply(pd.to_numeric, errors=errors)
    except ValueError as e:
        print(
            "Auto-converting data to floats failed, probably because you have non-numeric data in some rows. You can set errors = 'coerce' to set these failures to NaN"
        )
        raise (e)

    return ratings


def get_size_in_mb(arr):
    """Calculates size of ndarray in megabytes"""
    if isinstance(arr, (np.ndarray, csr_matrix)):
        return arr.data.nbytes / 1e6
    else:
        raise TypeError("input must by a numpy array or scipy csr sparse matrix")


def get_sparsity(arr):
    """Calculates sparsity of ndarray (0 - 1)"""
    if isinstance(arr, np.ndarray):
        return 1 - (np.count_nonzero(arr) / arr.size)
    elif isinstance(arr, pd.DataFrame):
        if arr.isnull().any().any():
            return get_sparsity(arr.fillna(0).to_numpy())
        else:
            return get_sparsity(arr.to_numpy())
    else:
        raise TypeError("input must be a numpy array")


# Can try to speed this up with numba, but lose support for pandas and scipy so we'd have to rewrite distance functions in numpy/python
def nanpdist(arr, metric="euclidean", return_square=True):
    """
    Just like scipy.spatial.distance.pdist or sklearn.metrics.pairwise_distances, but respects NaNs by only comparing the overlapping values from pairs of rows.

    Args:
        arr (np.ndarray): 2d array
        metric (str; optional): distance metric to use. Must be supported by scipy
        return_square (boo; optional): return a symmetric 2d distance matrix like sklearn instead of a 1d vector like pdist; Default True

    Return:
        np.ndarray: symmetric 2d array of distances
    """

    has_nans = pd.DataFrame(arr).isnull().any().any()
    if not has_nans:
        out = pdist(arr, metric=metric)
    else:
        nrows = arr.shape[0]
        out = np.zeros(nrows * (nrows - 1) // 2, dtype=float)
        mask = np.isfinite(arr)
        vec_mask = np.zeros(arr.shape[1], dtype=bool)
        k = 0
        for row1_idx in range(nrows - 1):
            for row2_idx in range(row1_idx + 1, nrows):
                vec_mask = np.logical_and(mask[row1_idx], mask[row2_idx])
                masked_row1, masked_row2 = (
                    arr[row1_idx][vec_mask],
                    arr[row2_idx][vec_mask],
                )
                out[k] = pdist(np.vstack([masked_row1, masked_row2]), metric=metric)
                k += 1

    if return_square:
        if out.ndim == 1:
            out = squareform(out)
    else:
        if out.ndim == 2:
            out = squareform(out)
    return out


def create_train_test_mask(data, n_mask_items=0.1, random_state=None):
    """
    Given a pandas dataframe create a boolean mask such that n_mask_items columns are `False` and the rest are `True`. Critically, each row is masked independently. This function does not alter the input dataframe.

    Args:
        data (pd.DataFrame): input dataframe
        n_train_items (float, optional): if an integer is passed its raw value is used. Otherwise if a float is passed its taken to be a (rounded) percentage of the total items. Defaults to 0.1 (10% of the columns of data).

    Raises:
        TypeError: [description]

    Returns:
        pd.DataFrame: boolean dataframe of same shape as data
    """

    if data.isnull().any().any():
        raise ValueError("data already contains NaNs and further masking is ambiguous!")

    if isinstance(n_mask_items, (float, np.floating)) and 1 >= n_mask_items > 0:
        n_false_items = int(np.round(data.shape[1] * n_mask_items))

    elif isinstance(n_mask_items, (int, np.integer)):
        n_false_items = n_mask_items

    else:
        raise TypeError(
            f"n_train_items must be an integer or a float between 0-1, not {type(n_mask_items)} with value {n_mask_items}"
        )

    random = check_random_state(random_state)
    n_true_items = data.shape[1] - n_false_items
    mask = np.array([True] * n_true_items + [False] * n_false_items)
    mask = np.vstack(
        [
            random.choice(mask, replace=False, size=mask.shape)
            for _ in range(data.shape[0])
        ]
    )
    return pd.DataFrame(mask, index=data.index, columns=data.columns)


def load_movielens():  # pragma: no cover
    """Download and create a dataframe from the 100k movielens dataset"""
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    # With python context managers we don't need to save any temporary files
    print("Getting movielens...")
    try:
        with urlopen(url) as resp:
            with ZipFile(BytesIO(resp.read())) as myzip:
                with myzip.open("ml-100k/u.data") as myfile:
                    df = pd.read_csv(
                        myfile,
                        delimiter="\t",
                        names=["Subject", "Item", "Rating", "Timestamp"],
                    )
        return df
    except Exception as e:
        print(str(e))


def estimate_performance(
    algorithm,
    data: pd.DataFrame,
    n_iter: int = 10,
    return_agg: bool = True,
    agg_stats: tuple = ["mean", "std"],
    model_kwargs: dict = {},
    fit_kwargs: dict = {},
    random_state=None,
) -> pd.DataFrame:
    """
    Repeatedly call fit on a model and a dataset. Useful for benchmarking an algorithm on a dataset as each iteration will generate a new random mask.

    Args:
        algorithm (emotioncf.model): an uninitialized model, e.g. `Mean`
        data (pd.DataFrame): a users x item dataframe
        n_iter (int, optional): number of repetitions. Defaults to 10.
        return_agg (bool, optional): [description]. return mean and std over repetitions rather than the reptitions themselves Defaults to True.
        agg_stats (list): string names of statistics to compute over repetitions. Must be accepted by `pd.DataFrame.agg`; Default ('mean', 'std')
        model_kwargs (dict, optional): [description]. A dictionary of arguments passed when the model is first initialized, e.g. `Mean(**model_kwargs)`. Defaults to {}.
        fit_kwargs (dict, optional): Same as the `model_kwargs` but passed to `.fit()`. Defaults to {}.

    Returns:
        pd.DataFrame: aggregated or non-aggregated summary statistics
    """

    all_results = []
    for i in range(n_iter):
        model = algorithm(data=data, random_state=random_state, **model_kwargs)
        model.fit(**fit_kwargs)
        results = model.summary(return_cached=False)
        results["iter"] = i
        all_results.append(results)
    all_results = pd.concat(all_results, ignore_index=True)
    if return_agg:
        out = (
            all_results.groupby(["algorithm", "dataset", "group", "metric"])
            .score.agg(agg_stats)
            .reset_index()
        )
        return out
    else:
        return all_results


def flatten_dataframe(data: pd.DataFrame) -> list:
    """
    Given a 2d dataframe return a numpy array of arrays organized as (row_idx, col_idx, val). This function is analgous to numpy.ravel or numpy.flatten for arrays, with the addition of the row and column indices for each value

    Args:
        data (pd.DataFrame): input dataframe

    Returns:
        np.ndarray: arrayo of row, column, value triplets
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("input must be a pandas dataframe")

    out = zip(
        product(range(data.shape[0]), range(data.shape[1])), data.to_numpy().ravel()
    )
    return np.array([(elem[0][0], elem[0][1], elem[1]) for elem in out])


def unflatten_dataframe(
    data: np.ndarray, columns=None, index=None, num_rows=None, num_cols=None
) -> pd.DataFrame:
    """
    Reverse a flatten_dataframe operation to reconstruct the original unflattened dataframe

    Args:
        data (list, np.ndarray): list of (row_idx, col_idx, val) tuples or numpy array of [row_idx, col_idx, val] lists.
        columns (list, optional): column names of new dataframe. Defaults to None.
        index (list, optional): row names of new dataframe. Defaults to None.

    Returns:
        pd.DataFrame: original unflattened dataframe
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("input should be a numpy array")
    num_rows = int(data[:, 0].max()) + 1 if num_rows is None else num_rows
    num_cols = int(data[:, 1].max()) + 1 if num_cols is None else num_cols
    out = np.empty((num_rows, num_cols))
    out[:] = np.nan
    for elem in data:
        out[int(elem[0]), int(elem[1])] = elem[2]
    return pd.DataFrame(out, index=index, columns=columns)


def downsample_dataframe(data, n_samples, sampling_freq=None, target_type="samples"):
    """
    Down sample a dataframe

    Args:
        data (pd.DataFrame): input data
        n_samples (int): number of samples.
        sampling_freq (int/float, optional): sampling frequency of data in hz. Defaults to None.
        target_type (str, optional): how to downsample; must be one of "samples", "seconds" or "hz". Defaults to "samples".

    Returns:
        pd.DataFrame: downsampled input data
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas dataframe")
    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer")
    if target_type not in ["samples", "seconds", "hz"]:
        raise ValueError("target_type must be 'samples', 'seconds', or 'hz'")

    if target_type in ["seconds", "hz"] and (
        n_samples is None or sampling_freq is None
    ):
        raise ValueError(
            f"if target_type = {target_type}, both sampling_freq and target must be provided"
        )

    if target_type == "seconds":
        n_samples = n_samples * sampling_freq
    elif target_type == "hz":
        n_samples = sampling_freq / n_samples
    else:
        n_samples = n_samples

    data = data.T
    idx = np.sort(np.repeat(np.arange(1, data.shape[0] / n_samples, 1), n_samples))
    if data.shape[0] > len(idx):
        idx = np.concatenate([idx, np.repeat(idx[-1] + 1, data.shape[0] - len(idx))])
    return data.groupby(idx).mean().T


# TODO: adjust the output to drop missing/observed, etc that are NaNs for training/testing splits
def approximate_generalization(
    algorithm,
    data: pd.DataFrame,
    n_folds: int = 5,
    return_agg: bool = True,
    agg_stats: tuple = ["mean", "std"],
    model_kwargs: dict = {},
    fit_kwargs: dict = {},
    random_state=None,
) -> pd.DataFrame:
    """
    Similar to estimate_performance but uses leave-one-fold-out cross-validation in addition random masking. **Note**: this is done by further masking the input data, thereby *increasing sparsity*. Specifically, data is always "split" into training and testing folds by masking user-item combinations such that folds have non-overlapping user-item scores and missing values. The number of folds requested controls the additional sparsity of each train and test split, e.g. n_folds = 5 means train = 4/5 (~80% of *observed values*); test = 1/5 (~20% of *observed values*). Models are estimated against the training set and then used to predict values in the test set without additional training.

    Args:
        algorithm (emotioncf.model): an uninitialized model, e.g. `Mean`
        data (pd.DataFrame): a users x item dataframe
        n_iter (int, optional): number of repetitions. Defaults to 10.
        return_agg (bool, optional): [description]. return mean and std over repetitions rather than the reptitions themselves Defaults to True.
        agg_stats (list): string names of statistics to compute over repetitions. Must be accepted by `pd.DataFrame.agg`; Default ('mean', 'std')
        model_kwargs (dict, optional): [description]. A dictionary of arguments passed when the model is first initialized, e.g. `Mean(**model_kwargs)`. Defaults to {}.
        fit_kwargs (dict, optional): Same as the `model_kwargs` but passed to `.fit()`. Defaults to {}.

    Returns:
        pd.DataFrame: aggregated or non-aggregated summary statistics
    """

    all_results = []
    fold = 1
    for train, test in split_train_test(data, n_folds):
        model = algorithm(data=train, random_state=random_state, **model_kwargs)
        model.fit(**fit_kwargs)
        train_results = model.summary(dataset="full")
        train_results["fold"] = fold
        train_results["cv"] = "train"
        test_results = model.summary(actual=test, dataset="full", return_cached=False)
        test_results["fold"] = fold
        test_results["cv"] = "test"
        all_results.append(train_results)
        all_results.append(test_results)
        fold += 1
    all_results = pd.concat(all_results, ignore_index=True).drop(columns=["dataset"])
    if return_agg:
        out = (
            all_results.groupby(["cv", "algorithm", "group", "metric"])
            .score.agg(agg_stats)
            .reset_index()
        )
        return out
    else:
        return all_results


def split_train_test(
    data: pd.DataFrame, n_folds: int = 5, shuffle=True, random_state=None
):
    """
    Custom train/test split generator for leave-one-fold out cross-validation. Given a user x item dataframe of dense or sparse data, generates n_folds worth of train/test split dataframes that have the same shape as data, but with non-overlapping values. This ensures that no user-item combination appears in both train and test splits. Useful for estimating out-of-sample performance. n_folds controls the train/test split ratio, e.g. n_folds = 5 means train = 4/5 (~80%); test = 1/5 (~20%)

    Args:
        data (pd.DataFrame): user x item dataframe
        n_folds (int, optional): number of train/test splits to generate. Defaults to 5.
        shuffle (bool, optional): randomize what user-item combinations appear in each split. If shuffle is True, folds will split in order of item user-item appearance; Default True
        random_state (int, None, numpy.random.mtrand.RandomState): random state for reproducibility; Default None

    Yields:
        (tuple): train, test dataframes for each fold with same shape as data
    """

    random_state = check_random_state(random_state)
    num_rows, num_cols = data.shape
    flat = flatten_dataframe(data)
    if shuffle:
        random_state.shuffle(flat)
    start, stop = 0, 0
    for fold in range(n_folds):
        start = stop
        stop += len(flat) // n_folds
        if fold < len(flat) % n_folds:
            stop += 1

        train = np.array([elem for elem in chain(flat[:start], flat[stop:])])
        test = np.array([elem for elem in flat[start:stop]])

        yield unflatten_dataframe(
            train, num_rows=num_rows, num_cols=num_cols
        ), unflatten_dataframe(test, num_rows=num_rows, num_cols=num_cols)
