"""
Utility functions and helpers
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import numbers
from itertools import product, chain
from typing import Union
from concurrent.futures import ThreadPoolExecutor
import os
import time

__all__ = [
    "create_user_item_matrix",
    "invert_user_item_matrix",
    "get_size_in_mb",
    "get_sparsity",
    "nanpdist",
    "create_sparse_mask",
    "estimate_performance",
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


def invert_user_item_matrix(df):
    """
    Reshapes a user x item matrix back into a longform dataframe with "User", "Item" and "Rating" columns. Convenience function to undo a call to `create_user_item_matrix`. Dataframe must have an index name of 'Users' and a columns name of 'Items'.

    Args:
        df (pd.DataFrame): user x item dataframe, e.g. the output of a call to `create_user_by_item_matrix`

    Returns:
        pd.DataFrame: longform dataframe with "User", "Item", "Rating" columns
    """

    if df.index.name != "User" or df.columns.name != "Item":
        raise TypeError(
            "input should be a user x item matrix with appropriately named columns and rows"
        )
    return df.reset_index().melt(id_vars="User", value_name="Rating")


def create_user_item_matrix(df, columns=None, force_float=True, errors="raise"):

    """Convert a longform dataframe containing columns with unique user ids, item ids, and ratings into a user x item wide matrix

    Args:
        df (Dataframe): input dataframe
        columns (list): list of length 3 with dataframe columns to use for reshaping. The first value should reflect unique individual identifier ("User"), the second a unique item identifier (e.g. "Item"), and the last the rating made by the individual on that item ("Rating"). Defaults to ["User", "Item", "Rating"]
        force_float (bool): force the resulting output to be float data types with errors being set to NaN; Default True
        errors (string): how to handle errors in pd.to_numeric; Default 'raise'

    Return:
        pd.DataFrame: user x item rating Dataframe

    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be pandas instance")
    if columns is None:
        columns = ["User", "Item", "Rating"]
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
    ratings.index.name = "User"
    ratings.columns.name = "Item"

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
        return arr.isnull().sum().sum() / arr.size
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


def create_sparse_mask(data, n_mask_items=0.2, random_state=None):
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

    out = zip(product(data.index, data.columns), data.to_numpy().ravel())
    return np.array([(elem[0][0], elem[0][1], elem[1]) for elem in out])


def unflatten_dataframe(
    data: np.ndarray,
    columns=None,
    index=None,
    num_rows=None,
    num_cols=None,
    index_name=None,
    columns_name=None,
) -> pd.DataFrame:
    """
    Reverse a flatten_dataframe operation to reconstruct the original unflattened dataframe

    Args:
        data (np.ndarray): n_items x 3 numpy array where columns represent row_idx, col_idx, and val at the location.
        columns (list, optional): column names of new dataframe. Defaults to None.
        index (list, optional): row names of new dataframe. Defaults to None.
        num_rows (int, optional): total number of rows. Useful if the flattened dataframe had a non-numerical non-ordered index. Default None which uses the max(row_idx)
        num_cols (int, optional): total number of cols. Useful if the flattened dataframe had a non-numerical non-ordered index. Default None which uses the max(col_idx)
        index_name (str; optional): Name of rows; Default None
        columns_name (str; optional): Name of columns; Default None

    Returns:
        pd.DataFrame: original unflattened dataframe
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("input should be a numpy array")
    if index is None and num_rows is None:
        index = list(dict.fromkeys(data[:, 0]))
        num_rows = len(index)
    elif index is not None and num_rows is None:
        num_rows = len(index)
    elif index is None and num_rows is not None:
        index = list(dict.fromkeys(data[:, 0]))
        if len(index) != num_rows:
            raise ValueError(
                "num_rows does not match the number of unique row_idx values in data"
            )
    if columns is None and num_cols is None:
        columns = list(dict.fromkeys(data[:, 1]))
        num_cols = len(columns)
    elif columns is not None and num_cols is None:
        num_cols = len(columns)
    elif columns is None and num_cols is not None:
        columns = list(dict.fromkeys(data[:, 1]))
        if len(columns) != num_cols:
            raise ValueError(
                "num_cols does not match the number of unique col_idx values in data"
            )
    out = np.empty((num_rows, num_cols))
    out[:] = np.nan
    out = pd.DataFrame(out, index=index, columns=columns)
    for elem in data:
        out.loc[elem[0], elem[1]] = np.float(elem[2])
    out.index.name = index_name
    out.columns.name = columns_name
    return out


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
            "if target_type = {target_type}, both sampling_freq and target must be provided"
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


def split_train_test(
    data: pd.DataFrame, n_folds: int = 10, shuffle=True, random_state=None
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
            train,
            num_rows=num_rows,
            num_cols=num_cols,
            index=data.index,
            columns=data.columns,
            index_name=data.index.name,
            columns_name=data.columns.name,
        ), unflatten_dataframe(
            test,
            num_rows=num_rows,
            num_cols=num_cols,
            index=data.index,
            columns=data.columns,
            index_name=data.index.name,
            columns_name=data.columns.name,
        )


def estimate_performance(
    algorithm,
    data: pd.DataFrame,
    n_iter: int = 10,
    n_folds: int = 10,
    n_mask_items: Union[int, np.floating] = 0.2,
    return_agg: bool = True,
    return_full_performance: bool = False,
    agg_stats: tuple = ["mean", "std"],
    fit_kwargs: dict = {},
    random_state=None,
    parallelize=False,
    timeit=True,
    verbose=False,
) -> pd.DataFrame:
    """
    Repeatedly fit a model with data to estimate performance. If input data is dense (contains no missing values) then this function will fit the model `n_iter` times after applying a different random mask each iteration according to `n_mask_items`. This is useful for testing how an algorithm *would have performed* given data of a specified sparsity.

    On the other hand, if input data is already sparse, no further masking will be performed and the data will instead be split into `n_folds` training and testing folds. Evaluation will be performed based on how well non-missing values in testing folds are recovered. **Note**: this *increases the sparsity* of the input dataset. The number of folds requested controls the additional sparsity of each train and test split. For example, with the default `n_folds=10`, each training split will contain 9/10 folds = ~90% of *observed values* (additional sparsity of 10%); each test split will contain 1/10 folds (~10% of *observed values*).

    Args:
        algorithm (neighbors.model): an uninitialized model, e.g. `Mean`
        data (pd.DataFrame): a users x item dataframe
        n_iter (int, optional): number of repetitions for dense data. Defaults to 10.
        n_folds (int, optional): number of folds for CV on sparse data. Defaults to 10.
        n_mask_items (int/float, optional): how much randomly sparsify dense data each iteration; Defaults to masking out 20% of observed values
        return_agg (bool, optional): Return mean and std over repetitions rather than the reptitions themselves Defaults to True.
        return_full_performance (bool, optional): return the performance against both "observed" and "missing" or just "missing" values if using dense data and `n_iter`. Likewise return performance of both "train" and "test" or just "test" splits if using sparse data and `n_folds`; Default False
        agg_stats (list): string names of statistics to compute over repetitions. Must be accepted by `pd.DataFrame.agg`; Default ('mean', 'std')
        fit_kwargs (dict, optional): A dictionary of arguments passed to the `.fit()` method of the model. Defaults to {}.
        random_state (None, int, RandomState): a seed or random state used for all internal random operations (e.g. random masking, cv splitting, etc). Passing None will generate a new random seed; Default None.
        parallelize (bool, optional): Use multiple *threads* to run n_iter or n_folds in parallel. To save memory and prevent data duplication, this does not use multiple *processes* like joblib. Some algorithms like `NNMF_sgd` can see significant speed-ups (2-3x) when `True`. Others, like `NNMF_mult` do not not gain much benefit or may even slow down. Default False.
        timeit (bool, option): include a column of estimation + prediction duration for each iteration/fold in the results. Ignored when return_agg=True; Default True.
        verbose (bool, optional): print information messages on execution; Default False.


    Returns:
        pd.DataFrame: aggregated or non-aggregated summary statistics
    """

    sparsity = get_sparsity(data)
    random_state = check_random_state(random_state)
    # Set max threads to the new default in Python 3.8
    max_threads = min(32, os.cpu_count() + 4)

    if sparsity == 0.0:
        # DENSE DATA so re-mask each iteration
        print(f"Data sparsity is {np.round(sparsity*100,2)}%. Using random masking...")

        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_iter)

        def _run_dense(
            i,
            algorithm,
            data,
            random_state,
            n_mask_items,
            return_full_performance,
            timeit,
            fit_kwargs,
        ):
            if timeit:
                start = time.time()
            model = algorithm(
                data=data,
                random_state=random_state,
                n_mask_items=n_mask_items,
                verbose=False,
            )
            model.fit(**fit_kwargs)
            # observed + missing
            if return_full_performance:
                results = model.summary(dataset=["missing", "observed"])
            # only missing
            else:
                results = model.summary(dataset="missing")
            results["iter"] = i + 1

            # Individual user results
            user_results = model.user_results
            user_results["iter"] = i + 1

            if timeit:
                stop = time.time()
                duration = stop - start
                results["duration"] = duration
            return results, user_results

        if parallelize:
            if verbose:
                print("Parallelizing across threads...")

            with ThreadPoolExecutor(max_threads) as threads:
                all_results = threads.map(
                    _run_dense,
                    range(n_iter),
                    [algorithm] * n_iter,
                    [data] * n_iter,
                    seeds,
                    [n_mask_items] * n_iter,
                    [return_full_performance] * n_iter,
                    [timeit] * n_iter,
                    [fit_kwargs] * n_iter,
                )
        else:
            all_results = map(
                _run_dense,
                range(n_iter),
                [algorithm] * n_iter,
                [data] * n_iter,
                seeds,
                [n_mask_items] * n_iter,
                [return_full_performance] * n_iter,
                [timeit] * n_iter,
                [fit_kwargs] * n_iter,
            )

        col_order = ["algorithm", "dataset", "iter", "group", "metric", "score"]
        user_agg_drop_col = ["iter"]
    else:
        # SPARSE DATA so split observed values according to n_folds
        print(
            f"Data sparsity is {np.round(sparsity*100,2)}%. Using cross-validation..."
        )

        def _run_sparse(
            i,
            train,
            test,
            algorithm,
            random_state,
            return_full_performance,
            timeit,
            fit_kwargs,
        ):
            if timeit:
                start = time.time()
            model = algorithm(data=train, random_state=random_state, verbose=False)
            model.fit(**fit_kwargs)
            test_results = model.summary(actual=test, dataset="full")
            test_results["cv_fold"] = i + 1
            test_results["dataset"] = test_results.dataset.map({"full": "test"})

            user_test_results = model.user_results
            user_test_results["cv_fold"] = i + 1
            user_test_results.columns = [
                "rmse_test",
                "mse_test",
                "mae_test",
                "correlation_test",
                "cv_fold",
            ]

            if return_full_performance:
                train_results = model.summary(dataset="full")
                train_results["cv_fold"] = i + 1
                train_results["dataset"] = train_results.dataset.map({"full": "train"})
                test_results = pd.concat(
                    [test_results, train_results], ignore_index=True
                )

                user_train_results = model.user_results
                user_train_results.columns = [
                    "rmse_train",
                    "mse_train",
                    "mae_train",
                    "correlation_train",
                ]
                user_test_results = pd.concat(
                    [user_train_results, user_test_results], axis=1
                )

            if timeit:
                stop = time.time()
                duration = stop - start
                test_results["duration"] = duration
            return test_results, user_test_results

        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_folds)
        splits = list(split_train_test(data, n_folds, random_state=random_state))
        trains = [elem[0] for elem in splits]
        tests = [elem[1] for elem in splits]

        if parallelize:
            if verbose:
                print("Parallelizing across threads...")

            with ThreadPoolExecutor(max_threads) as threads:
                all_results = threads.map(
                    _run_sparse,
                    range(n_folds),
                    trains,
                    tests,
                    [algorithm] * n_folds,
                    seeds,
                    [return_full_performance] * n_folds,
                    [timeit] * n_folds,
                    [fit_kwargs] * n_folds,
                )
        else:
            all_results = map(
                _run_sparse,
                range(n_folds),
                trains,
                tests,
                [algorithm] * n_folds,
                seeds,
                [return_full_performance] * n_folds,
                [timeit] * n_folds,
                [fit_kwargs] * n_folds,
            )

        col_order = ["algorithm", "dataset", "cv_fold", "group", "metric", "score"]
        user_agg_drop_col = ["cv_fold"]

    # Collect results
    all_results = list(all_results)
    group_results = [elem[0] for elem in all_results]
    user_results = [
        elem[1].reset_index().rename(columns={"User": "user"}) for elem in all_results
    ]
    group_results = (
        pd.concat(group_results, ignore_index=True)
        .sort_values(by=["group", "dataset", "metric"])
        .reset_index(drop=True)
    )
    user_results = pd.concat(user_results, ignore_index=True)

    # Handle aggregation
    if return_agg:
        col_order = ["algorithm", "dataset", "group", "metric"] + agg_stats
        group_results = (
            group_results.groupby(["algorithm", "dataset", "group", "metric"])
            .score.agg(agg_stats)
            .reset_index()
            .sort_values(by=["dataset", "group", "metric"])
            .reset_index(drop=True)[col_order]
        )
        user_results = (
            user_results.groupby("user").mean().drop(columns=user_agg_drop_col)
        )
        return group_results, user_results
    else:
        if timeit:
            col_order += ["duration"]
        return group_results[col_order], user_results
