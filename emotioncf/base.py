"""
Base algorithm classes. All other algorithms inherit from these classes which means they have access to all their methods and attributes. You won't typically utilize these classes directly unless you're creating a custom estimator.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
import matplotlib.pyplot as plt
from .utils import create_train_test_mask
import warnings

__all__ = ["Base", "BaseNMF"]


class Base(object):
    """
    All other models and base classes inherit from this class.
    """

    def __init__(self, data, mask=None, n_mask_items=None, verbose=True):
        """
        Initialize a base collaborative filtering model

        Args:
            data (pd.DataFrame): users x items dataframe
            mask (pd.DataFrame, optional): A boolean dataframe used to split the data into training and testing. Defaults to None.
            n_mask_items (int/float, optional): number of items to mask out, while the rest are treated as observed; Defaults to None.
            verbose (bool; optional): print any initialization warnings; Default True

        Raises:
            ValueError: [description]
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe instance")
        self.data = data
        self.predictions = None
        self.is_fit = False
        self.mask = None
        self.is_masked = False
        self.masked_data = data
        self.is_mask_dilated = False
        self.dilated_mask = None
        self.dilated_by_nsamples = None
        self.n_mask_items = n_mask_items
        self.data_range = self.data.max().max() - self.data.min().min()
        self.is_dense = True
        self.results = None

        # Check for null values in input data and if they exist treat the data as already masked; check with Luke about this...
        if data.isnull().any().any():
            if verbose:
                print("data contains NaNs...treating as pre-masked")
            self.mask = ~data.isnull()
            self.masked_data = self.data[self.mask]
            self.is_masked = True
            self.is_dense = False

        # Otherwise apply any user provided mask or tell them data was pre-masked
        if mask is not None:
            if self.is_masked:
                raise ValueError(
                    "mask was provided, but data already contains missing values that were used for masking. This is an ambiguous operation. If a mask is provided data should not contain missing values!"
                )
            self.mask = mask
            self.masked_data = self.data[self.mask]
            self.is_masked = True

        # Same for n_train_items
        if self.n_mask_items is not None:
            if self.is_masked:
                raise ValueError(
                    "n_train_items was provided, but data already contains missing values that were used for masking. This is an ambiguous operation. If a n_train_items is provided data should not contain missing values!"
                )
            self.create_masked_data(n_mask_items=self.n_mask_items)

        if mask is None and n_mask_items is None and not self.is_masked and verbose:
            print(
                "Model initialized with dense data. Make sure to utilize `.create_masked_data` prior to fitting or reinitialize with a mask or n_mask_items"
            )

    def __repr__(self):
        out = f"{self.__class__.__module__}.{self.__class__.__name__}(data={self.data.shape}, is_fit={self.is_fit}, is_masked={self.is_masked}, is_mask_dilated={self.is_mask_dilated})"
        if self.is_mask_dilated:
            out += f", dilated_by_nsamples={self.dilated_by_nsamples}"
        return out

    def score(self, metric="rmse", by_subject=False, dataset="missing", verbose=True):
        """Get the performance of a fitted model by comparing observed and predicted data.

        Args:
            metric (str; optional): what metric to compute, one of 'rmse', 'mse', 'mae' or 'correlation'; Default 'rmse'.
            dataset (str; optional): how to compute scoring, either using 'observed', 'missing' or 'full'. Default 'missing'.

        Returns:
            float: score

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        if metric not in ["rmse", "mse", "mae", "correlation"]:
            raise ValueError(
                "metric must be one of 'rmse', 'mse', 'mae', or 'correlation'"
            )
        # Get dataframes of observed and predicted values
        # This will be a dense or sparse matrix the same shape as the input data
        actual, pred = self._retrieve_predictions(dataset)

        # TODO: implement by subject scoring
        if by_subject:
            raise NotImplementedError()
        else:
            if actual is not None:

                actual, pred = actual.to_numpy().flatten(), pred.to_numpy().flatten()

                if metric == "rmse":
                    return np.sqrt(np.nanmean((pred - actual) ** 2))
                elif metric == "mse":
                    return np.nanmean((pred - actual) ** 2)
                elif metric == "mae":
                    return np.nanmean(np.abs(pred - actual))
                elif metric == "correlation":
                    nans = np.logical_or(np.isnan(actual), np.isnan(pred))
                    return pearsonr(actual[~nans], pred[~nans])[0]
            else:
                if verbose:
                    warnings.warn(
                        "Cannot score predictions on missing data because true values were never observed!"
                    )
                return None

    def get_sub_corr(self, dataset="test"):
        """Calculate observed/predicted correlation for each subject in matrix

        Args:
            dataset (str): Get correlation on 'all' data, the 'train' data, or the 'test' data

        Returns:
            r (float): Correlation

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")
        if not self.is_predict:
            raise ValueError("You must predict() model first before using this method.")

        r = []
        # Note: the following mask prevents NaN values from being passed to `pearsonr()`.
        # However, it does not guaratee that no correlation values will be NaN, e.g. if only one
        # rating for a given subject is non-null in both test and train groups for a given
        # dataset, or variance is otherwise zero.
        if dataset == "all":
            noNanMask = (~np.isnan(self.data)) & (~np.isnan(self.predictions))
            for i in self.data.index:
                r.append(
                    pearsonr(
                        self.data.loc[i, :][noNanMask.loc[i, :]],
                        self.predictions.loc[i, :][noNanMask.loc[i, :]],
                    )[0]
                )
        elif self.is_mask:
            if dataset == "train":
                noNanMask = (~np.isnan(self.masked_data)) & (
                    ~np.isnan(self.predictions)
                )
                if self.is_mask_dilated:
                    for i in self.masked_data.index:
                        r.append(
                            pearsonr(
                                self.masked_data.loc[i, self.dilated_mask.loc[i, :]][
                                    noNanMask.loc[i, :]
                                ],
                                self.predictions.loc[i, self.dilated_mask.loc[i, :]][
                                    noNanMask.loc[i, :]
                                ],
                            )[0]
                        )
                else:
                    for i in self.masked_data.index:
                        r.append(
                            pearsonr(
                                self.masked_data.loc[i, self.train_mask.loc[i, :]][
                                    noNanMask.loc[i, :]
                                ],
                                self.predictions.loc[i, self.train_mask.loc[i, :]][
                                    noNanMask.loc[i, :]
                                ],
                            )[0]
                        )
            else:  # test
                noNanMask = (~np.isnan(self.data)) & (~np.isnan(self.predictions))
                for i in self.masked_data.index:
                    r.append(
                        pearsonr(
                            self.data.loc[i, ~self.train_mask.loc[i, :]][
                                noNanMask.loc[i, :]
                            ],
                            self.predictions.loc[i, ~self.train_mask.loc[i, :]][
                                noNanMask.loc[i, :]
                            ],
                        )[0]
                    )
        else:
            raise ValueError("Must run split_train_test() before using this option.")
        return np.array(r)

    def get_sub_mse(self, dataset="test"):
        """Calculate observed/predicted mse for each subject in matrix

        Args:
            dataset (str): Get mse on 'all' data, the 'train' data, or the 'test' data

        Returns:
            mse (float): mean squared error

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")
        if not self.is_predict:
            raise ValueError("You must predict() model first before using this method.")

        mse = []
        if dataset == "all":
            for i in self.data.index:
                actual = self.data.loc[i, :]
                pred = self.predictions.loc[i, :]
                mse.append(
                    np.nanmean(
                        (
                            pred[(~np.isnan(actual)) & (~np.isnan(pred))]
                            - actual[(~np.isnan(actual)) & (~np.isnan(pred))]
                        )
                        ** 2
                    )
                )
        elif self.is_mask:
            if dataset == "train":
                if self.is_mask_dilated:
                    for i in self.masked_data.index:
                        actual = self.masked_data.loc[i, self.dilated_mask.loc[i, :]]
                        pred = self.predictions.loc[i, self.dilated_mask.loc[i, :]]
                        mse.append(
                            np.nanmean(
                                (
                                    pred[(~np.isnan(actual)) & (~np.isnan(pred))]
                                    - actual[(~np.isnan(actual)) & (~np.isnan(pred))]
                                )
                                ** 2
                            )
                        )
                else:
                    for i in self.data.index:
                        actual = self.masked_data.loc[i, self.train_mask.loc[i, :]]
                        pred = self.predictions.loc[i, self.train_mask.loc[i, :]]
                        mse.append(
                            np.nanmean(
                                (
                                    pred[(~np.isnan(actual)) & (~np.isnan(pred))]
                                    - actual[(~np.isnan(actual)) & (~np.isnan(pred))]
                                )
                                ** 2
                            )
                        )
            else:
                for i in self.data.index:
                    actual = self.data.loc[i, ~self.train_mask.loc[i, :]]
                    pred = self.predictions.loc[i, ~self.train_mask.loc[i, :]]
                    mse.append(
                        np.nanmean(
                            (
                                pred[(~np.isnan(actual)) & (~np.isnan(pred))]
                                - actual[(~np.isnan(actual)) & (~np.isnan(pred))]
                            )
                            ** 2
                        )
                    )
        else:
            raise ValueError("Must run split_train_test() before using this option.")
        return np.array(mse)

    def create_masked_data(self, n_mask_items=0.1):
        """
        Create a mask and apply it to data using number of items or % of items

        Args:
            n_itmes (int/float, optional): if an integer is passed its raw value is used. Otherwise if a float is passed its taken to be a (rounded) percentage of the total items; Default .1 (10% of the data)
        """

        if (
            isinstance(n_mask_items, np.floating)
            and (n_mask_items >= 1.0 or n_mask_items <= 0.0)
        ) or (
            isinstance(n_mask_items, int)
            and (n_mask_items >= self.data.shape[1] or n_mask_items <= 0)
        ):
            raise TypeError(
                "n_items should a float between 0-1 or an integer < the number of items"
            )
        self.mask = create_train_test_mask(self.data, n_mask_items)
        self.masked_data = self.data[self.mask]
        self.is_masked = True
        self.n_mask_items = n_mask_items

    def plot_predictions(self, dataset="train", verbose=True, heatmapkwargs={}):
        """Create plot of actual and predicted data

        Args:
            dataset (str): plot 'all' data, the 'train' data, or the 'test' data
            verbose (bool; optional): print the averaged subject correlation while plotting; Default True

        Returns:
            r (float): Correlation

        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        if not self.is_predict:
            raise ValueError("You must predict() model first before using this method.")

        if self.is_mask:
            data = self.masked_data.copy()
        else:
            data = self.data.copy()

        heatmapkwargs.setdefault("square", False)
        heatmapkwargs.setdefault("xticklabels", False)
        heatmapkwargs.setdefault("yticklabels", False)
        vmax = (
            data.max().max()
            if data.max().max() > self.predictions.max().max()
            else self.predictions.max().max()
        )
        vmin = (
            data.min().min()
            if data.min().min() < self.predictions.min().min()
            else self.predictions.min().min()
        )

        heatmapkwargs.setdefault("vmax", vmax)
        heatmapkwargs.setdefault("vmin", vmin)

        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
        sns.heatmap(data, ax=ax[0], **heatmapkwargs)
        ax[0].set_title("Actual User/Item Ratings")
        ax[0].set_xlabel("Items", fontsize=18)
        ax[0].set_ylabel("Users", fontsize=18)
        sns.heatmap(self.predictions, ax=ax[1], **heatmapkwargs)
        ax[1].set_title("Predicted User/Item Ratings")
        ax[1].set_xlabel("Items", fontsize=18)
        ax[1].set_ylabel("Users", fontsize=18)
        f.tight_layout()

        actual, pred = self._retrieve_predictions(dataset)

        ax[2].scatter(
            actual[(~np.isnan(actual)) & (~np.isnan(pred))],
            pred[(~np.isnan(actual)) & (~np.isnan(pred))],
        )
        ax[2].set_xlabel("Actual Ratings")
        ax[2].set_ylabel("Predicted Ratings")
        ax[2].set_title("Predicted Ratings")

        r = self.get_sub_corr(dataset=dataset).mean()
        if verbose:
            print("Average Subject Correlation: %s" % r)

        return f, r

    def downsample(self, sampling_freq=None, target=None, target_type="samples"):

        """Downsample rating matrix to a new target frequency or number of samples using averaging.

        Args:
            sampling_freq (int/float):  Sampling frequency of data
            target (int/float): downsampling target
            target_type (str): type of target can be [samples,seconds,hz]

        """

        if sampling_freq is None:
            raise ValueError("Please specify the sampling frequency of the data.")
        if target is None:
            raise ValueError("Please specify the downsampling target.")
        if target_type is None:
            raise ValueError(
                "Please specify the type of target to downsample to [samples,seconds,hz]."
            )

        def ds(data, sampling_freq=sampling_freq, target=None, target_type="samples"):
            if target_type == "samples":
                n_samples = target
            elif target_type == "seconds":
                n_samples = target * sampling_freq
            elif target_type == "hz":
                n_samples = sampling_freq / target
            else:
                raise ValueError(
                    'Make sure target_type is "samples", "seconds", or "hz".'
                )

            data = data.T
            idx = np.sort(
                np.repeat(np.arange(1, data.shape[0] / n_samples, 1), n_samples)
            )
            if data.shape[0] > len(idx):
                idx = np.concatenate(
                    [idx, np.repeat(idx[-1] + 1, data.shape[0] - len(idx))]
                )
            return data.groupby(idx).mean().T

        self.data = ds(
            self.data,
            sampling_freq=sampling_freq,
            target=target,
            target_type=target_type,
        )

        if self.is_mask:
            self.train_mask = ds(
                self.train_mask,
                sampling_freq=sampling_freq,
                target=target,
                target_type=target_type,
            )
            self.train_mask.loc[:, :] = self.train_mask > 0
            self.masked_data = ds(
                self.masked_data,
                sampling_freq=sampling_freq,
                target=target,
                target_type=target_type,
            )
            if self.is_mask_dilated:
                self.dilated_mask = ds(
                    self.dilated_mask,
                    sampling_freq=sampling_freq,
                    target=target,
                    target_type=target_type,
                )
                self.dilated_mask.loc[:, :] = self.dilated_mask > 0

        if self.is_predict:
            self.predictions = ds(
                self.predictions,
                sampling_freq=sampling_freq,
                target=target,
                target_type=target_type,
            )

    def to_long_df(self):

        """ Create a long format pandas dataframe with observed, predicted, and mask."""

        observed = pd.DataFrame(columns=["Subject", "Item", "Rating", "Condition"])
        for row in self.data.iterrows():
            tmp = pd.DataFrame(columns=observed.columns)
            tmp["Rating"] = row[1]
            tmp["Item"] = self.data.columns
            tmp["Subject"] = row[0]
            tmp["Condition"] = "Observed"
            if self.is_masked:
                if self.is_mask_dilated:
                    tmp["Mask"] = self.dilated_mask.loc[row[0]]
                else:
                    tmp["Mask"] = self.mask.loc[row[0]]
            observed = observed.append(tmp)

        if self.is_fit:
            predicted = pd.DataFrame(columns=["Subject", "Item", "Rating", "Condition"])
            for row in self.predictions.iterrows():
                tmp = pd.DataFrame(columns=predicted.columns)
                tmp["Rating"] = row[1]
                tmp["Item"] = self.predictions.columns
                tmp["Subject"] = row[0]
                tmp["Condition"] = "Predicted"
                if self.is_masked:
                    tmp["Mask"] = self.mask.loc[row[0]]
                predicted = predicted.append(tmp)
            observed = observed.append(predicted)
        return observed

    def _retrieve_predictions(self, dataset):
        """Helper function to extract predicted values

        Args:
            dataset (str): should be one of 'full', 'observed', or 'missing''

        Returns:
            actual (array): true values
            predicted (array): predicted values
        """

        if dataset not in ["full", "observed", "missing"]:
            raise ValueError("dataset must be one of ['full','missing']")

        if dataset == "full":
            return (self.data, self.predictions)

        elif dataset == "observed":
            return (
                self.data[self.masked_data.notnull()],
                self.predictions[self.masked_data.notnull()],
            )
        elif dataset == "missing":
            # This happens if the input data already has NaNs that were not the result of a masking operation by a model on dense data
            if self.is_dense:
                return (
                    self.data[self.masked_data.isnull()],
                    self.predictions[self.masked_data.isnull()],
                )
            else:
                return (None, self.predictions[self.masked_data.isnull()])

    def _conv_ts_mean_overlap(self, sub_rating, n_samples=5):

        """Dilate each rating by n samples (centered).  If dilated samples are overlapping they will be averaged.

        Args:
            sub_rating (array): vector of data for subject
            n_samples (int):  number of samples to dilate each rating

        Returns:
            sub_rating_conv_mn (array): subject rating vector with each rating dilated n_samples (centered) with mean of overlapping

        """

        # Notes:  Could add custom filter input
        bin_sub_rating = ~sub_rating.isnull()
        if np.any(sub_rating.isnull()):
            sub_rating.fillna(0, inplace=True)
        filt = np.ones(n_samples)
        bin_sub_rating_conv = np.convolve(bin_sub_rating, filt)[: len(bin_sub_rating)]
        sub_rating_conv = np.convolve(sub_rating, filt)[: len(sub_rating)]
        sub_rating_conv_mn = deepcopy(sub_rating_conv)
        sub_rating_conv_mn[bin_sub_rating_conv >= 1] = (
            sub_rating_conv_mn[bin_sub_rating_conv >= 1]
            / bin_sub_rating_conv[bin_sub_rating_conv >= 1]
        )
        new_mask = bin_sub_rating_conv == 0
        sub_rating_conv_mn[new_mask] = np.nan
        return sub_rating_conv_mn

    def _dilate_ts_rating_samples(self, n_samples=None):

        """Alias for `.dilate_mask`"""

        if not self.is_masked and n_samples is not None:
            raise ValueError("Make sure model instance has been masked.")

        # Always reset to the undilated mask first
        self.masked_data = self.data[self.mask] if self.mask is not None else self.data

        if n_samples is not None:
            if isinstance(n_samples, np.floating) or (n_samples >= self.data.shape[1]):
                raise TypeError("nsamples should be an integer < the number of items")

            # Update masked data with dilation
            self.masked_data = self.masked_data.apply(
                lambda x: self._conv_ts_mean_overlap(x, n_samples=n_samples),
                axis=1,
                result_type="broadcast",
            )
            # Save dilated mask
            self.dilated_mask = ~self.masked_data.isnull()
            self.is_mask_dilated = True
            self.dilated_by_nsamples = n_samples
        else:
            self.dilated_mask = None
            self.is_mask_dilated = False
            self.dilated_by_nsamples = None

    def dilate_mask(self, n_samples=None):

        """Dilate sparse time-series data by n_samples.
        Overlapping data will be averaged. This method computes and stores the dilated mask in `.dilated_mask` and internally updates the `.masked_data`. Repeated calls to this method on the same model instance **do not** stack, but rather perform a new dilation on the original masked data. Called this method with `None` will undo any dilation. This is an alias to `._dilate_ts_rating_samples`

        Args:
            nsamples (int):  Number of samples to dilate data

        """

        self._dilate_ts_rating_samples(n_samples=n_samples)

    def fit(self):
        """Replaced by sub-classes. This call just ensures that a model's data is sparse prior to fitting"""
        if not self.is_masked:
            raise ValueError(
                "You're trying to fit on a dense matrix, because model data has not been masked! Either call the `.create_masked_data` method prior to fitting or re-initialize the model and set the `mask` or `n_mask_items` arguments."
            )

    def summary(self, verbose=True, long_df=False, return_cached=True):
        """Summary"""
        if not self.is_fit:
            raise ValueError("Model has not been fit!")
        if return_cached and self.results is not None:
            if verbose:
                print(
                    "Returning cached scores...set cache=False to force recomputation"
                )
            results = self.results
        else:
            results = {
                "algorithm": self.__class__.__name__,
                # "predictions": self.to_long_df(),
                "n_items": self.data.shape[1],
                "n_users": self.data.shape[0],
            }
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                for metric in ["rmse", "mse", "mae", "correlation"]:
                    out = {}
                    for dataset in ["full", "missing", "observed"]:
                        out[dataset] = self.score(
                            metric=metric, dataset=dataset, verbose=verbose
                        )
                    results[metric] = out
            results = pd.DataFrame(results)
            self.results = results
            if verbose and w:
                print(w[-1].message)
        if long_df:
            out = results[["algorithm", "rmse", "mse", "mae", "correlation"]]
            out = (
                out.reset_index()
                .melt(
                    id_vars=["index", "algorithm"],
                    var_name="metric",
                    value_name="score",
                )
                .rename(columns={"index": "dataset"})
                .sort_values(by=["dataset", "metric"])
                .reset_index(drop=True)
            )
        else:
            out = results

        return out

        # return Results(**results)


class BaseNMF(Base):
    """
    Base class for NMF algorithms.
    """

    def __init__(self, data, mask=None, n_mask_items=None, verbose=True):
        super().__init__(data, mask, n_mask_items, verbose)
        self.error_history = []

    def plot_learning(self, save=False):
        """
        Plot training error over iterations for diagnostic purposes

        Args:
            save (bool/str/Path, optional): if a string or path is provided will save the figure to that location. Defaults to False.

        Returns:
            tuple: (figure handle, axes handle)
        """

        if self.is_fit:
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            _ = ax.plot(range(1, len(self.error_history) + 1), self.error_history)
            ax.set(
                xlabel="Iteration",
                ylabel="Normalized RMSE",
                title=f"Final Normalized RMSE: {np.round(self._norm_rmse, 3)}\nConverged: {self.converged}",
            )
            if save:
                plt.savefig(save, bbox_inches="tight")
            return f, ax
        else:
            raise ValueError("Model has not been fit.")
