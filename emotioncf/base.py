import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy
import matplotlib.pyplot as plt
from .utils import create_sparse_mask, downsample_dataframe, check_random_state
import warnings
import seaborn as sns

__all__ = ["Base", "BaseNMF"]


class Base(object):
    """
    This is the base class for all model types.
    """

    def __init__(
        self,
        data,
        mask=None,
        n_mask_items=None,
        data_range=None,
        random_state=None,
        verbose=True,
    ):
        """
        Initialize a base collaborative filtering model

        Args:
            data (pd.DataFrame): users x items dataframe
            mask (pd.DataFrame, optional): A boolean dataframe used to split the data into 'observed' and 'missing' datasets. Defaults to None.
            n_mask_items (int/float, optional): number of items to mask out, while the rest are treated as observed; Defaults to None.
            data_range (int/float, optional): max - min of the data; Default computed from the input data. This is useful to set manually in case the input data do not span the full range of possible values
            random_state (None, int, RandomState): a seed or random state used for all internal random operations (e.g. randomly mask half the data given n_mask_item = .05). Passing None will generate a new random seed. Default None.
            verbose (bool; optional): print any initialization warnings; Default True

        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe instance")
        self.data = data.copy()
        self.predictions = None
        self.is_fit = False
        self.mask = None  # boolean matrix of observed value indices
        self.is_masked = False
        self.masked_data = data  # boolean mask applied to data
        self.is_mask_dilated = False
        self.dilated_mask = None  # booleanized masked_data after dilation
        self.dilated_by_nsamples = None
        self.n_mask_items = n_mask_items
        self.data_range = (
            self.data.max().max() - self.data.min().min()
            if data_range is None
            else data_range
        )
        self.is_dense = True
        self.overall_results = None
        self.user_results = None
        self.n_users = data.shape[0]
        self.n_items = data.shape[1]
        self.random_state = check_random_state(random_state)

        # Check for null values in input data and if they exist treat the data as already masked
        if self.data.isnull().any().any():
            if verbose:
                print("data contains NaNs...treating as pre-masked")
            self.mask = ~self.data.isnull()
            self.is_masked = True
            self.is_dense = False

        # Otherwise apply any user provided mask or tell them data was pre-masked
        if mask is not None:
            if self.is_masked:
                raise ValueError(
                    "mask was provided, but data already contains missing values that were used for masking. This is an ambiguous operation. If a mask is provided data should not contain missing values!"
                )
            if mask.shape != data.shape:
                raise ValueError("mask must be the same shape as data")
            self.mask = mask.copy()
            self.masked_data = self.data[self.mask]
            self.is_masked = True

        # Same for n_train_items
        if self.n_mask_items is not None:
            if self.is_masked:
                raise ValueError(
                    "n_mask_items was provided, but data already contains missing values that were used for masking. This is an ambiguous operation. If a n_mask_items is provided data should not contain missing values!"
                )
            self.create_masked_data(n_mask_items=self.n_mask_items)

        if mask is None and n_mask_items is None and not self.is_masked and verbose:
            print(
                "Model initialized with dense data. Make sure to utilize `.create_masked_data` prior to fitting or reinitialize with a mask or n_mask_items"
            )

    def __repr__(self):
        out = f"{self.__class__.__module__}.{self.__class__.__name__}(n_users={self.n_users}, n_items={self.n_items}, is_fit={self.is_fit}, is_masked={self.is_masked}, is_mask_dilated={self.is_mask_dilated}"
        if self.is_mask_dilated:
            out += f", dilated_by_nsamples={self.dilated_by_nsamples}"
        out += ")"
        return out

    def score(
        self,
        metric="rmse",
        dataset="missing",
        by_user=True,
        actual=None,
    ):
        """Get the performance of a fitted model by comparing observed and predicted data. This method is primarily useful if you want to calculate a single metric. Otherwise you should prefer the `.summary()` method instead, which scores all metrics.

        Args:
            metric (str; optional): what metric to compute, one of 'rmse', 'mse', 'mae' or 'correlation'; Default 'rmse'.
            dataset (str; optional): how to compute scoring, either using 'observed', 'missing' or 'full'. Default 'missing'.
            by_user (bool; optional): whether to return a single score over all data points or a pandas Series of scores per user. Default True.
            actual (pd.DataFrame, None; optional): a dataframe to score against; Default is None which uses the data provided when the model was initialized

        Returns:
            float/pd.Series: score

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        if metric not in ["rmse", "mse", "mae", "correlation"]:
            raise ValueError(
                "metric must be one of 'rmse', 'mse', 'mae', or 'correlation'"
            )
        # Get dataframes of observed and predicted values
        # This will be a dense or sparse matrix the same shape as the input data
        model_actual, pred = self._retrieve_predictions(dataset)

        if actual is None:
            actual = model_actual
        else:
            if actual.shape != self.data.shape:
                raise ValueError(
                    "actual values dataframe supplied but shape does not match original data"
                )

        if actual is None:
            warnings.warn(
                "Cannot score predictions on missing data because true values were never observed!"
            )
            return None

        with warnings.catch_warnings():
            # Catch 'Mean of empty slice' warnings from np.nanmean
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if by_user:
                scores = []
                for userid in range(actual.shape[0]):
                    user_actual = actual.iloc[userid, :].values
                    user_pred = pred.iloc[userid, :].values
                    if metric == "rmse":
                        score = np.sqrt(np.nanmean((user_pred - user_actual) ** 2))
                    elif metric == "mse":
                        score = np.nanmean((user_pred - user_actual) ** 2)
                    elif metric == "mae":
                        score = np.nanmean(np.abs(user_pred - user_actual))
                    elif metric == "correlation":
                        nans = np.logical_or(np.isnan(user_actual), np.isnan(user_pred))
                        if len(user_actual[~nans]) < 2 or len(user_pred[~nans]) < 2:
                            score = np.nan
                        else:
                            score = pearsonr(user_actual[~nans], user_pred[~nans])[0]
                    scores.append(score)
                return pd.Series(scores, index=actual.index, name=f"{metric}_{dataset}")
            else:
                actual, pred = actual.to_numpy().flatten(), pred.to_numpy().flatten()

                if metric == "rmse":
                    return np.sqrt(np.nanmean((pred - actual) ** 2))
                elif metric == "mse":
                    return np.nanmean((pred - actual) ** 2)
                elif metric == "mae":
                    return np.nanmean(np.abs(pred - actual))
                elif metric == "correlation":
                    nans = np.logical_or(np.isnan(actual), np.isnan(pred))
                    if len(actual[~nans]) < 2 or len(pred[~nans]) < 2:
                        return np.nan
                    else:
                        return pearsonr(actual[~nans], pred[~nans])[0]

    def create_masked_data(self, n_mask_items=0.2):
        """
        Create a mask and apply it to data using number of items or % of items

        Args:
            n_items (int/float, optional): if an integer is passed its raw value is used. Otherwise if a float is passed its taken to be a (rounded) percentage of the total items; Default 0.1 (10% of the data)
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
        self.mask = create_sparse_mask(
            self.data, n_mask_items, random_state=self.random_state
        )
        self.masked_data = self.data[self.mask]
        self.is_masked = True
        self.n_mask_items = n_mask_items

    def plot_predictions(
        self,
        dataset="missing",
        figsize=(16, 8),
        label_fontsize=16,
        hide_title=False,
        heatmap_kwargs={},
    ):
        """Create plot of actual vs predicted values.

        Args:
            dataset (str; optional): one of 'full', 'observed', or 'missing'. Default 'missing'.
            figsize (tuple; optional): matplotlib figure size; Default (16,8)
            label_fontsize (int; optional): fontsize for all axis labels and titles; Default 16
            hide_title (bool; optional): hide title containing RMSE and correlation performance if available; Default False
            heatmap_kwargs (dict, optional): addition arguments to seaborn.heatmap.

        Returns:
            tuple: (figure handle, axis handle)

        """

        if not self.is_fit:
            raise ValueError("Model has not been fit")

        vmax = max(self.data.max().max(), self.data.max().max())
        vmin = min(self.data.min().min(), self.data.min().min())

        actual, pred = self._retrieve_predictions(dataset)

        if actual is None:
            ncols = 2
            warnings.warn(
                "Cannot score predictions on missing data because true values were never observed!"
            )
        else:
            ncols = 3

        heatmap_kwargs.setdefault("square", False)
        heatmap_kwargs.setdefault("xticklabels", False)
        heatmap_kwargs.setdefault("yticklabels", False)
        heatmap_kwargs.setdefault("vmax", vmax)
        heatmap_kwargs.setdefault("vmin", vmin)

        f, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

        # The original data matrix (potentially masked)
        sns.heatmap(self.masked_data, ax=ax[0], **heatmap_kwargs)
        ax[0].set_title("Actual User/Item Ratings", fontsize=label_fontsize)
        ax[0].set_xlabel("Items", fontsize=label_fontsize)
        ax[0].set_ylabel("Users", fontsize=label_fontsize)

        # The predicted data matrix
        sns.heatmap(self.predictions, ax=ax[1], **heatmap_kwargs)
        ax[1].set_title("Predicted User/Item Ratings", fontsize=label_fontsize)
        ax[1].set_xlabel("Items", fontsize=label_fontsize)
        ax[1].set_ylabel("Users", fontsize=label_fontsize)
        f.tight_layout()

        # Scatter plot if we can calculate it
        if actual is not None:
            nans = np.logical_or(np.isnan(actual), np.isnan(pred))
            ax[2].scatter(
                actual[~nans],
                pred[~nans],
            )
            ax[2].set_xlabel("Actual", fontsize=label_fontsize)
            ax[2].set_ylabel("Predicted", fontsize=label_fontsize)
            ax[2].set_title("Ratings", fontsize=label_fontsize)
            sns.despine()

            r = self.score(dataset=dataset, by_user=True, metric="correlation")
            rmse = self.score(dataset=dataset, by_user=True, metric="rmse")
            if not hide_title:
                plt.suptitle(
                    f"Mean RMSE: {np.round(rmse.mean(),3)} +/- {np.round(rmse.std(), 3)}\nMean Correlation: {np.round(r.mean(), 3)} +/- {np.round(r.std(), 3)}",
                    y=1.07,
                    fontsize=label_fontsize + 2,
                )
            plt.subplots_adjust(wspace=0.2)

        return f, ax

    def downsample(self, n_samples, sampling_freq=None, target_type="samples"):

        """
        Downsample a model's rating matrix to a new target frequency or number of samples using averaging. Also downsamples a model's mask and dilated mask if they exist as well as a model's predictions if it's already been fit.

        If target_type = 'samples' and sampling_freq is None, the new user x item matrix will have shape users x items * (1 / n_samples).

        If target_type = 'seconds', the new user x item matrix will have shape users x items * (1 / n_samples * sampling_freq).

        If target_type = 'hz', the new user x item matrix will have shape users x items * (1 / sampling_freq / n_samples).

        Args:
            n_samples (int): number of samples
            sampling_freq (int/float):  Sampling frequency of data; Default None
            target_type (str, optional): how to downsample; must be one of "samples", "seconds" or "hz". Defaults to "samples".

        """

        self.data = downsample_dataframe(
            self.data,
            sampling_freq=sampling_freq,
            n_samples=n_samples,
            target_type=target_type,
        )

        if self.is_masked:
            # Also downsample mask
            self.mask = downsample_dataframe(
                self.mask,
                sampling_freq=sampling_freq,
                n_samples=n_samples,
                target_type=target_type,
            )
            # Ensure mask stays boolean
            self.mask.loc[:, :] = self.mask > 0

            # Masked data
            self.masked_data = downsample_dataframe(
                self.masked_data,
                sampling_freq=sampling_freq,
                n_samples=n_samples,
                target_type=target_type,
            )
            # Dilated mask
            if self.is_mask_dilated:
                self.dilated_mask = downsample_dataframe(
                    self.dilated_mask,
                    sampling_freq=sampling_freq,
                    n_samples=n_samples,
                    target_type=target_type,
                )
                # Ensure mask stays boolean
                self.dilated_mask.loc[:, :] = self.dilated_mask > 0

        if self.is_fit:
            self.predictions = downsample_dataframe(
                self.predictions,
                sampling_freq=sampling_freq,
                n_samples=n_samples,
                target_type=target_type,
            )

    def to_long_df(self):

        """Create a long format pandas dataframe with observed, predicted, and mask."""

        observed = pd.DataFrame(columns=["User", "Item", "Rating", "Condition"])
        for row in self.data.iterrows():
            tmp = pd.DataFrame(columns=observed.columns)
            tmp["Rating"] = row[1]
            tmp["Item"] = self.data.columns
            tmp["User"] = row[0]
            tmp["Condition"] = "Observed"
            if self.is_masked:
                if self.is_mask_dilated:
                    tmp["Mask"] = self.dilated_mask.loc[row[0]]
                else:
                    tmp["Mask"] = self.mask.loc[row[0]]
            observed = observed.append(tmp)

        if self.is_fit:
            predicted = pd.DataFrame(columns=["User", "Item", "Rating", "Condition"])
            for row in self.predictions.iterrows():
                tmp = pd.DataFrame(columns=predicted.columns)
                tmp["Rating"] = row[1]
                tmp["Item"] = self.predictions.columns
                tmp["User"] = row[0]
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
            raise ValueError("dataset must be one of ['full', 'observed', 'missing']")

        if dataset == "full":
            return (self.data, self.predictions)

        # NOTE: always use self.mask to grab locations of observed and missing values rather than where masked_data is or isn't null. This is because dilation will by design *reduce* the sparsity of self.masked_data, meaning non-null indices are only a subset of the true missing values
        elif dataset == "observed":
            return (
                self.data[self.mask],
                self.predictions[self.mask],
            )
        elif dataset == "missing":
            if self.is_dense:
                return (
                    self.data[~self.mask],
                    self.predictions[~self.mask],
                )
            else:
                # This happens if the input data already has NaNs that were not the result of a masking operation by a model on dense data. In this case we never observed ground truth values to compare against predictions
                return (None, self.predictions[~self.mask])

    @staticmethod
    def _conv_ts_mean_overlap(sub_rating, n_samples=5):

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

    def dilate_mask(self, n_samples=None):

        """Dilate sparse time-series data by n_samples.
        Overlapping data will be averaged. This method computes and stores the dilated mask in `.dilated_mask` and internally updates the `.masked_data`. Repeated calls to this method on the same model instance **do not** stack, but rather perform a new dilation on the original masked data. Called this method with `None` will undo any dilation.

        Args:
            nsamples (int):  Number of samples to dilate data

        """

        if self.mask is None:
            raise ValueError("Model has no mask and requires one to perform dilation")
        if not self.is_masked and n_samples is not None:
            raise ValueError("Make sure model instance has been masked.")

        if isinstance(n_samples, np.floating) or (
            n_samples is not None and n_samples >= self.data.shape[1]
        ):
            raise TypeError("nsamples should be an integer < the number of items")

        # Always reset to the undilated mask first
        self.masked_data = self.data[self.mask]

        if n_samples is not None:
            # After masking, perform dilation and save as the new masked data
            self.masked_data = self.masked_data.apply(
                lambda x: self._conv_ts_mean_overlap(x, n_samples=n_samples),
                axis=1,
                result_type="broadcast",
            )
            # Calculate and save dilated mask
            self.dilated_mask = ~self.masked_data.isnull()
            self.is_mask_dilated = True
            self.dilated_by_nsamples = n_samples
        else:
            self.dilated_mask = None
            self.is_mask_dilated = False
            self.dilated_by_nsamples = None

    def fit(self, **kwargs):
        """Replaced by sub-classes. This call just ensures that a model's data is sparse prior to fitting"""
        if not self.is_masked:
            raise ValueError(
                "You're trying to fit on a dense matrix, because model data has not been masked! Either call the `.create_masked_data` method prior to fitting or re-initialize the model and set the `mask` or `n_mask_items` arguments."
            )
        if kwargs.get("dilate_by_nsamples", None) and self.is_mask_dilated:
            warnings.warn(
                ".fit() was called with dilate_by_nsamples=None, but model mask is already dilated! This will undo dilation and then fit a model. Instead pass dilate_by_nsamples, directly to .fit()"
            )

    def summary(self, verbose=False, actual=None, dataset=None):
        """
        Calculate the performance of a model and return a dataframe of results. Computes performance across all, observed, and missing datasets. Scores using rmse, mse, mae, and correlation. Computes scores across all subjects (i.e. ignoring the fact that ratings are clustered by subject) and the mean performance for each metric after calculating per-subject performance.

        Args:
            verbose (bool, optional): Print warning messages during scoring. Defaults to False.
            actual (pd.DataFrame, None; optional): a dataframe to score against; Default is None which uses the data provided when the model was initialized
            dataset (str/None): dataset to score. Must be one of 'full', 'observed','missing' or None to score both 'observed' and 'missing'; Default None

        Returns:
            pd.DataFrame: long-form dataframe of model performance
        """

        if not self.is_fit:
            raise ValueError("Model has not been fit!")

        if dataset is None:
            if actual is None:
                if self.is_dense:
                    dataset = ["missing", "observed"]
                else:
                    dataset = ["observed"]
            else:
                dataset = ["missing", "observed"]

        elif isinstance(dataset, str):
            if actual is None and not self.is_dense and dataset in ["full", "missing"]:
                raise ValueError(
                    "Cannot score predictions on missing values because no ground truth was observed"
                )
            dataset = [dataset]

        # Compute results for all metrics, all datasets, separately for group and by subject
        group_results = {
            "algorithm": self.__class__.__name__,
        }
        subject_results = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for metric in ["rmse", "mse", "mae", "correlation"]:
                this_group_result = {}
                this_subject_result = []
                for dat in dataset:
                    this_group_result[dat] = self.score(
                        metric=metric, dataset=dat, actual=actual, by_user=False
                    )
                    this_subject_result.append(
                        self.score(
                            metric=metric,
                            dataset=dat,
                            by_user=True,
                            actual=actual,
                        )
                    )
                # Dict of group results for this metric
                group_results[metric] = this_group_result
                # Dataframe of subject results for this metric
                this_subject_result = pd.concat(this_subject_result, axis=1)
                subject_results.append(this_subject_result)
                group_results[f"{metric}_user"] = dict(
                    zip(
                        dataset,
                        this_subject_result.mean().values,
                    )
                )
        # Save final results to longform df
        self.user_results = pd.concat(subject_results, axis=1)
        group_results = pd.DataFrame(group_results)
        group_results = (
            group_results.reset_index()
            .melt(
                id_vars=["index", "algorithm"],
                var_name="metric",
                value_name="score",
            )
            .rename(columns={"index": "dataset"})
            .sort_values(by=["dataset", "metric"])
            .reset_index(drop=True)
            .assign(
                group=lambda df: df.metric.apply(
                    lambda x: "user" if "user" in x else "all"
                ),
                metric=lambda df: df.metric.replace(
                    {
                        "correlation_user": "correlation",
                        "mse_user": "mse",
                        "rmse_user": "rmse",
                        "mae_user": "mae",
                    }
                ),
            )
            .sort_values(by=["dataset", "group", "metric"])
            .reset_index(drop=True)[
                ["algorithm", "dataset", "group", "metric", "score"]
            ]
        )
        self.overall_results = group_results
        if verbose:
            if w:
                print(w[-1].message)
            print(
                "User performance results (not returned) are accessible using .user_results"
            )
            print(
                "Overall performance results (returned) are accesible using .overall_results"
            )

        return group_results

    def transform(self, return_only_predictions=False):
        """
        Return a user x item matrix of predictions after a model has been fit

        Args:
            return_only_predictions (bool, optional): Returns both training and testing predictions rather than simply filling in missing values with predictions. Defaults to False.

        Returns:
            pd.DataFrame: user x item ratings
        """

        if not self.is_fit:
            raise ValueError("Model has not been fit!")
        if return_only_predictions:
            return self.predictions
        else:
            # Propagate observed values to return object
            out = self.data[self.mask]
            # Fill in missing values with predictions
            out[~self.mask] = self.predictions[~self.mask]
            return out


class BaseNMF(Base):
    """
    Base class for NMF algorithms.
    """

    def __init__(
        self, data, mask=None, n_mask_items=None, verbose=True, random_state=None
    ):
        super().__init__(
            data, mask, n_mask_items, random_state=random_state, verbose=verbose
        )
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
            sns.despine()
            if save:
                plt.savefig(save, bbox_inches="tight")
            return f, ax
        else:
            raise ValueError("Model has not been fit.")

    def plot_factors(self, save=False, **kwargs):
        """
        Plot user x factor and item x factor matrices

        Args:
            save (bool/str/Path, optional): if a string or path is provided will save the figure to that location. Defaults to False.
            kwargs: additional arguments to seaborn.heatmap

        Returns:
            tuple: (figure handle, axes handle)
        """

        if self.is_fit:
            f, axs = plt.subplots(1, 2, figsize=(12, 6))
            if hasattr(self, "W"):
                _ = sns.heatmap(self.W, ax=axs[0], **kwargs)
                _ = sns.heatmap(self.H, ax=axs[1], **kwargs)
            else:
                _ = sns.heatmap(self.user_vecs, ax=axs[0], **kwargs)
                _ = sns.heatmap(self.item_vecs, ax=axs[1], **kwargs)
            axs[0].set_xlabel("Factor", fontsize=18)
            axs[0].set_ylabel("User", fontsize=18)
            axs[0].set_title("User Factors", fontsize=18)
            axs[1].set_xlabel("Item", fontsize=18)
            axs[1].set_ylabel("Factor", fontsize=18)
            axs[1].set_title("Item Factors", fontsize=18)
            if save:
                plt.savefig(save, bbox_inches="tight")
            return f, axs
        else:
            raise ValueError("Model has not been fit.")
