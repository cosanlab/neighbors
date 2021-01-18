"""
Classes that perform various types of collaborative filtering

"""

from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy

__all__ = ["Mean", "KNN", "NNMF_mult", "NNMF_sgd"]


class BaseCF(object):

    """ Base Collaborative Filtering Class """

    def __init__(self, data, mask=None, n_train_items=None):
        """
        Initialize a base collaborative filtering model

        Args:
            data (pd.DataFrame): users x items dataframe
            mask ([type], optional): [description]. Defaults to None.
            n_train_items ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe instance")
        self.data = data
        self.predictions = None
        self.is_fit = False
        self.is_predict = False
        self.is_mask_dilated = False
        self.dilated_mask = None
        if mask is not None:
            self.train_mask = mask
            self.masked_data = self.data[self.train_mask]
            self.is_mask = True
        elif self.data.isnull().any().any():
            self.train_mask = ~self.data.isnull()
            self.masked_data = self.data[self.train_mask]
            self.is_mask = True
        else:
            self.is_mask = False

        if n_train_items is not None:
            self.split_train_test(n_train_items=n_train_items)

    def __repr__(self):
        return "%s(rating=%s)" % (self.__class__.__name__, self.data.shape)

    def get_mse(self, dataset="all"):
        """Get overall mean squared error for predicted compared to actual for all items and subjects.

        Args:
            dataset (str): Get mse on 'all' data, the 'train' data, or the 'test' data

        Returns:
            mse (float): mean squared error

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")
        if not self.is_predict:
            raise ValueError("You must predict() model first before using this method.")

        actual, pred = self._retrieve_predictions(dataset)

        return np.mean((pred - actual) ** 2)

    def get_corr(self, dataset="all"):
        """Get overall correlation for predicted compared to actual for all items and subjects.

        Args:
            dataset (str): Get correlation on 'all' data, the 'train' data, or the 'test' data

        Returns:
            r (float): Correlation
        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        if not self.is_predict:
            raise ValueError("You must predict() model first before using this method.")

        actual, pred = self._retrieve_predictions(dataset)

        return pearsonr(actual, pred)[0]

    def get_sub_corr(self, dataset="all"):
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

    def get_sub_mse(self, dataset="all"):
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
                    np.mean(
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
                            np.mean(
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
                            np.mean(
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
                        np.mean(
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

    def split_train_test(self, n_train_items=0.1):
        """
        Split data into training and testing sets

        Args:
            n_train_items (int/float, optional): if an integer is passed its raw value is used. Otherwise if a float is passed its taken to be a (rounded) percentage of the total items; Default .1 (10% of the data)
        """

        if isinstance(n_train_items, (float, np.floating)) and 1 >= n_train_items > 0:
            self.n_train_items = int(np.round(self.data.shape[1] * n_train_items))

        elif isinstance(n_train_items, (int, np.integer)):
            self.n_train_items = n_train_items

        else:
            raise TypeError(
                f"n_train_items must be an integer or a float between 0-1, not {type(n_train_items)} with value {n_train_items}"
            )

        self.train_mask = self.data.copy()
        self.train_mask.loc[:, :] = np.zeros(self.data.shape).astype(bool)

        for sub in self.data.index:
            sub_train_rating_item = np.random.choice(
                self.data.columns, replace=False, size=self.n_train_items
            )
            self.train_mask.loc[sub, sub_train_rating_item] = True

        self.masked_data = self.data[self.train_mask]
        self.is_mask = True

    def plot_predictions(self, dataset="train", heatmapkwargs={}):
        """Create plot of actual and predicted data

        Args:
            data (str): plot 'all' data, the 'train' data, or the 'test' data

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

        r = self.get_corr(dataset=dataset)
        print("Correlation: %s" % r)

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
            if self.is_mask:
                if self.is_mask_dilated:
                    tmp["Mask"] = self.dilated_mask.loc[row[0]]
                else:
                    tmp["Mask"] = self.train_mask.loc[row[0]]
            observed = observed.append(tmp)

        if self.is_predict:
            predicted = pd.DataFrame(columns=["Subject", "Item", "Rating", "Condition"])
            for row in self.predictions.iterrows():
                tmp = pd.DataFrame(columns=predicted.columns)
                tmp["Rating"] = row[1]
                tmp["Item"] = self.predictions.columns
                tmp["Subject"] = row[0]
                tmp["Condition"] = "Predicted"
                if self.is_mask:
                    tmp["Mask"] = self.train_mask.loc[row[0]]
                predicted = predicted.append(tmp)
            observed = observed.append(predicted)
        return observed

    def _retrieve_predictions(self, dataset):
        """Helper function to extract predicted values

        Args:
            dataset (str): can be ['all', 'train', 'test']

        Returns:
            actual (array): true values
            predicted (array): predicted values
        """

        if dataset not in ["all", "train", "test"]:
            raise ValueError("data must be ['all','train','test']")

        if dataset == "all":
            if self.is_mask:
                if self.is_mask_dilated:
                    actual = self.masked_data.values[self.dilated_mask]
                    predicted = self.predictions.values[self.dilated_mask]
                else:
                    actual = self.masked_data.values[self.train_mask]
                    predicted = self.predictions.values[self.train_mask]
            else:
                actual = self.data.values.flatten()
                predicted = self.predictions.values.flatten()
        elif self.is_mask:
            if dataset == "train":
                actual = self.masked_data.values[self.train_mask]
                predicted = self.predictions.values[self.train_mask]
            else:  # test
                actual = self.data.values[~self.train_mask]
                predicted = self.predictions.values[~self.train_mask]
                if np.all(np.isnan(actual)):
                    raise ValueError(
                        "No test data available. Use data='all' or 'train'"
                    )
        else:
            raise ValueError("Must run split_train_test() before using this option.")

        return actual, predicted

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

        """Helper function to dilate sparse time-series data by n_samples.
        Overlapping data will be averaged. Will update mask with new values.

        Args:
            n_samples (int):  Number of samples to dilate data

        Returns:
            masked_data (Dataframe): dataframe instance that has been dilated by n_samples
        """

        if n_samples is None:
            raise ValueError("Please specify number of samples to dilate.")

        if not self.is_mask:
            raise ValueError("Make sure cf instance has been masked.")

        self.masked_data = self.data[self.train_mask]
        self.masked_data = self.masked_data.apply(
            lambda x: self._conv_ts_mean_overlap(x, n_samples=n_samples),
            axis=1,
            result_type="broadcast",
        )
        self.dilated_mask = ~self.masked_data.isnull()
        self.is_mask_dilated = True
        return self.masked_data


class Mean(BaseCF):

    """ CF using Item Mean across subjects"""

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.mean = None

    def fit(self, dilate_ts_n_samples=None, **kwargs):

        """Fit collaborative model to train data.  Calculate similarity between subjects across items

        Args:
            metric (str): type of similarity {"correlation","cosine"}
            dilate_ts_n_samples (int): will dilate masked samples by n_samples to leverage auto-correlation in estimating time-series data

        """

        if self.is_mask:
            if dilate_ts_n_samples is not None:
                _ = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)
                self.mean = self.masked_data[self.dilated_mask].mean(
                    skipna=True, axis=0
                )
            else:
                self.mean = self.masked_data[self.train_mask].mean(skipna=True, axis=0)
        else:
            self.mean = self.data.mean(skipna=True, axis=0)
        self.is_fit = True

    def predict(self, **kwargs):

        """Predict missing items using other subject's item means.

        Args:
            k (int): number of closest neighbors to use

        Returns:
            predicted_rating (Dataframe): adds field to object instance

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        self.predictions = self.data.copy()
        for row in self.data.iterrows():
            self.predictions.loc[row[0]] = self.mean
        self.is_predict = True


class KNN(BaseCF):

    """ K-Nearest Neighbors CF algorithm"""

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.subject_similarity = None

    def fit(self, metric="pearson", dilate_ts_n_samples=None, **kwargs):

        """Fit collaborative model to train data.  Calculate similarity between subjects across items

        Args:
            metric (str): type of similarity {"pearson",,"spearman","correlation","cosine"}.  Note pearson and spearman are way faster.
        """

        if self.is_mask:
            data = self.data[self.train_mask]
        else:
            data = self.data.copy()

        if dilate_ts_n_samples is not None:
            data = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)
            data = data[self.dilated_mask]

        def cosine_similarity(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        if metric in ["pearson", "kendall", "spearman"]:
            sim = data.T.corr(method=metric)
        elif metric in ["correlation", "cosine"]:
            sim = pd.DataFrame(np.zeros((data.shape[0], data.shape[0])))
            sim.columns = data.index
            sim.index = data.index
            for x in data.iterrows():
                for y in data.iterrows():
                    if metric == "correlation":
                        sim.loc[x[0], y[0]] = pearsonr(
                            x[1][(~x[1].isnull()) & (~y[1].isnull())],
                            y[1][(~x[1].isnull()) & (~y[1].isnull())],
                        )[0]
                    elif metric == "cosine":
                        sim.loc[x[0], y[0]] = cosine_similarity(
                            x[1][(~x[1].isnull()) & (~y[1].isnull())],
                            y[1][(~x[1].isnull()) & (~y[1].isnull())],
                        )
        else:
            raise NotImplementedError(
                "%s is not implemented yet. Try ['pearson','spearman','correlation','cosine']"
                % metric
            )
        self.subject_similarity = sim
        self.is_fit = True

    def predict(self, k=None, **kwargs):
        """Predict Subject's missing items using similarity based collaborative filtering.

        Args:
            k (int): number of closest neighbors to use

        Returns:
            predicted_rating (Dataframe): adds field to object instance

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        if self.is_mask:
            data = self.masked_data.copy()
        else:
            data = self.data.copy()

        pred = pd.DataFrame(np.zeros(data.shape))
        pred.columns = data.columns
        pred.index = data.index
        for row in data.iterrows():
            if k is not None:
                top_subjects = (
                    self.subject_similarity.loc[row[0]]
                    .drop(row[0])
                    .sort_values(ascending=False)[0:k]
                )
            else:
                top_subjects = (
                    self.subject_similarity.loc[row[0]]
                    .drop(row[0])
                    .sort_values(ascending=False)
                )
            top_subjects = top_subjects[~top_subjects.isnull()]  # remove nan subjects
            for col in data.iteritems():
                pred.loc[row[0], col[0]] = np.dot(
                    top_subjects, self.data.loc[top_subjects.index, col[0]].T
                ) / len(top_subjects)
        self.predictions = pred
        self.is_predict = True


class NNMF_mult(BaseCF):
    """Train non negative matrix factorization model using multiplicative updates.
    Allows masking to only learn the train weights.

    Based on http://stackoverflow.com/questions/22767695/
    python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat

    """

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.H = None
        self.W = None

    def fit(
        self,
        n_factors=None,
        max_iterations=100,
        error_limit=1e-6,
        fit_error_limit=1e-6,
        verbose=False,
        dilate_ts_n_samples=None,
        **kwargs,
    ):

        """Fit NNMF collaborative filtering model to train data using multiplicative updating.

        Args:
            n_factors (int): Number of factors or components
            max_iterations (int):  maximum number of interations (default=100)
            error_limit (float): error tolerance (default=1e-6)
            fit_error_limit (float): fit error tolerance (default=1e-6)
            verbose (bool): verbose output during fitting procedure (default=True)
            dilate_ts_n_samples (int): will dilate masked samples by n_samples to leverage auto-correlation in estimating time-series data

        """

        eps = 1e-5

        n_users, n_items = self.data.shape

        if n_factors is None:
            n_factors = n_items

        # Initial guesses for solving X ~= WH. H is random [0,1] scaled by sqrt(X.mean() / n_factors)
        avg = np.sqrt(np.nanmean(self.data) / n_factors)
        self.H = avg * np.random.rand(n_items, n_factors)  # H = Y
        self.W = avg * np.random.rand(n_users, n_factors)  # W = A

        if self.is_mask:
            if dilate_ts_n_samples is not None:
                masked_X = self._dilate_ts_rating_samples(
                    n_samples=dilate_ts_n_samples
                ).values
                mask = self.dilated_mask.values
            else:
                mask = self.train_mask.values
                masked_X = self.data.values * mask
            masked_X[np.isnan(masked_X)] = 0
        else:
            masked_X = self.data.values
            mask = np.ones(self.data.shape)

        X_est_prev = np.dot(self.W, self.H)

        ctr = 1
        fit_residual = 100
        while ctr <= max_iterations or fit_residual < fit_error_limit:
            # while ctr <= max_iterations or curRes < error_limit or fit_residual < fit_error_limit:
            # Update W: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
            self.W *= np.dot(masked_X, self.H.T) / np.dot(
                mask * np.dot(self.W, self.H), self.H.T
            )
            self.W = np.maximum(self.W, eps)

            # Update H: Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
            self.H *= np.dot(self.W.T, masked_X) / np.dot(
                self.W.T, mask * np.dot(self.W, self.H)
            )
            self.H = np.maximum(self.H, eps)

            # Evaluate
            X_est = np.dot(self.W, self.H)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est
            # curRes = linalg.norm(mask * (masked_X - X_est), ord='fro')
            if ctr % 10 == 0 and verbose:
                print("\tCurrent Iteration {}:".format(ctr))
                print("\tfit residual", np.round(fit_residual, 4))
                # print('\ttotal residual', np.round(curRes, 4))
            ctr += 1
        self.is_fit = True

    def predict(self, **kwargs):

        """Predict Subject's missing items using NNMF with multiplicative updating

        Args:
            data (Dataframe): pandas dataframe instance of data
            k (int): number of closest neighbors to use
        Returns:
            predicted_rating (Dataframe): adds field to object instance
        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        self.predictions = self.data.copy()
        self.predictions.loc[:, :] = np.dot(self.W, self.H)
        self.is_predict = True


# TODO: add stopping criteria as argument
# TODO: see if we can easily pass hyperparams skleanr grid-search
# TODO: see if we can manage sparse arrays and swap out pandas for numpy
# TODO: see if we can use a real sgd optimizer
# TODO: fix but when training size is really small, probably related to no variance in correlating predictions with actual
class NNMF_sgd(BaseCF):
    """Train non negative matrix factorization model using stochastic gradient descent.
    Allows masking to only learn the train weights.

    This code is based off of Ethan Rosenthal's excellent tutorial
    on collaborative filtering https://blog.insightdatascience.com/
    explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#.kkr7mzvr2

    """

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)

    def fit(
        self,
        n_factors=None,
        item_fact_reg=0.0,
        user_fact_reg=0.0,
        item_bias_reg=0.0,
        user_bias_reg=0.0,
        learning_rate=0.001,
        n_iterations=100,
        verbose=False,
        dilate_ts_n_samples=None,
        **kwargs,
    ):

        """Fit NNMF collaborative filtering model to train data using stochastic gradient descent.

        Args:
            n_factors (int): Number of factors or components
            max_iterations (int):  maximum number of interations (default=100)
            error_limit (float): error tolerance (default=1e-6)
            fit_error_limit (float): fit error tolerance (default=1e-6)
            verbose (bool): verbose output during fitting procedure (default=True)
            dilate_ts_n_samples (int): will dilate masked samples by n_samples to leverage auto-correlation in estimating time-series data

        """

        # initialize variables
        n_users, n_items = self.data.shape
        if n_factors is None:
            n_factors = n_items

        if dilate_ts_n_samples is not None:
            self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)

        if self.is_mask:
            if self.is_mask_dilated:
                data = self.masked_data[self.dilated_mask]
                sample_row, sample_col = self.dilated_mask.values.nonzero()
                self.global_bias = data[self.dilated_mask].mean().mean()
            else:
                data = self.masked_data[self.train_mask]
                sample_row, sample_col = self.train_mask.values.nonzero()
                self.global_bias = data[self.train_mask].mean().mean()
        else:
            data = self.data.copy()
            sample_row, sample_col = zip(*np.argwhere(~np.isnan(data.values)))
            self.global_bias = data.values[~np.isnan(data.values)].mean()

        # initialize latent vectors
        self.user_vecs = np.random.normal(
            scale=1.0 / n_factors, size=(n_users, n_factors)
        )
        self.item_vecs = np.random.normal(
            scale=1.0 / n_factors, size=(n_items, n_factors)
        )

        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg

        # train weights
        ctr = 1
        while ctr <= n_iterations:
            if ctr % 10 == 0 and verbose:
                print("\tCurrent Iteration: {}".format(ctr))

            training_indices = np.arange(len(sample_row))
            np.random.shuffle(training_indices)

            for idx in training_indices:
                u = sample_row[idx]
                i = sample_col[idx]
                prediction = self._predict_single(u, i)

                # Use changes in e to determine tolerance
                e = data.iloc[u, i] - prediction  # error

                # Update biases
                self.user_bias[u] += learning_rate * (
                    e - self.user_bias_reg * self.user_bias[u]
                )
                self.item_bias[i] += learning_rate * (
                    e - self.item_bias_reg * self.item_bias[i]
                )

                # Update latent factors
                self.user_vecs[u, :] += learning_rate * (
                    e * self.item_vecs[i, :] - self.user_fact_reg * self.user_vecs[u, :]
                )
                self.item_vecs[i, :] += learning_rate * (
                    e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i, :]
                )
            ctr += 1
        self.is_fit = True

    def predict(self, **kwargs):

        """Predict Subject's missing items using NNMF with stochastic gradient descent

        Args:
            data (Dataframe): pandas dataframe instance of data
            k (int): number of closest neighbors to use
        Returns:
            predicted_rating (Dataframe): adds field to object instance
        """

        self.predictions = self.data.copy()
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                self.predictions.iloc[u, i] = self._predict_single(u, i)
        self.is_predict = True

    def _predict_single(self, u, i):
        """ Single user and item prediction."""
        prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
        prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        return prediction
