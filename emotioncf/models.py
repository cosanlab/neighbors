"""
Core algorithms for collaborative filtering
"""

import pandas as pd
import numpy as np
from .base import Base, BaseNMF
from .utils import nanpdist
from ._fit import sgd, mult

__all__ = ["Mean", "KNN", "NNMF_mult", "NNMF_sgd"]


class Mean(Base):
    """
    The Mean algorithm simply uses the mean of other users to make predictions about items. It's primarily useful as a good baseline model.
    """

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.mean = None

    def fit(self, dilate_ts_n_samples=None, **kwargs):

        """Fit collaborative model to train data.  Calculate similarity between subjects across items

        Args:
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

    def predict(self):

        """Predict missing items using other subject's item means.

        Returns:
            predicted_rating (Dataframe): adds field to object instance

        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        self.predictions = self.data.copy()
        for row in self.data.iterrows():
            self.predictions.loc[row[0]] = self.mean
        self.is_predict = True


class KNN(Base):
    """
    The K-Nearest Neighbors algorithm makes predictions using a weighted mean of a subset of similar users. Similarity can be controlled via the `metric` argument to the `.fit` method, and the number of other users can be controlled with the `k` argument to the `.predict` method.
    """

    def __init__(self, data, mask=None, n_mask_items=None):
        super().__init__(data, mask, n_mask_items)
        self.subject_similarity = None
        self._last_metric = None
        self._last_dilate_by_nsamples = None

    def fit(self, k=None, metric="pearson", dilate_by_nsamples=None, skip_refit=False):

        """Fit collaborative model to train data.  Calculate similarity between subjects across items. Repeated called to fit with different k, but the same previous arguments will re-use the computed user x user similarity matrix.

        Args:
            k (int): number of closest neighbors to use
            metric (str; optional): type of similarity. One of 'pearson', 'spearman', 'kendall', 'cosine', or 'correlation'. 'correlation' is just an alias for 'pearson'. Default 'pearson'.
            skip_refit (bool; optional): skip re-estimation of user x user similarity matrix. Faster if only exploring different k and no other model parameters or masks are changing. Default False.
        """

        metrics = ["pearson", "spearman", "kendall", "cosine", "correlation"]
        if metric not in metrics:
            raise ValueError(f"metric must be one of {metrics}")

        # If fit is being called more than once in a row with different k, but no other arguments are changing, reuse the last computed similarity matrix to save time. Otherwise re-calculate it
        if not skip_refit:
            self.dilate_mask(dilate_by_nsamples=dilate_by_nsamples)
            if metric in ["pearson", "kendall", "spearman"]:
                # Fall back to pandas
                sim = self.masked_data.T.corr(method=metric)
            else:
                sim = pd.DataFrame(
                    1 - nanpdist(self.masked_data.to_numpy(), metric=metric),
                    index=self.masked_data.index,
                    columns=self.masked_data.index,
                )

            self.subject_similarity = sim
        self._predict(k=k)
        self.is_fit = True

    def _predict(self, k=None):
        """Make predictions using computed subject similarities.

        Args:
            k (int): number of closest neighbors to use

        """

        data = self.masked_data if self.is_masked else self.data
        predictions = []

        # Get top k most similar other subjects for each subject
        # We loop instead of apply because we want to retain row indices and column indices
        for user_idx in range(data.shape[0]):
            if k is not None:
                top_subjects = (
                    self.subject_similarity.loc[user_idx]
                    .drop(user_idx)
                    .sort_values(ascending=False)[: k + 1]
                )
            else:
                top_subjects = (
                    self.subject_similarity.loc[user_idx]
                    .drop(user_idx)
                    .sort_values(ascending=False)
                )
            # remove nan subjects
            top_subjects = top_subjects[~top_subjects.isnull()]

            # Get item predictions
            predictions.append(
                np.dot(top_subjects, self.data.loc[top_subjects.index])
                / len(top_subjects)
            )

        self.predictions = pd.DataFrame(
            predictions, index=data.index, columns=data.columns
        )


class NNMF_mult(BaseNMF):
    """
    The non-negative matrix factorization algorithm tries to decompose a users x items matrix into two additional matrices: users x factors and factors x items.

    Training is performed via multiplicative updating and continues until convergence or the maximum number of training iterations has been reached. Unlike the `NNMF_sgd`, this implementation takes no hyper-parameters and thus is simpler and faster to use, but less flexible, i.e. no regularization.

    The number of factors, convergence, and maximum iterations can be controlled with the `n_factors`, `tol`, and `max_iterations` arguments to the `.fit` method. By default the number of factors = the number items.

    The implementation here follows closely that of Lee & Seung, 2001 (eq 4): https://papers.nips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf

    """

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.H = None  # factors x items
        self.W = None  # user x factors
        self.n_factors = None

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, n_factors={self.n_factors})"

    def fit(
        self,
        n_factors=None,
        n_iterations=5000,
        tol=1e-6,
        eps=1e-6,
        verbose=False,
        dilate_ts_n_samples=None,
    ):

        """Fit NNMF collaborative filtering model to train data using multiplicative updating.

        Given non-negative matrix `V` find non-negative factors `W` and `H` by minimizing `||V - WH||^2`.

        Args:
            n_factors (int, optional): number of factors to learn. Defaults to None which includes all factors.
            n_iterations (int, optional): total number of training iterations if convergence is not achieved. Defaults to 5000.
            tol (float, optional): Convergence criteria. Model is considered converged if the change in error during training < tol. Defaults to 0.001.
            eps (float; optiona): small value added to denominator of update rules to avoid divide-by-zero errors; Default 1e-6.
            verbose (bool, optional): print information about training. Defaults to False.
            dilate_ts_n_samples (int, optional): How many items to dilate by prior to training. Defaults to None.
            save_learning (bool, optional): Save error for each training iteration for diagnostic purposes. Set this to False if memory is a limitation and the n_iterations is very large. Defaults to True.
        """

        n_users, n_items = self.data.shape

        if n_factors is None:
            n_factors = n_items

        self.n_factors = n_factors

        # Initialize W and H at non-negative scaled random values
        # We use random initialization scaled by the data, like sklearn: https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/decomposition/_nmf.py#L334
        avg = np.sqrt(np.nanmean(self.data) / n_factors)
        self.H = avg * np.random.rand(n_factors, n_items)
        self.W = avg * np.random.rand(n_users, n_factors)

        # Appropriately handle mask
        # TODO: refactor this masking block to match the SGD implementation
        # Unlike SGD, we explicitly set missing data to 0 so that it gets ignored in the multiplicative update. See Zhu, 2016 for a justification of using a binary mask matrix like this during the updates: https://arxiv.org/pdf/1612.06037.pdf
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

        # Ger data randge for norm_mse
        data_range = self.data.max().max() - self.data.min().min()

        # Run multiplicative updating
        error_history, converged, n_iter, delta, norm_rmse, W, H = mult(
            masked_X,
            self.W,
            self.H,
            data_range,
            eps,
            tol,
            n_iterations,
            verbose,
        )

        # Save outputs to model
        self.W, self.H = W, H
        self.error_history = error_history
        self.is_fit = True
        self._n_iter = n_iter
        self._delta = delta
        self._norm_rmse = norm_rmse
        self.converged = converged

        if verbose:
            if self.converged:
                print("\n\tCONVERGED!")
                print(f"\n\tFinal Iteration: {self._n_iter}")
                print(f"\tFinal Delta: {np.round(self._delta)}")
            else:
                print("\tFAILED TO CONVERGE (n_iter reached)")
                print(f"\n\tFinal Iteration: {self._n_iter}")
                print(f"\tFinal delta exceeds tol: {tol} <= {np.round(self._delta, 5)}")

            print(f"\tFinal Norm Error: {np.round(100*norm_rmse, 2)}%")

    def predict(self):

        """Predict Subject's missing items using NNMF with multiplicative updating

        Returns:
            predicted_rating (Dataframe): adds field to object instance
        """

        if not self.is_fit:
            raise ValueError("You must fit() model first before using this method.")

        self.predictions = self.data.copy()
        self.predictions.loc[:, :] = np.dot(self.W, self.H)
        self.is_predict = True


class NNMF_sgd(BaseNMF):
    """
    The non-negative matrix factorization algorithm tries to decompose a users x items matrix into two additional matrices: users x factors and factors x items.

    Training is performed via stochastic-gradient-descent and continues until convergence or the maximum number of iterations has been reached. Unlike `NNMF_mult` errors during training are used to update latent factors *separately* for each user/item combination. Additionally this implementation is more flexible as it supports hyperparameters for various kinds of regularization at the cost of increased computation time.

    The number of factors, convergence, and maximum iterations can be controlled with the `n_factors`, `tol`, and `max_iterations` arguments to the `.fit` method. By default the number of factors = the number items.

    """

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.n_factors = None

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, n_factors={self.n_factors})"

    def fit(
        self,
        n_factors=None,
        item_fact_reg=0.0,
        user_fact_reg=0.0,
        item_bias_reg=0.0,
        user_bias_reg=0.0,
        learning_rate=0.001,
        n_iterations=5000,
        tol=1e-6,
        verbose=False,
        dilate_ts_n_samples=None,
    ):
        """
        Fit NNMF collaborative filtering model using stochastic-gradient-descent

        Args:
            n_factors (int, optional): number of factors to learn. Defaults to None which includes all factors.
            item_fact_reg (float, optional): item factor regularization to apply. Defaults to 0.0.
            user_fact_reg (float, optional): user factor regularization to apply. Defaults to 0.0.
            item_bias_reg (float, optional): item factor bias term to apply. Defaults to 0.0.
            user_bias_reg (float, optional): user factor bias term to apply. Defaults to 0.0.
            learning_rate (float, optional): how quickly to integrate errors during training. Defaults to 0.001.
            n_iterations (int, optional): total number of training iterations if convergence is not achieved. Defaults to 5000.
            tol (float, optional): Convergence criteria. Model is considered converged if the change in error during training < tol. Defaults to 0.001.
            verbose (bool, optional): print information about training. Defaults to False.
            dilate_ts_n_samples (int, optional): How many items to dilate by prior to training. Defaults to None.
            save_learning (bool, optional): Save error for each training iteration for diagnostic purposes. Set this to False if memory is a limitation and the n_iterations is very large. Defaults to True.
            fast_sdg (bool; optional): Use an JIT compiled SGD for faster fitting. Note that verbose outputs are not compatible with this option and error history is always saved; Default False
        """

        # initialize variables
        n_users, n_items = self.data.shape
        if n_factors is None:
            n_factors = n_items

        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.error_history = []

        # Perform dilation if requested
        if dilate_ts_n_samples is not None:
            self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)

        # Appropriately handle mask
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

        # Convert tuples cause numba complains
        sample_row, sample_col = np.array(sample_row), np.array(sample_col)

        # Initialize user and item biases and latent vectors
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_vecs = np.random.normal(
            scale=1.0 / n_factors, size=(n_users, n_factors)
        )
        self.item_vecs = np.random.normal(
            scale=1.0 / n_factors, size=(n_items, n_factors)
        )

        # Get data range for norm_rmse
        data_range = self.data.max().max() - self.data.min().min()

        # Run SGD
        (
            error_history,
            converged,
            n_iter,
            delta,
            norm_rmse,
            user_bias,
            user_vecs,
            item_bias,
            item_vecs,
        ) = sgd(
            data.to_numpy(),
            self.global_bias,
            data_range,
            tol,
            self.user_bias,
            self.user_vecs,
            self.user_bias_reg,
            self.user_fact_reg,
            self.item_bias,
            self.item_vecs,
            self.item_bias_reg,
            self.item_fact_reg,
            n_iterations,
            sample_row,
            sample_col,
            learning_rate,
            verbose,
        )
        # Save outputs to model
        (
            self.error_history,
            self.user_bias,
            self.user_vecs,
            self.item_bias,
            self.item_vecs,
        ) = (
            error_history,
            user_bias,
            user_vecs,
            item_bias,
            item_vecs,
        )

        self.is_fit = True
        self._n_iter = n_iter
        self._delta = delta
        self._norm_rmse = norm_rmse
        self.converged = converged
        if verbose:
            if self.converged:
                print("\n\tCONVERGED!")
                print(f"\n\tFinal Iteration: {self._n_iter}")
                print(f"\tFinal Delta: {np.round(self._delta)}")
            else:
                print("\tFAILED TO CONVERGE (n_iter reached)")
                print(f"\n\tFinal Iteration: {self._n_iter}")
                print(f"\tFinal delta exceeds tol: {tol} <= {np.round(self._delta, 5)}")

            print(f"\tFinal Norm Error: {np.round(100*norm_rmse, 2)}%")

    def predict(self):

        """Predict Subject's missing items using NNMF with stochastic gradient descent

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
