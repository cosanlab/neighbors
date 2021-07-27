"""
Core algorithms for collaborative filtering
"""

import pandas as pd
import numpy as np
from .base import Base, BaseNMF
from .utils import nanpdist
from ._fit import sgd, mult
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from numba.core.errors import NumbaPerformanceWarning

__all__ = ["Mean", "KNN", "NNMF_mult", "NNMF_sgd"]


class Mean(Base):
    """
    The Mean algorithm simply uses the mean of other users to make predictions about items. It's primarily useful as a good baseline model.
    """

    def __init__(
        self, data, mask=None, n_mask_items=None, verbose=True, random_state=None
    ):
        """
        Args:
            data (pd.DataFrame): users x items dataframe
            mask (pd.DataFrame, optional): A boolean dataframe used to split the data into 'observed' and 'missing' datasets. Defaults to None.
            n_mask_items (int/float, optional): number of items to mask out, while the rest are treated as observed; Defaults to None.
            data_range (int/float, optional): max - min of the data; Default computed from the input data. This is useful to set manually in case the input data do not span the full range of possible values
            random_state (None, int, RandomState): a seed or random state used for all internal random operations (e.g. randomly mask half the data given n_mask_item = .05). Passing None will generate a new random seed. Default None.
            verbose (bool; optional): print any initialization warnings; Default True

        """
        super().__init__(
            data, mask, n_mask_items, random_state=random_state, verbose=verbose
        )
        self.mean = None

    def fit(self, dilate_by_nsamples=None, axis=0, **kwargs):

        """Fit model to train data. Simply learns item-wise mean using observed (non-missing) values.

        Args:
            dilate_ts_n_samples (int): will dilate masked samples by n_samples to leverage auto-correlation in estimating time-series data
            axis (int): dimension along which to compute mean, 0 = mean across users separately by item, 1 = mean across items separately by user; Default 0

        """

        # Call parent fit which acts as a guard for non-masked data
        super().fit()

        self.dilate_mask(n_samples=dilate_by_nsamples)
        self.mean = self.masked_data.mean(skipna=True, axis=axis)
        self._predict()
        self.is_fit = True

    def _predict(self):

        """Predict missing items using other subject's item means."""

        # Always predict mean (learned on observed values) for observed and missing values
        self.predictions = pd.concat([self.mean] * self.data.shape[0], axis=1).T
        self.predictions.index = self.data.index
        self.predictions.columns = self.data.columns


class KNN(Base):
    """
    The K-Nearest Neighbors algorithm makes predictions using a weighted mean of a subset of similar users. Similarity can be controlled via the `metric` argument to the `.fit` method, and the number of other users can be controlled with the `k` argument to the `.predict` method. NOTE: If user similiarity cannot be computed or no observed ratings have been made by the top k simililar users, this algorithm will fallback to the global mean on observed data for prediction (i.e. like the `Mean` model).
    """

    def __init__(
        self, data, mask=None, n_mask_items=None, verbose=True, random_state=None
    ):
        """
        Args:
            data (pd.DataFrame): users x items dataframe
            mask (pd.DataFrame, optional): A boolean dataframe used to split the data into 'observed' and 'missing' datasets. Defaults to None.
            n_mask_items (int/float, optional): number of items to mask out, while the rest are treated as observed; Defaults to None.
            data_range (int/float, optional): max - min of the data; Default computed from the input data. This is useful to set manually in case the input data do not span the full range of possible values
            random_state (None, int, RandomState): a seed or random state used for all internal random operations (e.g. randomly mask half the data given n_mask_item = .05). Passing None will generate a new random seed. Default None.
            verbose (bool; optional): print any initialization warnings; Default True

        """
        super().__init__(
            data, mask, n_mask_items, random_state=random_state, verbose=verbose
        )
        self.user_similarity = None
        self.metric = None

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, similarity_metric={self.metric})"

    def fit(
        self,
        k=10,
        metric="correlation",
        axis=0,
        dilate_by_nsamples=None,
        skip_refit=False,
        **kwargs,
    ):

        """Fit collaborative model to train data.  Calculate similarity between subjects across items. Repeated called to fit with different k, but the same previous arguments will re-use the computed user x user similarity matrix.

        Args:
            k (int): maximum number of other users to use when making a prediction for a single user. If set to None will use all users. Default 10. Note: it's possible for predictions to come from fewer than k other users if a particular user has fewer similar neighbors with positive similarity scores.
            metric (str; optional): type of similarity. One of 'correlation', 'spearman', 'kendall', 'cosine', or 'pearson'. 'pearson' is just an alias for 'correlation'. Default 'correlation'.
            axis (int): dimension along which to compute mean, 0 = mean across users separately by item, 1 = mean across items separately by user; Default 0
            skip_refit (bool; optional): skip re-estimation of user x user similarity matrix. Faster if only exploring different k and no other model parameters or masks are changing. Default False.
        """

        metrics = ["pearson", "spearman", "kendall", "cosine", "correlation"]
        if metric not in metrics:
            raise ValueError(f"metric must be one of {metrics}")

        self.metric = metric

        if metric == "correlation":
            metric = "pearson"

        # Call parent fit which acts as a guard for non-masked data
        super().fit()

        # If fit is being called more than once in a row with different k, but no other arguments are changing, reuse the last computed similarity matrix to save time. Otherwise re-calculate it
        if not skip_refit:
            self.dilate_mask(n_samples=dilate_by_nsamples)

            # Store the mean because we'll use it in cases we can't make a prediction
            self.mean = self.masked_data.mean(skipna=True, axis=axis)

            if metric in ["pearson", "kendall", "spearman"]:
                # Fall back to pandas
                sim = self.masked_data.T.corr(method=metric)
            else:
                # Convert distance metrics to similarity (currently only cosine)
                sim = pd.DataFrame(
                    1 - nanpdist(self.masked_data.to_numpy(), metric=metric),
                    index=self.masked_data.index,
                    columns=self.masked_data.index,
                )

            self.user_similarity = sim
        self._predict(k=k)
        self.is_fit = True

    def _predict(self, k):
        """Make predictions using computed user similarities.

        Args:
            k (int): number of closest neighbors to use

        """

        predictions = self.masked_data.copy()

        for row_idx, _ in self.masked_data.iterrows():

            user_prediction_error = False
            # Get the similarity of this user to all other users, ignoring self-similarity
            top_user_sims = self.user_similarity.loc[row_idx].drop(row_idx)
            if top_user_sims.isnull().all():
                warnings.warn(
                    f"User {row_idx} has no variance in their ratings. Impossible to compute similarity with other users. Falling back to global mean for all predictions"
                )
                user_prediction_error = True  # can't predict
            else:
                # Remove nan users and sort
                top_user_sims = top_user_sims[~top_user_sims.isnull()].sort_values(
                    ascending=False
                )
                if len(top_user_sims) == 0:
                    user_prediction_error = True  # can't predict
                else:
                    # Get top k if requested
                    if k is not None:
                        top_user_sims = top_user_sims[: k + 1]

                    # Rescale similarity scores to the range 0 - 1, which has the effect of zeroing out negative similarities for currently supported similarity metrics.
                    # NOTE: we should revisit this approach for non-normalized similarity metrics e.g. euclidean distance
                    top_user_sims = top_user_sims.clip(lower=0, upper=1)

                    # No top users with positive correlations
                    if len(np.nonzero(top_user_sims.to_numpy())[0]) == 0:
                        user_prediction_error = True
                    else:
                        # NOTE: this code block is just a vectorized version of looping over every item for the current user and seeing whether we have observed ratings for each of the k other users to make a prediction with. We do this because for each item the *actual* number of other users' data availble for prediction will vary between 0-k based the pattern of sparsity

                        # Get the observed ratings from top users
                        top_user_ratings = self.masked_data.loc[top_user_sims.index, :]

                        # Make predictions = user_similarity_scores (column vector) * user x item (matrix of observed ratings)
                        # Do this in pandas rather than numpy because numpy will return nans when summing items if any item is nan
                        # Yields user x item matrix of ratings scaled by similarities
                        preds = (top_user_sims * top_user_ratings.T).T
                        # Add up the ratings from other users ignoring NaNs; this serves as the numerator of the formula
                        rating_sums = preds.sum()

                        # Now some of the values in preds will be nan because we never observed a rating for that user + item combo. We need to know how many are nans and which exact ones, because we need to sum down users for preds and then divide by the sum of the similarity weights we did end up using.
                        # Get locations of where we were able to make a prediction.
                        preds_mask = ~preds.isnull()
                        # Broadcast the user similarity vector over the user x item matrix so each column now contains the user similarity score if observed a prediction from that user and a 0 if not (True is converted to 1 during this multiplication whereas False is converted to 0)
                        user_sims_mat = (preds_mask.T * top_user_sims).T
                        # Now we can just sum down the rows which will give us the sum of the similarity weights we actually used
                        sim_sums = user_sims_mat.sum()
                        # Finally get the predictions by dividing the sum of ratings by sum of similarities we ended up using. This is how Surprise does it too: https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/knns.py#L124
                        preds = rating_sums / sim_sums

                        # For items we can't predict because we never observed any ratings from the top k users for that item, fill in with the global mean for that item
                        if preds.isnull().any():
                            preds[preds.isnull()] = self.mean[preds.isnull()]

                        predictions.loc[row_idx] = preds.to_numpy()

            # Handle cases where we were unable to make any predictions for this user
            if user_prediction_error:
                warnings.warn(
                    f"Not enough similar users with data to make any predictions for user {row_idx}. Falling back to global mean for all predictions"
                )
                predictions.loc[row_idx, :] = self.mean.to_numpy()

        self.predictions = predictions

    def plot_user_similarity(
        self, figsize=(8, 8), label_fontsize=16, hide_title=False, heatmap_kwargs={}
    ):
        """
        Plot a heatmap of user x user similarities learned on the observed data

        Args:
            figsize (tuple, optional): matplotlib figure size. Defaults to (8, 8).
            label_fontsize (int; optional): fontsize for title text; Default 16
            hide_title (bool; optional): hide title containing metric information; Default False
            heatmap_kwargs (dict, optional): addition arguments to seaborn.heatmap.

        Returns:
            ax: matplotib axis handle
        """
        if not self.is_fit:
            raise ValueError("Model as not been fit")
        if self.metric in ["correlation", "pearson", "spearman"]:
            vmin, vmax = -1, 1
            cmap = "RdBu_r"
        else:
            vmin, vmax = 0, 1
            cmap = None

        _, ax = plt.subplots(1, 1, figsize=figsize)
        _ = ax.set(xlabel=None, ylabel=None)

        ax = sns.heatmap(
            self.user_similarity,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            square=True,
            ax=ax,
            **heatmap_kwargs,
        )
        if not hide_title:
            _ = ax.set_title(f"Metric: {self.metric}", fontsize=label_fontsize)
        return ax


class NNMF_mult(BaseNMF):
    """
    The non-negative matrix factorization algorithm tries to decompose a users x items matrix into two additional matrices: users x factors and factors x items.

    Training is performed via multiplicative updating and continues until convergence or the maximum number of training iterations has been reached. Unlike the `NNMF_sgd`, this implementation takes no hyper-parameters and thus is simpler and faster to use, but less flexible, i.e. no regularization.

    The number of factors, convergence, and maximum iterations can be controlled with the `n_factors`, `tol`, and `max_iterations` arguments to the `.fit` method. By default the number of factors = the number items.

    The implementation here follows closely that of Lee & Seung, 2001 (eq 4): https://papers.nips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf

    *Note*: `random_state` does not control the sgd fit, only the initialization of the factor matrices
    """

    def __init__(
        self, data, mask=None, n_mask_items=None, verbose=True, random_state=None
    ):
        """
        Args:
            data (pd.DataFrame): users x items dataframe
            mask (pd.DataFrame, optional): A boolean dataframe used to split the data into 'observed' and 'missing' datasets. Defaults to None.
            n_mask_items (int/float, optional): number of items to mask out, while the rest are treated as observed; Defaults to None.
            data_range (int/float, optional): max - min of the data; Default computed from the input data. This is useful to set manually in case the input data do not span the full range of possible values
            random_state (None, int, RandomState): a seed or random state used for all internal random operations (e.g. randomly mask half the data given n_mask_item = .05). Passing None will generate a new random seed. Default None.
            verbose (bool; optional): print any initialization warnings; Default True

        """
        super().__init__(
            data, mask, n_mask_items, random_state=random_state, verbose=verbose
        )
        self.H = None  # factors x items
        self.W = None  # user x factors
        self.n_factors = None

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, n_factors={self.n_factors})"

    def fit(
        self,
        n_factors=None,
        n_iterations=1000,
        tol=1e-6,
        eps=1e-6,
        verbose=False,
        dilate_by_nsamples=None,
        **kwargs,
    ):

        """Fit NNMF collaborative filtering model to train data using multiplicative updating.

        Given non-negative matrix `V` find non-negative factors `W` and `H` by minimizing `||V - WH||^2`.

        Args:
            n_factors (int, optional): number of factors to learn. Defaults to None which includes all factors.
            n_iterations (int, optional): total number of training iterations if convergence is not achieved. Defaults to 5000.
            tol (float, optional): Convergence criteria. Model is considered converged if the change in error during training < tol. Defaults to 0.001.
            eps (float; optiona): small value added to denominator of update rules to avoid divide-by-zero errors; Default 1e-6.
            verbose (bool, optional): print information about training. Defaults to False.
            dilate_by_nsamples (int, optional): How many items to dilate by prior to training. Defaults to None.
        """

        # Call parent fit which acts as a guard for non-masked data
        super().fit()

        n_users, n_items = self.data.shape

        if (
            isinstance(n_factors, int) and (n_factors > n_items and n_factors > n_users)
        ) or isinstance(n_factors, np.floating):
            raise TypeError("n_factors must be an integer < number of items and users")

        if n_factors is None:
            n_factors = min([n_users, n_items])

        self.n_factors = n_factors

        # Initialize W and H as non-negative scaled random values
        # We use random initialization scaled by the number of factors not unlike sklearn: https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/decomposition/_nmf.py#L334
        self.W = np.abs(
            self.random_state.normal(scale=1.0 / n_factors, size=(n_users, n_factors))
        )
        self.H = np.abs(
            self.random_state.normal(scale=1.0 / n_factors, size=(n_factors, n_items))
        )

        # Whereas in SGD we explity pass in indices of training data for fitting, here we set testing indices to 0 so they have no impact on the multiplicative update. See Zhu, 2016 for more details: https://arxiv.org/pdf/1612.06037.pdf
        self.dilate_mask(n_samples=dilate_by_nsamples)

        # fillna(0) is equivalent to hadamard (element-wise) product with a binary mask
        X = self.masked_data.fillna(0).to_numpy()

        # Run multiplicative updating
        # Silence numba warning until this issue gets fixed: https://github.com/numba/numba/issues/4585
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
            error_history, converged, n_iter, delta, norm_rmse, W, H = mult(
                X,
                self.W,
                self.H,
                self.data_range,
                eps,
                tol,
                n_iterations,
                verbose,
            )

        # Save outputs to model
        self.W, self.H = W, H
        self.error_history = error_history
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
                print(f"\tFinal delta exceeds tol: {tol} <= {self._delta}")

            print(f"\tFinal Norm Error: {np.round(100*norm_rmse, 2)}%")
        self._predict()
        self.is_fit = True

    def _predict(self):

        """Predict subjects' missing items using NNMF with multiplicative updating"""

        self.predictions = pd.DataFrame(
            self.W @ self.H, index=self.data.index, columns=self.data.columns
        )


class NNMF_sgd(BaseNMF):
    """
    The non-negative matrix factorization algorithm tries to decompose a users x items matrix into two additional matrices: users x factors and factors x items.

    Training is performed via stochastic-gradient-descent and continues until convergence or the maximum number of iterations has been reached. Unlike `NNMF_mult` errors during training are used to update latent factors *separately* for each user/item combination. Additionally this implementation is more flexible as it supports hyperparameters for various kinds of regularization at the cost of increased computation time.

    The number of factors, convergence, and maximum iterations can be controlled with the `n_factors`, `tol`, and `max_iterations` arguments to the `.fit` method. By default the number of factors = the number items.

    `random_state` does not control the sgd fit, only the initialization of the factor matrices

    **Important Note**: model fitting can be highly sensitive to the regularization hyper-parameters passed to `.fit`. These hyper-parameters control the amount of regularization used when learning user and item factors and biases. By default *no regularization* is performed. For some combinations of hyper-parameters (e.g. large `user_fact_reg` and small `item_fact_reg`) latent vectors can blow up to infinity producing `NaNs` in model estimates. Model fitting will not fail in these cases so **caution** should be taken when making use of hyper-parameters.

    """

    def __init__(
        self, data, mask=None, n_mask_items=None, verbose=True, random_state=None
    ):
        """
        Args:
            data (pd.DataFrame): users x items dataframe
            mask (pd.DataFrame, optional): A boolean dataframe used to split the data into 'observed' and 'missing' datasets. Defaults to None.
            n_mask_items (int/float, optional): number of items to mask out, while the rest are treated as observed; Defaults to None.
            data_range (int/float, optional): max - min of the data; Default computed from the input data. This is useful to set manually in case the input data do not span the full range of possible values
            random_state (None, int, RandomState): a seed or random state used for all internal random operations (e.g. randomly mask half the data given n_mask_item = .05). Passing None will generate a new random seed. Default None.
            verbose (bool; optional): print any initialization warnings; Default True

        """
        super().__init__(
            data, mask, n_mask_items, random_state=random_state, verbose=verbose
        )
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
        n_iterations=1000,
        tol=1e-6,
        verbose=False,
        dilate_by_nsamples=None,
        **kwargs,
    ):
        """
        Fit NNMF collaborative filtering model using stochastic-gradient-descent. **Note:** Some combinations of fit parameters may lead to degenerate fits due to use and item vectors converging to infinity. Because no constraints are imposed on the values these parameters can take, please adjust them with caution. If you encounter NaNs in your predictions it's likely because of the specific combination of parameters you chose and you can try refitting with the default settings (i.e. no regularization and learning rate = 0.001). Use `verbose=True` to help determine at what iteration these degenerate fits occur.

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
            dilate_by_nsamples (int, optional): How many items to dilate by prior to training. Defaults to None.
        """

        # Call parent fit which acts as a guard for non-masked data
        super().fit()

        # initialize variables
        n_users, n_items = self.data.shape

        if (
            isinstance(n_factors, int) and (n_factors > n_items and n_factors > n_users)
        ) or isinstance(n_factors, np.floating):
            raise TypeError("n_factors must be an integer < number of items and users")

        if n_factors is None:
            n_factors = min([n_users, n_items])

        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.error_history = []

        # Perform dilation if requested
        self.dilate_mask(n_samples=dilate_by_nsamples)

        # Get indices of training data to compute; np.nonzero returns a tuple of row and column indices that when iterated over simultaneosly yield the [row_index, col_index] of each training observation
        if self.is_mask_dilated:
            row_indices, col_indices = self.dilated_mask.values.nonzero()
        else:
            row_indices, col_indices = self.mask.values.nonzero()

        # Convert tuples cause numba complains
        row_indices, col_indices = np.array(row_indices), np.array(col_indices)

        # Initialize global, user, and item biases and latent vectors
        self.global_bias = self.masked_data.mean().mean()
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Initialize random values oriented these as user x factor, factor x item
        self.user_vecs = np.abs(
            self.random_state.normal(scale=1.0 / n_factors, size=(n_users, n_factors))
        )
        self.item_vecs = np.abs(
            self.random_state.normal(scale=1.0 / n_factors, size=(n_factors, n_items))
        )

        X = self.masked_data.to_numpy()

        # Generate seed for shuffling within sgd
        seed = self.random_state.randint(np.iinfo(np.int32).max)

        # Run SGD
        # Silence numba warning until this issue gets fixed: https://github.com/numba/numba/issues/4585
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
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
                X,
                seed,
                self.global_bias,
                self.data_range,
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
                row_indices,
                col_indices,
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
                print(f"\tFinal delta exceeds tol: {tol} <= {self._delta}")

            print(f"\tFinal Norm Error: {np.round(100*norm_rmse, 2)}%")

        self._predict()
        self.is_fit = True

    def _predict(self):

        """Predict User's missing items using NNMF with stochastic gradient descent"""

        # user x factor * factor item + biases
        predictions = self.user_vecs @ self.item_vecs
        predictions = (
            (predictions.T + self.user_bias).T + self.item_bias + self.global_bias
        )
        self.predictions = pd.DataFrame(
            predictions, index=self.data.index, columns=self.data.columns
        )
