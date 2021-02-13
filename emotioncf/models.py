"""
Core algorithms for collaborative filtering
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from .base import Base, BaseNMF
from .utils import nanpdist

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

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.subject_similarity = None

    # TODO speed up cosine (use sklearn pairwise)
    # TODO remove correlation but make sure pearson can handle null values
    def fit(self, metric="pearson", dilate_ts_n_samples=None, **kwargs):

        """Fit collaborative model to train data.  Calculate similarity between subjects across items

        Args:
            metric (str; optional): type of similarity. One of 'pearson', 'spearman', 'kendall', 'cosine', or 'correlation'. 'correlation' is just an alias for 'pearson'. Default 'pearson'.
        """

        metrics = ["pearson", "spearman", "kendall", "cosine", "correlation"]
        if metric not in metrics:
            raise ValueError(f"metric must be one of {metrics}")

        if self.is_mask:
            data = self.data[self.train_mask]
        else:
            data = self.data.copy()

        if dilate_ts_n_samples is not None:
            data = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)
            data = data[self.dilated_mask]

        if metric in ["pearson", "kendall", "spearman"]:
            # Fall back to pandas
            sim = data.T.corr(method=metric)
        else:
            sim = pd.DataFrame(
                1 - nanpdist(data.to_numpy(), metric=metric),
                index=data.index,
                columns=data.index,
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


class NNMF_mult(BaseNMF):
    """
    The Non-negative Matrix Factorization algorithm tries to decompose a users x items matrix into two additional matrices: users x factors and factors x items. Training is performed via multiplicative updating and continues until convergence or the maximum number of training iterations has been reached. The number of factors, convergence, and maximum iterations can be controlled with the `n_factors`, `tol`, and `max_iterations` arguments to the `.fit` method. By default the number of factors = the number items.
    """

    def __init__(self, data, mask=None, n_train_items=None):
        super().__init__(data, mask, n_train_items)
        self.H = None
        self.W = None
        self.n_factors = None

    def __repr__(self):
        return f"{super().__repr__()[:-1]}, n_factors={self.n_factors})"

    def fit(
        self,
        n_factors=None,
        max_iterations=100,
        fit_error_limit=1e-6,
        error_limit=1e-6,
        verbose=False,
        dilate_ts_n_samples=None,
        save_learning=True,
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
            save_learning (bool; optional): save a list of the error rate over iterations; Default True

        """

        eps = 1e-5

        n_users, n_items = self.data.shape

        if n_factors is None:
            n_factors = n_items

        self.n_factors = n_factors

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
        # TODO change to np.inf but make sure it doesn't screw up < fit_error_limit
        # TODO: Go over matrix math below, something is up with it cause n_factors doesn't work
        fit_residual = -np.inf
        current_resid = -np.inf
        error_limit = error_limit
        self.error_history = []
        while (
            ctr <= max_iterations
            or current_resid < error_limit
            or fit_residual < fit_error_limit
        ):
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
            # This is basically error gradient for convergence purposes
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            # Save this iteration's predictions
            X_est_prev = X_est
            # Update the residuals; note that the norm = RMSE not MSE
            current_resid = np.linalg.norm(masked_X - mask * X_est, ord="fro")
            # Norm the residual with respect to the max of the dataset so we can use a common convergence threshold
            current_resid /= masked_X.max()

            if save_learning:
                self.error_history.append(current_resid)
            # curRes = linalg.norm(mask * (masked_X - X_est), ord='fro')
            if ctr % 10 == 0 and verbose:
                print("\tCurrent Iteration {}:".format(ctr))
                print("\tfit residual", np.round(fit_residual, 4))
            ctr += 1
        self.is_fit = True

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


# TODO: see if we can easily pass hyperparams skleanr grid-search
# TODO: see if we can manage sparse arrays and swap out pandas for numpy
class NNMF_sgd(BaseNMF):
    """
    The Non-negative Matrix Factorization algorithm tries to decompose a users x items matrix into two additional matrices: users x factors and factors x items. Training is performed via stochastic-gradient-descent. Unlike `NNMF_mult` errors during training are used to update latent factors *separately* for each user/item combination. The number of factors, convergence, and maximum iterations can be controlled with the `n_factors`, `tol`, and `max_iterations` arguments to the `.fit` method. By default the number of factors = the number items.

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
        tol=0.001,
        verbose=False,
        dilate_ts_n_samples=None,
        save_learning=True,
        **kwargs,
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
        """

        # initialize variables
        n_users, n_items = self.data.shape
        if n_factors is None:
            n_factors = n_items

        self.n_factors = n_factors

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
        last_e = 0
        delta = np.inf
        max_norm = data.abs().max().max()
        self.error_history = []
        converged = False
        norm_e = np.inf
        with tqdm(total=n_iterations) as t:
            for ctr in range(1, n_iterations + 1):
                if verbose:
                    t.set_description(
                        f"Norm Error: {np.round(100*norm_e, 2)}% Delta Convg: {np.round(delta, 4)}||{tol}"
                    )
                    # t.set_postfix(Delta=f"{np.round(delta, 4)}")

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
                        e * self.item_vecs[i, :]
                        - self.user_fact_reg * self.user_vecs[u, :]
                    )
                    self.item_vecs[i, :] += learning_rate * (
                        e * self.user_vecs[u, :]
                        - self.item_fact_reg * self.item_vecs[i, :]
                    )
                # Normalize the current error with respect to the max of the dataset
                norm_e = np.abs(e) / max_norm
                # Compute the delta
                delta = np.abs(np.abs(norm_e) - np.abs(last_e))
                if save_learning:
                    self.error_history.append(norm_e)
                if delta < tol:
                    converged = True
                    break
                t.update()
        # Save the last normalize error
        last_e = norm_e
        self.is_fit = True
        self._n_iter = ctr
        self._delta = delta
        self._norm_e = norm_e
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

            print(f"\tFinal Norm Error: {np.round(100*norm_e, 2)}%")

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
