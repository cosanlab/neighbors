"""
Module that holds functions supporting various model methods. Not designed to be user-facing
"""

import numpy as np
import numba as nb


@nb.njit(cache=True, nogil=True)
def sgd(
    data,
    seed,
    global_bias,
    data_range,
    tol,
    user_bias,
    user_vecs,
    user_bias_reg,
    user_fact_reg,
    item_bias,
    item_vecs,
    item_bias_reg,
    item_fact_reg,
    n_iterations,
    row_indices,
    col_indices,
    learning_rate,
    verbose,
):
    """SGD Update. This implementation is nearly identitical the the SVD implementation used by Simon Funk in the Netflix challenge and implemented in Surprise with a few small differences. We currently only support a single learning rate for all parameters (Surprise supports independent learning rates, but doesn't use them by default), we don't train in batched epochs but rather over *all* training data in each iteration, and we force user and item factor values to be >=0 after each pass over the training data."""

    error_history = np.zeros((n_iterations))
    converged = False
    last_e = 0
    norm_rmse = np.inf
    delta = np.inf
    np.random.seed(seed)
    for this_iter in range(n_iterations):

        # Generate shuffled order to loop over training data
        # Recall that row_indices and col_indices need to be looped over simultaneously to properly index each training value at [row, col]
        training_indices = np.arange(len(row_indices))
        np.random.shuffle(training_indices)

        if verbose and this_iter > 0 and this_iter % 10 == 0:
            disp_norm_error = np.round(100 * norm_rmse, 2)
            # Numba doesn't like f-strings
            print(
                "Iter:",
                this_iter,
                " Norm RMSE:",
                disp_norm_error,
                "%  Delta Convg:",
                delta,
                "||",
                tol,
            )

        # Loop over every data point in the training set, make a prediction, calculate error, and update biases and vectors
        # Very similar to Surprise's SVD implementation: https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/matrix_factorization.pyx#L159
        # Because we're iterating an user-item combo at a time, track total error
        total_error = 0
        for idx in training_indices:
            u = row_indices[idx]
            i = col_indices[idx]

            prediction = global_bias + user_bias[u] + item_bias[i]
            prediction += user_vecs[u, :] @ item_vecs[:, i]

            # Use changes in e to determine tolerance
            e = data[u, i] - prediction  # error

            # Update biases
            user_bias[u] += learning_rate * (e - user_bias_reg * user_bias[u])
            item_bias[i] += learning_rate * (e - item_bias_reg * item_bias[i])

            # Update latent factors
            user_vecs[u, :] += learning_rate * (
                e * item_vecs[:, i] - user_fact_reg * user_vecs[u, :]
            )
            item_vecs[:, i] += learning_rate * (
                e * user_vecs[u, :] - item_fact_reg * item_vecs[:, i]
            )

            # Keep track of total squared error
            total_error += np.power(e, 2)

        # Force non-negativity. Surprise does this per-epoch via re-initialization. We do this per sweep over all training data, e.g. see: https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/matrix_factorization.pyx#L671
        user_vecs = np.maximum(user_vecs, 0)
        item_vecs = np.maximum(item_vecs, 0)

        # Normalize the current error with respect to the range of the dataset
        rmse = np.sqrt(total_error / len(training_indices))
        norm_rmse = rmse / data_range
        error_history[this_iter] = norm_rmse

        # Compute the delta to see if we should stop iterating
        delta = np.abs(np.abs(norm_rmse) - np.abs(last_e))
        # If we've converged break out of iteration
        if delta < tol:
            converged = True
            error_history = error_history[: this_iter + 1]
            break
        # Otherwise update the error
        last_e = norm_rmse

    return (
        error_history,
        converged,
        this_iter,
        delta,
        norm_rmse,
        user_bias,
        user_vecs,
        item_bias,
        item_vecs,
    )


@nb.njit(cache=True, nogil=True)
def mult(X, W, H, data_range, eps, tol, n_iterations, verbose):
    """Lee & Seung (2001) multiplicative update rule"""

    last_e = 0
    error_history = np.zeros((n_iterations))
    converged = False
    norm_rmse = np.inf
    delta = np.inf

    for this_iter in range(n_iterations):

        if verbose and this_iter > 0 and this_iter % 10 == 0:
            disp_norm_error = np.round(100 * norm_rmse, 2)
            # Numba doesn't like f-strings
            print(
                "Iter:",
                this_iter,
                " Norm RMSE:",
                disp_norm_error,
                "%  Delta Convg:",
                delta,
                "||",
                tol,
            )

        # Update H
        numer = W.T @ X
        denom = W.T @ W @ H + eps
        H *= numer
        H /= denom

        # Update W
        numer = X @ H.T
        denom = W @ H @ H.T + eps
        W *= numer
        W /= denom

        # Make prediction and get error
        errors = X - W @ H
        rmse = np.sqrt(np.mean(np.power(errors, 2)))

        # Normalize current error with respect to max of dataset
        norm_rmse = rmse / data_range
        error_history[this_iter] = norm_rmse

        # Compute delta to see if we should stop iterating
        delta = np.abs(np.abs(norm_rmse) - np.abs(last_e))
        if delta < tol:
            converged = True
            error_history = error_history[: this_iter + 1]
            break
        # Otherwise update the error
        last_e = norm_rmse

    return error_history, converged, this_iter, delta, norm_rmse, W, H
