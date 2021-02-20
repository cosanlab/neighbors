"""
Module that holds functions supporting various model methods. Not designed to be user-facing
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def sgd(
    data,
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
    sample_row,
    sample_col,
    learning_rate,
    verbose,
):
    """SGD Update"""

    error_history = []
    converged = False
    last_e = 0
    norm_rmse = np.inf
    delta = np.inf
    for this_iter in range(n_iterations):

        # Shuffle training indices
        training_indices = np.arange(len(sample_row))
        np.random.shuffle(training_indices)

        if verbose and this_iter > 0 and this_iter % 10 == 0:
            disp_norm_error = np.round(100 * norm_rmse, 2)
            disp_delta = np.round(delta, 4)
            # Numba doesn't like f-strings
            print(
                "Iter:",
                this_iter,
                " Norm RMSE:",
                disp_norm_error,
                "%  Delta Convg:",
                disp_delta,
                "||",
                tol,
            )

        # Loop over every data point in the training set, make a prediction, calculate error, and update biases and eights
        # Because we're iterating an user-item combo at a time, track total error
        total_error = 0
        for idx in training_indices:
            u = sample_row[idx]
            i = sample_col[idx]

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

        # Normalize the current error with respect to the range of the dataset
        rmse = np.sqrt(total_error / len(training_indices))
        norm_rmse = rmse / data_range
        error_history.append(norm_rmse)

        # Compute the delta to see if we should stop iterating
        delta = np.abs(np.abs(norm_rmse) - np.abs(last_e))
        # If we've converged break out of iteration
        if delta < tol:
            converged = True
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


@nb.njit(cache=True)
def mult(X, W, H, data_range, eps, tol, n_iterations, verbose):
    """Lee & Seung (2001) multiplicative update rule"""

    last_e = 0
    error_history = []
    converged = False
    norm_rmse = np.inf
    delta = np.inf

    for this_iter in range(n_iterations):

        if verbose and this_iter > 0 and this_iter % 10 == 0:
            disp_norm_error = np.round(100 * norm_rmse, 2)
            disp_delta = np.round(delta, 4)
            # Numba doesn't like f-strings
            print(
                "Iter:",
                this_iter,
                " Norm RMSE:",
                disp_norm_error,
                "%  Delta Convg:",
                disp_delta,
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
        # error = np.linalg.norm(X - W @ H, ord="fro")

        # Normalize current error with respect to max of dataset
        norm_rmse = rmse / data_range
        error_history.append(norm_rmse)

        # Compute delta to see if we should stop iterating
        delta = np.abs(np.abs(norm_rmse) - np.abs(last_e))
        if delta < tol:
            converged = True
            break
        # Otherwise update the error
        last_e = norm_rmse

    return error_history, converged, this_iter, delta, norm_rmse, W, H
