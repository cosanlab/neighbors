"""
Module that holds functions supporting various model methods. Not designed to be user-facing
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def sgd(
    data,
    global_bias,
    max_norm,
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
    error_history = []
    converged = False
    last_e = 0
    norm_e = np.inf
    delta = np.inf
    for this_iter in range(n_iterations):

        # Shuffle training indices
        training_indices = np.arange(len(sample_row))
        np.random.shuffle(training_indices)

        if verbose and this_iter > 0 and this_iter % 10 == 0:
            disp_norm_error = np.round(100 * norm_e, 2)
            disp_delta = np.round(delta, 4)
            # Numba doesn't like f-strings
            print(
                "Iter: ",
                this_iter,
                " Norm Error: ",
                disp_norm_error,
                "% Delta Convg: ",
                disp_delta,
                "||",
                tol,
            )

        # Loop over every data point in the training set, make a prediction, calculate error, and update biases and eights
        for idx in training_indices:
            u = sample_row[idx]
            i = sample_col[idx]

            prediction = global_bias + user_bias[u] + item_bias[i]
            prediction += user_vecs[u, :].dot(item_vecs[i, :].T)

            # Use changes in e to determine tolerance
            e = data[u, i] - prediction  # error

            # Update biases
            user_bias[u] += learning_rate * (e - user_bias_reg * user_bias[u])
            item_bias[i] += learning_rate * (e - item_bias_reg * item_bias[i])

            # Update latent factors
            user_vecs[u, :] += learning_rate * (
                e * item_vecs[i, :] - user_fact_reg * user_vecs[u, :]
            )
            item_vecs[i, :] += learning_rate * (
                e * user_vecs[u, :] - item_fact_reg * item_vecs[i, :]
            )

        # Normalize the current error with respect to the max of the dataset
        norm_e = np.abs(e) / max_norm
        error_history.append(norm_e)

        # Compute the delta to see if we should stop iterating
        delta = np.abs(np.abs(norm_e) - np.abs(last_e))
        if delta < tol:
            converged = True
            break

    return (
        error_history,
        converged,
        this_iter,
        delta,
        norm_e,
        user_bias,
        user_vecs,
        item_bias,
        item_vecs,
    )
