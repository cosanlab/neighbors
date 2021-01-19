"""
Utility functions
"""

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["get_size_in_mb", "get_sparsity"]


def get_size_in_mb(arr):
    if isinstance(arr, (np.ndarray, csr_matrix)):
        return arr.data.nbytes / 1e6
    else:
        raise TypeError("input must by a numpy array or scipy csr sparse matrix")


def get_sparsity(arr):
    if isinstance(arr, (np.ndarray)):
        return 1 - (np.count_nonzero(arr) / arr.size)
    else:
        raise TypeError("input must be a numpy array")
