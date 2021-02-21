"""
Results dataclass to conveniently hold fit statistics from a model
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Results:
    """This class holds results from all models and is returned when a `model.summary()` method is called."""

    algorithm: str
    rmse: np.floating = field(init=True, repr=False)
    mse: np.floating = field(init=True, repr=False)
    mae: np.floating = field(init=True, repr=False)
    correlation: np.floating = field(init=True, repr=False)
    n_items: int
    n_users: int
    # sub_rmse: np.ndarray
    # sub_corr: np.ndarray
    predictions: pd.DataFrame = field(init=True, repr=False)
