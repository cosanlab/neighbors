"""
Included datasets
"""

import pandas as pd
import numpy as np
import pkg_resources
from .utils import check_random_state

__all__ = ["load_movielens100k", "load_toymat"]


def load_movielens100k():
    """
    Load the MovieLens 100k dataset fetched from: https://grouplens.org/datasets/movielens/. The version of this dataset is identical to the one include Surprise python package.

    Returns:
        pd.DataFrame: long-form dataframe user, item, rating, timestamp columns
    """
    stream = pkg_resources.resource_stream(__name__, "data/movielens100k.csv")
    return pd.read_csv(stream)


def load_toymat(users=50, items=100, random_state=None):
    """
    Generate a toy user x item dataframe

    Args:
        users (int, optional): number of users. Defaults to 50.
        items (int, optional): number of items. Defaults to 100.
        random_state ([type], optional): random state for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: users x items dataframe of ratings
    """

    random = check_random_state(random_state)
    rat = random.rand(users, items) * 50
    for x in np.arange(0, rat.shape[1], 5):
        rat[0 : int(users / 2), x] = rat[0 : int(users / 2), x] + x
    for x in np.arange(0, rat.shape[1], 3):
        rat[int(users / 2) : users, x] = rat[int(users / 2) : users, x] + x
    rat[int(users / 2) : users] = rat[int(users / 2) : users, ::-1]
    rat = pd.DataFrame(rat)
    rat.index.name = "User"
    rat.columns.name = "Item"
    return pd.DataFrame(rat)
