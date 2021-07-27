"""
Test data loading
"""
import pandas as pd
from neighbors import load_movielens100k, load_toymat


def test_load_movielens100k():
    df = load_movielens100k()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100000, 4)


def test_load_toy_useritem_data():
    df = load_toymat()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 100)
    df = load_toymat(100, 50)
    assert df.shape == (100, 50)
