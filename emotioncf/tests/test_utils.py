"""
Test utility functions
"""
import pandas as pd
from emotioncf import create_sub_by_item_matrix


def test_create_sub_by_item_matrix(simulate_long_data):
    rating = create_sub_by_item_matrix(simulate_long_data)
    assert isinstance(rating, pd.DataFrame)
    assert rating.shape == (50, 100)

    renamed = simulate_long_data.rename(
        columns={"Subject": "A", "Item": "B", "Rating": "C"}
    )
    rating = create_sub_by_item_matrix(renamed, columns=["A", "B", "C"])
    assert isinstance(rating, pd.DataFrame)
    assert rating.shape == (50, 100)
