import pandas as pd

__all__ = ["create_sub_by_item_matrix"]

# class Ratings(object):
#    def __init__():


def create_sub_by_item_matrix(df):

    """Convert a pandas long data frame of a single rating into a subject by item matrix

    Args:
        df: pandas dataframe instance.  Must have column names ['Subject','Item','Rating]

    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be pandas instance")
    if not all([x in df.columns for x in ["Subject", "Item", "Rating"]]):
        raise ValueError("df must contain ['Subject','Item','Rating] as column names")

    ratings = df.pivot(index="Subject", columns="Item", values="Rating").reset_index(
        drop=True
    )
    return ratings.astype(float)
