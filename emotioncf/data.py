"""
Functions that perform various data transformations

"""
import pandas as pd

__all__ = ["create_sub_by_item_matrix"]


def create_sub_by_item_matrix(df, columns=None, force_float=True, errors="raise"):

    """Convert a pandas long data frame of a single rating into a subject by item matrix

    Args:
        df (Dataframe): input dataframe
        columns (list): list of length 3 with dataframe columns to use for reshaping. The first value should reflect unique individual identifier ("Subject"), the second a unique item identifier ("Item", "Timepoint"), and the last the rating made by the individual on that item ("Rating"). Defaults to ["Subject", "Item", "Rating"]
        force_float (bool): force the resulting output to be float data types with errors being set to NaN; Default True
        errors (string): how to handle errors in pd.to_numeric; Default 'raise'

    Return:
        pd.DataFrame: user x item rating Dataframe

    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be pandas instance")
    if columns is None:
        columns = ["Subject", "Item", "Rating"]
    if not all([x in df.columns for x in columns]):
        raise ValueError(
            f"df is missing some or all of the following columns: {columns}"
        )

    ratings = df[columns]
    ratings = ratings.pivot(index=columns[0], columns=columns[1], values=columns[2])
    try:
        if force_float:
            ratings = ratings.apply(pd.to_numeric, errors=errors)
    except ValueError as e:
        print(
            "Auto-converting data to floats failed, probably because you have non-numeric data in some rows. You can set errors = 'coerce' to set these failures to NaN"
        )
        raise (e)

    return ratings
