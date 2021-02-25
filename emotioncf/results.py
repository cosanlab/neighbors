"""
Results dataclass to conveniently hold fit statistics from a model
"""

# from dataclasses import dataclass, field
# import numpy as np
# import pandas as pd


# @dataclass(frozen=True)
# class Results:
#     """This class holds results from all models and is returned when a `model.summary()` method is called."""

#     algorithm: str
#     rmse: np.floating = field(init=True, repr=True)
#     mse: np.floating = field(init=True, repr=True)
#     mae: np.floating = field(init=True, repr=True)
#     correlation: np.floating = field(init=True, repr=True)
#     n_items: int
#     n_users: int
#     # sub_rmse: np.ndarray
#     # sub_corr: np.ndarray
#     predictions: pd.DataFrame = field(init=True, repr=False)

#     def to_df(self):
#         df = pd.DataFrame([self.rmse, self.mae, self.correlation])
#         df["metric"] = ["rmse", "mae", "correlation"]
#         df = df.melt(id_vars="metric", var_name="dataset", value_name="score")
#         df = (
#             df[["dataset", "metric", "score"]]
#             .sort_values(by=["dataset", "metric"])
#             .reset_index(drop=True)
#         )
#         return df
