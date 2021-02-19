"""
Define pytest "fixtures" aka the "Arrange" or "Setup" step of test-driven-development:

1. Arrange
2. Act
3. Assert

"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def simulate_wide_data():
    """Generate user x item dataframe with 50 rows and 100 columns"""

    np.random.seed(0)
    i = 100
    s = 50
    rat = np.random.rand(s, i) * 50
    for x in np.arange(0, rat.shape[1], 5):
        rat[0 : int(s / 2), x] = rat[0 : int(s / 2), x] + x
    for x in np.arange(0, rat.shape[1], 3):
        rat[int(s / 2) : s, x] = rat[int(s / 2) : s, x] + x
    rat[int(s / 2) : s] = rat[int(s / 2) : s, ::-1]
    return pd.DataFrame(rat)


@pytest.fixture(scope="module")
def simulate_long_data(simulate_wide_data):
    """Melt generated user x item dataframe to a (user * item, 3) shaped dataframe"""

    out = pd.DataFrame(columns=["Subject", "Item", "Rating"])
    for _, row in simulate_wide_data.iterrows():
        sub = pd.DataFrame(columns=out.columns)
        sub["Rating"] = row[1]
        sub["Item"] = simulate_wide_data.columns
        sub["Subject"] = row[0]
        out = out.append(sub)
    return out


@pytest.fixture(scope="module")
def simulate_simple_dataframe():
    ratings_dict = {
        "Subject": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        "Item": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "Rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3, 1],
    }
    return pd.DataFrame(ratings_dict)
