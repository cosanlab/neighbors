import numpy as np
import pandas as pd
from emotioncf.cf import Mean, KNN, NNMF_multiplicative, NNMF_sgd
from emotioncf.data import create_sub_by_item_matrix
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# TODO: Look into cleaning tests and using pytest fixtures
def simulate_data(data_type="data_long"):
    i = 100
    s = 50
    rat = np.random.rand(s, i) * 50
    for x in np.arange(0, rat.shape[1], 5):
        rat[0 : int(s / 2), x] = rat[0 : int(s / 2), x] + x
    for x in np.arange(0, rat.shape[1], 3):
        rat[int(s / 2) : s, x] = rat[int(s / 2) : s, x] + x
    rat[int(s / 2) : s] = rat[int(s / 2) : s, ::-1]
    rat = pd.DataFrame(rat)
    if data_type == "data_long":
        out = pd.DataFrame(columns=["Subject", "Item", "Rating"])
        for row in rat.iterrows():
            sub = pd.DataFrame(columns=out.columns)
            sub["Rating"] = row[1]
            sub["Item"] = rat.columns
            sub["Subject"] = row[0]
            out = out.append(sub)
        return out
    elif data_type == "data_wide":
        return rat


def basecf_method_test(cf=None, data=None):
    assert cf.train_mask.shape == (50, 100)
    assert cf.predicted_ratings.shape == (50, 100)
    mse = cf.get_mse(data=data)
    r = cf.get_corr(data=data)
    sub_r = cf.get_sub_corr(data=data)
    sub_mse = cf.get_sub_mse(data=data)
    assert isinstance(mse, float)
    assert isinstance(r, float)
    assert isinstance(sub_r, np.ndarray)
    assert len(sub_r) == cf.ratings.shape[0]
    assert isinstance(sub_mse, np.ndarray)
    assert len(sub_mse) == cf.ratings.shape[0]
    assert mse > 0
    assert r > 0
    assert np.mean(sub_r) > 0
    print(data)
    print(("mse: %s") % mse)
    print(("r: %s") % r)
    print(("mean sub r: %s") % np.mean(sub_r))

    df = cf.to_long_df()
    assert isinstance(df, pd.DataFrame)
    if cf.is_predict:
        assert "Condition" in df.columns
        assert "Observed" in df["Condition"].unique()
        assert "Predicted" in df["Condition"].unique()
        assert df.shape[0] == cf.ratings.shape[0] * cf.ratings.shape[1] * 2
    if cf.is_mask:
        assert "Mask" in df.columns
    cf.plot_predictions(data=data)
    plt.close()


def basecf_method_all_tests(cf=None):
    basecf_method_test(cf=cf, data="all")
    basecf_method_test(cf=cf, data="train")
    basecf_method_test(cf=cf, data="test")


def test_create_sub_by_item_matrix():
    rating = create_sub_by_item_matrix(simulate_data(data_type="data_long"))
    assert isinstance(rating, pd.DataFrame)
    assert rating.shape == (50, 100)


def test_cf_mean():
    cf = Mean(simulate_data(data_type="data_wide"))
    cf.fit()
    cf.predict()
    cf = Mean(simulate_data(data_type="data_wide"))
    cf.split_train_test(n_train_items=20)
    cf.fit()
    cf.predict()
    basecf_method_all_tests(cf=cf)
    cf.fit(dilate_ts_n_samples=2)
    cf.predict()
    basecf_method_all_tests(cf=cf)


def test_cf_knn():
    cf = KNN(simulate_data(data_type="data_wide"))
    cf.fit(metric="pearson")
    cf.predict()
    cf.split_train_test(n_train_items=50)
    cf.fit()
    cf.predict()
    basecf_method_all_tests(cf=cf)

    cf.fit(metric="correlation")
    cf.predict(k=10)
    basecf_method_all_tests(cf=cf)

    cf.fit(metric="cosine")
    cf.predict(k=10)
    basecf_method_all_tests(cf=cf)


def test_cf_knn_dil():
    cf = KNN(simulate_data(data_type="data_wide"))
    cf.split_train_test(n_train_items=20)
    cf.fit(dilate_ts_n_samples=2, metric="pearson")
    cf.predict()
    basecf_method_all_tests(cf=cf)

    cf.fit(dilate_ts_n_samples=2, metric="correlation")
    cf.predict()
    basecf_method_all_tests(cf=cf)


def test_cf_nnmf_multiplicative():
    cf = NNMF_multiplicative(simulate_data(data_type="data_wide"))
    cf.fit()
    cf.predict()
    cf.split_train_test(n_train_items=50)
    cf.fit()
    basecf_method_all_tests(cf=cf)

    cf.fit(dilate_ts_n_samples=2)
    cf.predict()
    basecf_method_all_tests(cf=cf)


def test_cf_nnmf_sgd():
    cf = NNMF_sgd(simulate_data(data_type="data_wide"))
    cf.fit(
        n_iterations=20,
        user_fact_reg=0,
        item_fact_reg=0,
        user_bias_reg=0,
        item_bias_reg=0,
        learning_rate=0.001,
    )
    cf.predict()

    cf.split_train_test(n_train_items=50)
    cf.fit(
        n_iterations=20,
        user_fact_reg=0,
        item_fact_reg=0,
        user_bias_reg=0,
        item_bias_reg=0,
        learning_rate=0.001,
    )
    basecf_method_all_tests(cf=cf)

    cf.fit(
        n_iterations=20,
        user_fact_reg=0,
        item_fact_reg=0,
        user_bias_reg=0,
        item_bias_reg=0,
        learning_rate=0.001,
        dilate_ts_n_samples=2,
    )
    cf.predict()
    basecf_method_all_tests(cf=cf)


def test_downsample():
    cf = Mean(simulate_data(data_type="data_wide"))
    cf.downsample(sampling_freq=10, target=2, target_type="samples")
    assert cf.ratings.shape == (50, 50)
    cf = Mean(simulate_data(data_type="data_wide"))
    cf.downsample(sampling_freq=10, target=5, target_type="hz")
    assert cf.ratings.shape == (50, 50)
    cf = Mean(simulate_data(data_type="data_wide"))
    cf.downsample(sampling_freq=10, target=2, target_type="seconds")
    assert cf.ratings.shape == (50, 5)
    cf = Mean(simulate_data(data_type="data_wide"))
    cf.split_train_test(n_train_items=20)
    cf.fit()
    cf.predict()
    cf.downsample(sampling_freq=10, target=2, target_type="samples")
    assert cf.ratings.shape == (50, 50)
    assert cf.train_mask.shape == (50, 50)
    assert cf.predicted_ratings.shape == (50, 50)

    cf = Mean(simulate_data(data_type="data_wide"))
    cf.split_train_test(n_train_items=20)
    cf.fit(dilate_ts_n_samples=2)
    cf.predict()
    cf.downsample(sampling_freq=10, target=2, target_type="samples")
    assert cf.dilated_mask.shape == (50, 50)
    assert cf.train_mask.shape == (50, 50)
    assert cf.predicted_ratings.shape == (50, 50)
