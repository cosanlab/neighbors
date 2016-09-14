import numpy as np
import pandas as pd
from emotioncf.cf import Mean, KNN

def simulate_data(data_type = 'data_long'):
    rat = np.random.rand(50,100)*50
    for i in np.arange(0,rat.shape[1],5):
        rat[:,i] = rat[:,i] + i
    rat = pd.DataFrame(rat)
    if data_type is 'data_long':
        out = pd.DataFrame(columns=['Subject','Item','Rating'])
        for row in rat.iterrows():
            sub = pd.DataFrame(columns=out.columns)
            sub['Rating'] = row[1]
            sub['Item'] = rat.columns
            sub['Subject'] = row[0]
            out = out.append(sub)
        return out
    elif data_type is 'data_wide':
        return rat

def test_create_sub_by_item_matrix():
    rating = create_sub_by_item_matrix(simulate_data(data_type='data_long'))
    assert isinstance(rating,pd.DataFrame)
    assert rating.shape == (50,100)

def test_cf_mean():
    cf = Mean(simulate_data(data_type='data_wide'))
    cf.fit()
    cf.predict()
    assert cf.predicted_ratings.shape == (50,100)
    print(cf.get_mse())
    print(cf.get_corr())

def test_cf_knn():
    cf = KNN(simulate_data(data_type='data_wide'))
    cf.fit()
    cf.predict()
    assert cf.predicted_ratings.shape == (50,100)
    mse = cf.get_mse()
    r = cf.get_corr()
    assert isinstance(mse,float)
    assert isinstance(r,float)
    assert mse > 0
    assert r > 0
    print(mse)
    print(r)
    cf.fit(metric='correlation')
    cf.predict(k=10)
    assert cf.predicted_ratings.shape == (50,100)
    print(cf.get_mse())
    print(cf.get_corr())
    cf.fit(metric='cosine')
    cf.predict(k=10)
    assert cf.predicted_ratings.shape == (50,100)
    print(cf.get_mse())
    print(cf.get_corr())