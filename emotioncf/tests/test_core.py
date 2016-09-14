import numpy as np
import pandas as pd
from emotioncf.cf import Mean, KNN, NNMF_multiplicative, NNMF_sgd
from emotioncf.data import create_sub_by_item_matrix

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

def basecf_method_tests(cf=None):
    assert cf.predicted_ratings.shape == (50,100)
    mse = cf.get_mse()
    r = cf.get_corr()
    sub_r = cf.get_sub_corr()
    assert isinstance(mse,float)
    assert isinstance(r,float)
    assert isinstance(sub_r,np.ndarray)
    assert len(sub_r) == cf.ratings.shape[0]
    assert mse > 0
    assert r > 0
    assert np.mean(sub_r) > 0
    print(('mse: %s') % mse)
    print(('r: %s') % r)
    print(('mean sub r: %s') % np.mean(sub_r))

def test_create_sub_by_item_matrix():
    rating = create_sub_by_item_matrix(simulate_data(data_type='data_long'))
    assert isinstance(rating,pd.DataFrame)
    assert rating.shape == (50,100)
    
def test_cf_mean():
    cf = Mean(simulate_data(data_type='data_wide'))
    cf.fit()
    cf.predict()
    basecf_method_tests(cf=cf)

def test_cf_knn():
    cf = KNN(simulate_data(data_type='data_wide'))
    cf.fit()
    cf.predict()
    basecf_method_tests(cf=cf)

    cf.fit(metric='correlation')
    cf.predict(k=10)
    basecf_method_tests(cf=cf)

    cf.fit(metric='cosine')
    cf.predict(k=10)
    basecf_method_tests(cf=cf)

def test_cf_nnmf_multiplicative():
    cf = NNMF_multiplicative(simulate_data(data_type='data_wide'))
    cf.fit()
    cf.predict()
    basecf_method_tests(cf=cf)


