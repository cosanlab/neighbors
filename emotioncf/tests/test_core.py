import numpy as np
import pandas as pd
from emotioncf.cf import Mean, KNN, NNMF_multiplicative, NNMF_sgd
from emotioncf.data import create_sub_by_item_matrix

def simulate_data(data_type = 'data_long'):
    i = 100
    s = 50
    rat = np.random.rand(s,i)*50
    for x in np.arange(0,rat.shape[1],5):
        rat[0:s/2,x] = rat[0:s/2,x] + x
    for x in np.arange(0,rat.shape[1],3):
        rat[(s/2):s,x] = rat[(s/2):s,x] + x
    rat[(s/2):s] = rat[(s/2):s,::-1]
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
    cf.split_train_test(n_train_items=20)
    assert cf.train_mask.shape == (50,100)
    for x in cf.train_mask.sum(axis=1):
        assert cf.n_train_items == x 
    mse_all = cf.get_mse(data='all')
    r_all = cf.get_corr(data='all')
    sub_r_all = cf.get_sub_corr(data='all')
    assert isinstance(mse_all,float)
    assert isinstance(r_all,float)
    assert isinstance(sub_r_all,np.ndarray)
    assert len(sub_r_all) == cf.ratings.shape[0]
    assert mse_all > 0
    assert r_all > 0
    assert np.mean(sub_r_all) > 0
    print('All')
    print(('mse: %s') % mse_all)
    print(('r: %s') % r_all)
    print(('mean sub r: %s') % np.mean(sub_r_all))
    
    mse_tr = cf.get_mse(data='train')
    r_tr = cf.get_corr(data='train')
    sub_r_tr = cf.get_sub_corr(data='train')
    assert isinstance(mse_tr,float)
    assert isinstance(r_tr,float)
    assert isinstance(sub_r_tr,np.ndarray)
    assert len(sub_r_tr) == cf.ratings.shape[0]
    assert mse_tr > 0
    assert r_tr > 0
    assert np.mean(sub_r_tr) > 0
    print('Train')
    print(('mse: %s') % mse_tr)
    print(('r: %s') % r_tr)
    print(('mean sub r: %s') % np.mean(sub_r_tr))
    
    mse_te = cf.get_mse(data='test')
    r_te = cf.get_corr(data='test')
    sub_r_te = cf.get_sub_corr(data='test')
    assert isinstance(mse_te,float)
    assert isinstance(r_te,float)
    assert isinstance(sub_r_te,np.ndarray)
    assert len(sub_r_te) == cf.ratings.shape[0]
    assert mse_te > 0
    assert r_te > 0
    assert np.mean(sub_r_te) > 0
    print('Test')
    print(('mse: %s') % mse_te)
    print(('r: %s') % r_te)
    print(('mean sub r: %s') % np.mean(sub_r_te))

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
    
def test_cf_nnmf_sgd():
    cf = NNMF_sgd(simulate_data(data_type='data_wide'))
    cf.fit(n_iterations = 100,
           user_fact_reg=1,
           item_fact_reg=.001,
           user_bias_reg=0,
           item_bias_reg=0,
           learning_rate=.001)
    cf.predict()
    basecf_method_tests(cf=cf)
