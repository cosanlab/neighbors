import numpy as np
import pandas as pd
from emotioncf.core import CF, create_sub_by_item_matrix


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

def test_CF():
    n_train_items = 40
    cf = CF(simulate_data(data_type='data_wide'),n_train_items=n_train_items)
    assert cf.n_train_items == n_train_items
    
    # Test train_test_split()
    assert np.sum(np.sum(cf.test.isnull(),axis=1))/cf.test.shape[0] == n_train_items
    assert np.sum(np.sum(cf.train.isnull(),axis=1))/cf.train.shape[0] == 100-n_train_items
    assert(~np.any(cf.train==cf.test))
   


