import pandas as pd
import numpy as np

__all__ = ['create_sub_by_item_matrix']
__author__ = ["Luke Chang"]
__license__ = "MIT"

# class Ratings(object):
#    def __init__():


def create_sub_by_item_matrix(df):
    ''' Convert a pandas long data frame of a single rating into a subject by item matrix
    
        Args:
            df: pandas dataframe instance.  Must have column names ['Subject','Item','Rating]
            
    '''
    if not isinstance(df,pd.DataFrame):
        raise ValueError('df must be pandas instance')
    if np.any([not x in df.columns for x in ['Subject','Item','Rating']]):
        raise ValueError("df must contain ['Subject','Item','Rating] as column names")
        
    ratings = pd.DataFrame(columns=df.Item.unique(),index=df['Subject'].unique())
    for row in df.iterrows():
        ratings.loc[row[1]['Subject'], row[1]['Item']] = float(row[1]['Rating'])
    return ratings.astype(float)

def train_test_split(self, n_train_items=20):
    ''' Split ratings matrix into train and test items.  adds test and train matrices to CF instance.

    Args:
        train_items: number of items for test dictionary or list of specific items

    '''
    
    self.n_train_items = n_train_items
    self.train = self.ratings.copy()
    self.train.loc[:,:] = np.full(self.ratings.shape, np.nan)
    self.test = self.ratings.copy()

    for sub in self.ratings.index:
        sub_train_rating_item =  np.random.choice(self.ratings.columns,replace=False, size=n_train_items)
        self.train.loc[sub, sub_train_rating_item] = self.ratings.loc[sub, sub_train_rating_item]
        self.test.loc[sub, sub_train_rating_item] = np.nan
    assert(~np.any(self.train==self.test))