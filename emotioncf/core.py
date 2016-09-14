from __future__ import division
from scipy import linalg
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import pearsonr
from copy import deepcopy
from emotioncf.algorithms import NNMF, knn_similarity, knn_predict
from emotioncf.algorithms import nmf_multiplicative_fit, nmf_multiplicative_predict

__all__ = ['CF','create_sub_by_item_matrix','get_mse']
__author__ = ["Luke Chang"]
__license__ = "MIT"

class CF(object):
    def __init__(self,
                ratings,
                n_train_items=None):

        if not isinstance(ratings,pd.DataFrame):
            raise ValueError('ratings must be a pandas dataframe instance')            
        self.ratings = ratings

        if n_train_items is None:
            self.n_train_items = None
            self.train = None
            self.test = None
        else:
            self.train_test_split(n_train_items=n_train_items)
        
        # Initialize
        self.subject_similarity = None
        self.H = None
        self.W = None
        self.nnmf_sgd = None
    
    def __repr__(self):
        return '%s.%s(rating=%s, n_train_items=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.ratings.shape,
            self.n_train_items
            )

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

    def fit(self, method='nnmf_sgd', **kwargs):
        ''' Fit collaborative model to training data

        Args:
            method: type of algorithm to fit ['knn','nnmf_multiplicative','nnmf_sgd']
            params: dictionary of parameters to pass to algorithm

        '''
        print(method)
        print(kwargs)
        if self.train is None:
            raise ValueError('Make sure you have run train_test_split() method.')
        
        if method is 'knn':
            if 'k' not in kwargs:
                raise ValueError('Make sure you pass a valid "k".')
            self.subject_similarity = knn_similarity(self.train, **kwargs)

        if method is 'nnmf_multiplicative':
            self.W, self.H = nmf_multiplicative(self.train, n_components=params['n_factors'], max_iter=params['max_iterations'])
        
        if method is 'nnmf_sgd':
            if 'n_iterations' not in kwargs:
                raise ValueError('Make sure you pass n_iterations.')

            self.nnmf_sgd = NNMF(self.train, 
                            # mask=mask,
                            learning='sgd', 
                            **kwargs)
            self.nnmf_sgd.train(n_iterations)

    def predict(self, method='nnmf_sgd', **kwargs):
        ''' Predict new data after fitting collaborative filtering algorithm to data
        
        Args:
            method: type of algorithm to use in prediction ['knn','nnmf_multiplicative','nnmf_sgd']

        '''

        if method is 'knn':
            if self.subject_similarity is None:
                raise ValueError('Make sure you have run train method.')
            self.predicted_ratings = knn_predict(self.train, self.subject_similarity, k = params['k'])

        if method is 'nnmf_multiplicative':
            if self.H is None:
                raise ValueError('Make sure you have run train method.')

        if method is 'nnmf_sgd':
            if self.nnmf_sgd is None:
                raise ValueError('Make sure you have run train method.')
            self.predicted_ratings = self.nnmf_sgd.predict()

    def plot_predictions(self, mask=None):
        
        if self.train is None:
            raise ValueError('Need to run split_train_test() first.')

        if self.predicted_ratings is None:
            raise ValueError('Need to predict ratings first.')
        
        f, ax = plt.subplots(nrows=1,ncols=3, figsize=(15,8))
        sns.heatmap(self.ratings,vmax=100,vmin=0,ax=ax[0],square=False)
        ax[0].set_title('Actual User/Item Ratings')
        sns.heatmap(self.predicted_ratings,vmax=100,vmin=0,ax=ax[1],square=False)
        ax[1].set_title('Predicted User/Item Ratings')
        if isinstance(self.test,pd.DataFrame):
            actual = self.test.values.flatten()
        elif isinstance(self.test,np.ndarray):
            actual = self.test.flatten()
        else:
            raise ValueError('Make sure self.test are pandas data frame or numpy array.')
        if isinstance(self.predicted_ratings,pd.DataFrame):
            pred = self.predicted_ratings.values.flatten()
        elif isinstance(self.predicted_ratings,np.ndarray):
            pred = self.predicted_ratings.flatten()
        else:
            raise ValueError('Make sure predicted_ratings are pandas data frame or numpy array.')
        if mask is not None:
            if isinstance(mask,pd.DataFrame):
                mask = mask.values.flatten()
            elif isinstance(mask,np.ndarray):
                mask = mask.flatten()
            else:
                raise ValueError('Make sure mask is a pandas data frame or numpy array.')
            pred = pred[~mask]
            actual = actual[~mask]
        ax[2].scatter(actual[(~np.isnan(actual)) & (~np.isnan(pred))],pred[(~np.isnan(actual)) & (~np.isnan(pred))])
        ax[2].set_xlabel('Actual Ratings')
        ax[2].set_ylabel('Predicted Ratings')
        ax[2].set_title('Predicted Ratings')
        r = pearsonr(actual[(~np.isnan(actual)) & (~np.isnan(pred))],pred[(~np.isnan(actual)) & (~np.isnan(pred))])
        print('Correlation: %s' % r[0])
        return f, r
    
    def subject_correlations(self, mask=None):
        '''Calculate observed/predicted correlation for each subject in matrix'''
        if mask is not None:
            r = []
            for i,sub in enumerate(self.test.index):
                r.append(pearsonr(self.test.loc[sub,mask[i,:]],self.predicted_ratings[i,mask[i,:]])[0])
        else:
            r = []
            for i,sub in enumerate(self.test.index):
                r.append(pearsonr(self.test.loc[sub,:],self.predicted_ratings[i,:])[0])
        return r

    def evaluate(pred, actual, metric='correlation'):
        ''' Evaluate overall performance of prediction
            Args:
                pred: pd.DataFrame instance of predicted ratings
                actual: pd.DataFrame instance of true ratings
                metric: type of metric to evaluate can be ["correlation","mse","accuracy"]
        '''
        if metric is 'correlation':
            actual = actual.values.flatten()
            pred = pred.values.flatten()
            return pearsonr(pred[(~np.isnan(actual)) & (~np.isnan(pred))], actual[(~np.isnan(actual)) & (~np.isnan(pred))])[0]
        elif metric is 'mse':
            actual = actual.values.flatten()
            pred = pred.values.flatten()
            return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)
        elif metric is 'accuracy':
            return np.sum(pred==actual)
        else:
            raise ValueError('Metric must be ["correlation","mse","accuracy"]')

def get_mse(pred, actual, mask=None):
    ''' Get Mean Squared Error ignoring Missing Values '''
    if mask is not None:
        pred = pred[mask]
        actual = actual[mask]
        return np.mean((pred-actual)**2)
    else:
        actual = actual.flatten()
        pred = pred.flatten()
        return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)

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

    