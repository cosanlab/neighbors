from __future__ import division
from scipy import linalg
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import pearsonr
from copy import deepcopy


__all__ = ['Mean',
			'KNN']
__author__ = ["Luke Chang"]
__license__ = "MIT"

# Notes might consider making a ratings data class that can accomodate timeseries and tensors

class BaseCF(object):

	''' Base Collaborative Filtering Class '''

	def __init__(self, ratings, mask=None):
		if not isinstance(ratings, pd.DataFrame):
			raise ValueError('ratings must be a pandas dataframe instance')			
		self.ratings = ratings
		self.predicted_ratings = None
		self.is_fit = False
		self.is_predict = False

		if mask is not None:
			self.mask = mask

	def __repr__(self):
		return '%s(rating=%s)' % (
			self.__class__.__name__,
			self.ratings.shape
			)

	def get_mse(self, mask=None):

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		''' Get Mean Squared Error ignoring Missing Values '''
		if mask is not None:
			return np.mean((self.predicted_ratings[mask]-self.ratings[mask])**2)
		else:
			actual = self.ratings.values.flatten()
			pred = self.predicted_ratings.values.flatten()
			return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)

	def get_corr(self, mask=None):

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		''' Get Correlation ignoring Missing Values '''
		if mask is not None:
			return pearsonr(self.predicted_ratings[mask],self.ratings[mask])[0]
		else:
			actual = self.ratings.values.flatten()
			pred = self.predicted_ratings.values.flatten()
			return pearsonr(pred[(~np.isnan(actual)) & (~np.isnan(pred))], actual[(~np.isnan(actual)) & (~np.isnan(pred))])[0]

class Mean(BaseCF):

	''' CF using Item Mean across subjects'''

	def __init__(self, ratings):
		super(Mean, self).__init__(ratings)
		self.mean = None

	def fit(self, **kwargs):

		''' Fit collaborative model to training data.  Calculate similarity between subjects across items

		Args:
			metric: type of similarity {"correlation","cosine"}
		'''

		self.mean = self.ratings.mean(skipna=True, axis=0)
		self.is_fit = True

	def predict(self, **kwargs):

		''' Predict missing items using other subject's item means.

			Args:
				k: number of closest neighbors to use
			Returns:
				predicted_rating: (pd.DataFrame instance) adds field to object instance
		'''

		self.predicted_ratings = self.ratings.copy()
		for row in self.ratings.iterrows():
			self.predicted_ratings.loc[row[0]] = self.mean
		self.is_predict = True

class KNN(BaseCF):

	''' K-Nearest Neighbors CF algorithm'''

	def __init__(self, ratings):
		super(KNN, self).__init__(ratings)
		self.subject_similarity = None

	def fit(self, metric='correlation', **kwargs):

		''' Fit collaborative model to training data.  Calculate similarity between subjects across items

		Args:
			metric: type of similarity {"correlation","cosine"}

		'''
	
		if metric is 'correlation':
			sim = pd.DataFrame(np.zeros((self.ratings.shape[0],self.ratings.shape[0])))
			sim.columns=self.ratings.index
			sim.index=self.ratings.index
			for x in self.ratings.iterrows():
				for y in self.ratings.iterrows():
					sim.loc[x[0],y[0]] = pearsonr(x[1][(~x[1].isnull()) & (~y[1].isnull())],y[1][(~x[1].isnull()) & (~y[1].isnull())])[0] 
		elif metric is 'cosine':
			sim = self.ratings.dot(self.ratings.T)
			norms = np.array([np.sqrt(np.diagonal(sim.values))])
			sim.loc[:,:] = (sim.values / norms / norms.T)
		self.subject_similarity = sim
		self.is_fit = True

	def predict(self, k=None, **kwargs):

		''' Predict Subject's missing items using similarity based collaborative filtering.

			Args:
				ratings: pandas dataframe instance of ratings
				k: number of closest neighbors to use
			Returns:
				predicted_rating: (pd.DataFrame instance) adds field to object instance
		'''
		pred = pd.DataFrame(np.zeros(self.ratings.shape))
		pred.columns = self.ratings.columns
		pred.index = self.ratings.index
		for row in self.ratings.iterrows():
			if k is not None:
				top_subjects = self.subject_similarity.loc[row[0]].drop(row[0]).sort_values(ascending=False)[0:k]
			else:
				top_subjects = self.subject_similarity.loc[row[0]].drop(row[0]).sort_values(ascending=False)
			for col in self.ratings.iteritems():
				pred.loc[row[0],col[0]] = np.dot(top_subjects,self.ratings.loc[top_subjects.index,col[0]].T)/len(top_subjects)
		self.predicted_ratings = pred
		self.is_predict = True
