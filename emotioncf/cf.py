from __future__ import division
from scipy import linalg
import os
import pandas as pd
import numpy as np
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

		''' Get overall mean squared error for predicted compared to actual for all items and subjects. '''
		
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

		''' Get overall correlation for predicted compared to actual for all items and subjects. '''

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
	
	def get_sub_corr(self, mask=None):

		'''Calculate observed/predicted correlation for each subject in matrix'''

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		if mask is not None:
			r = []
			for i,sub in enumerate(self.ratings.index):
				r.append(pearsonr(self.ratings.loc[sub,mask[i,:]],self.predicted_ratings[i,mask[i,:]])[0])
		else:
			r = []
			for i,sub in enumerate(self.ratings.index):
				r.append(pearsonr(self.ratings.loc[sub,:], self.predicted_ratings.loc[sub,:])[0])
		return np.array(r)

	def plot_predictions(self, mask=None):

		''' Create plot of actual and predicted ratings'''

		import matplotlib.pyplot as plt
		import seaborn as sns		
		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')
		
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

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')

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

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')

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

class NNMF_multiplicative(BaseCF):
	''' Train non negative matrix factorization model using multiplicative updates.  
		Allows masking to only learn the training weights.

		Based on http://stackoverflow.com/questions/22767695/
		python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat
	
	'''
	
	def __init__(self, ratings):
		super(NNMF_multiplicative, self).__init__(ratings)
		self.H = None
		self.W = None
	
	def fit(self, 
		n_factors=None, 
		max_iterations=100,
		error_limit=1e-6, 
		fit_error_limit=1e-6, 
		verbose=False,
		**kwargs):

		''' Fit NNMF collaborative filtering model to training data using multiplicative updating.

		Args:
			n_factors (int): Number of factors or components
			max_iterations (int):  maximum number of interations (default=100)
			error_limit (float): error tolerance (default=1e-6)
			fit_error_limit (float): fit error tolerance (default=1e-6)
			verbose (bool): verbose output during fitting procedure (default=True)
		'''

		mask = ~np.isnan(self.ratings.values)
		# train[train.isnull()] = 0
		# X = X.values

		eps = 1e-5

		n_samples, n_features = self.ratings.shape
		if n_factors is None:
			n_factors = n_features

		# Initial guesses for solving X ~= WH. H is random [0,1] scaled by sqrt(X.mean() / n_factors)
		avg = np.sqrt(np.nanmean(self.ratings)/n_factors)
		self.H = avg*np.random.rand(n_features, n_factors) # H = Y
		self.W = avg*np.random.rand(n_samples, n_factors)   # W = A
		masked_X = mask * self.ratings.values
		X_est_prev = np.dot(self.W, self.H)

		for i in range(1, max_iterations + 1):
			# Update W: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
			self.W *= np.dot(masked_X, self.H.T) / np.dot(mask * np.dot(self.W,self.H), self.H.T)
			self.W = np.maximum(self.W, eps)

			# Update H: Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
			self.H *= np.dot(self.W.T, masked_X) / np.dot(self.W.T, mask * np.dot(self.W,self.H))
			self.H = np.maximum(self.H, eps)

			# Evaluate
			if i % 5 == 0 or i == 1 or i == max_iterations:
				X_est = np.dot(self.W,self.H)
				err = mask * (X_est_prev - X_est)
				fit_residual = np.sqrt(np.sum(err ** 2))
				X_est_prev = X_est
				curRes = linalg.norm(mask * (self.ratings.values - X_est), ord='fro')
				if verbose:
					print('Iteration {}:'.format(i)),
					print('fit residual', np.round(fit_residual, 4)),
					print('total residual', np.round(curRes, 4))
				if curRes < error_limit or fit_residual < fit_error_limit:
					break
		self.is_fit = True

	def predict(self, **kwargs):

		''' Predict Subject's missing items using NNMF with multiplicative updating

			Args:
				ratings: pandas dataframe instance of ratings
				k: number of closest neighbors to use
			Returns:
				predicted_rating: (pd.DataFrame instance) adds field to object instance
		'''

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')

		self.predicted_ratings = self.ratings.copy()
		self.predicted_ratings.loc[:,:] = np.dot(self.W, self.H)
		self.is_predict = True

class NNMF_sgd(BaseCF):
	''' Train non negative matrix factorization model using stochastic gradient descent.  
		Allows masking to only learn the training weights.

		This code is based off of Ethan Rosenthal's excellent tutorial 
		on collaborative filtering https://blog.insightdatascience.com/
		explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#.kkr7mzvr2
	
	'''
	
	def __init__(self, ratings):
		super(NNMF_sgd, self).__init__(ratings)


	def fit(self, 
		n_factors=None, 
		max_iterations=100,
		error_limit=1e-6, 
		fit_error_limit=1e-6, 
		verbose=False,
		**kwargs):

		''' Fit NNMF collaborative filtering model to training data using multiplicative updating.

		Args:
			n_factors (int): Number of factors or components
			max_iterations (int):  maximum number of interations (default=100)
			error_limit (float): error tolerance (default=1e-6)
			fit_error_limit (float): fit error tolerance (default=1e-6)
			verbose (bool): verbose output during fitting procedure (default=True)
		'''

	def predict(self, **kwargs):

		''' Predict Subject's missing items using NNMF with multiplicative updating

			Args:
				ratings: pandas dataframe instance of ratings
				k: number of closest neighbors to use
			Returns:
				predicted_rating: (pd.DataFrame instance) adds field to object instance
		'''

