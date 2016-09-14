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
				r.append(pearsonr(self.ratings.loc[sub,:],self.predicted_ratings[i,:])[0])
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

class NNMF_multiplicative(BaseCF):

	def nmf_multiplicative_fit(X, n_components=None, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6, verbose=True):
	''' Train non negative matrix factorization model using multiplicative updates.  
		Allows masking to only learn the training weights.

		Based on http://stackoverflow.com/questions/22767695/
		python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat
	
	'''
	
	mask = ~np.isnan(X.values)
	train[train.isnull()] = 0
	X = X.values

	eps = 1e-5

	n_samples, n_features = X.shape
	if n_components is None:
		n_components = n_features

	# Initial guesses for solving X ~= WH. H is random [0,1] scaled by sqrt(X.mean() / n_components)
	avg = np.sqrt(np.nanmean(X)/n_components)
	H = avg*np.random.rand(n_features, n_components) # H = Y
	W = avg*np.random.rand(n_samples, n_components)   # W = A
	masked_X = mask * X
	X_est_prev = np.dot(W, H)

	for i in range(1, max_iter + 1):
		# Update W: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
		W *= np.dot(masked_X, H.T) / np.dot(mask * np.dot(W,H), H.T)
#		 W = np.maximum(W, eps)

		# Update H: Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
		H *= np.dot(W.T, masked_X) / np.dot(W.T, mask * dot(W,H))
#		 H = np.maximum(H, eps)

		# Evaluate
		if i % 5 == 0 or i == 1 or i == max_iter:
			X_est = np.dot(W,H)
			err = mask * (X_est_prev - X_est)
			fit_residual = np.sqrt(np.sum(err ** 2))
			X_est_prev = X_est
			curRes = linalg.norm(mask * (X - X_est), ord='fro')
			if verbose:
				print('Iteration {}:'.format(i)),
				print('fit residual', np.round(fit_residual, 4)),
				print('total residual', np.round(curRes, 4))
			if curRes < error_limit or fit_residual < fit_error_limit:
				break
	return W, H

class NNMF_sgd(BaseCF):
	class NNMF():
	def __init__(self, 
				 ratings,
				 mask=None,
				 n_factors=40,
				 learning='sgd',
				 item_fact_reg=0.0, 
				 user_fact_reg=0.0,
				 item_bias_reg=0.0,
				 user_bias_reg=0.0,
				 learning_rate=0.1,
				 verbose=False):
		"""
		Train a matrix factorization model to predict empty 
		entries in a matrix. The terminology assumes a 
		ratings matrix which is ~ user x item

		This code is based off of Ethan Rosenthal's excellent tutorial 
		on collaborative filtering https://blog.insightdatascience.com/
		explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea#.kkr7mzvr2
		
		Params
		======
		ratings : (ndarray)
			User x Item matrix with corresponding ratings
		
		mask: (ndarray)
			Boolean matrix indicating missing values
		n_factors : (int)
			Number of latent factors to use in matrix 
			factorization model.  If None, will use full feature set
		learning : (str)
			Method of optimization. Options include 
			'sgd' or 'als'.
		
		item_fact_reg : (float)
			Regularization term for item latent factors
		
		user_fact_reg : (float)
			Regularization term for user latent factors
			
		item_bias_reg : (float)
			Regularization term for item biases
		
		user_bias_reg : (float)
			Regularization term for user biases
		
		verbose : (bool)
			Whether or not to printout training progress
		"""
		

		self.ratings = ratings
		self.mask = mask
		self.n_users, self.n_items = ratings.shape
		if n_factors is not None:
			self.n_factors = n_factors
		else:
			self.n_factors = self.n_items
		self.item_fact_reg = item_fact_reg
		self.user_fact_reg = user_fact_reg
		self.item_bias_reg = item_bias_reg
		self.user_bias_reg = user_bias_reg
		self.learning = learning
		if self.learning == 'sgd':
			self.learning_rate = learning_rate

			if self.mask is not None:
				self.sample_row, self.sample_col = self.mask.nonzero()
			else:
				self.sample_row, self.sample_col = self.ratings.nonzero()
			self.n_samples = len(self.sample_row)
		self._v = verbose

		self.initialize()
		
	def initialize(self):
		""" Initialize variables for matrix factorization """
		
		# initialize latent vectors		
		self.user_vecs = np.random.normal(scale=1./self.n_factors,
										  size=(self.n_users, self.n_factors))
		self.item_vecs = np.random.normal(scale=1./self.n_factors,
										  size=(self.n_items, self.n_factors))
		# Initialize biases
		if self.learning == 'sgd':
			self.user_bias = np.zeros(self.n_users)
			self.item_bias = np.zeros(self.n_items)
			if self.mask is not None:
				self.global_bias = np.mean(self.ratings[~mask])
			else:
				self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
				
	def als_step(self,
				 latent_vectors,
				 fixed_vecs,
				 ratings,
				 _lambda,
				 type='user'):
		"""
		One of the two ALS steps. Solve for the latent vectors
		specified by type.
		"""
		if type == 'user':
			# Precompute
			YTY = fixed_vecs.T.dot(fixed_vecs)
			lambdaI = np.eye(YTY.shape[0]) * _lambda

			for u in xrange(latent_vectors.shape[0]):
				latent_vectors[u, :] = solve((YTY + lambdaI), 
											 ratings[u, :].dot(fixed_vecs))
		elif type == 'item':
			# Precompute
			XTX = fixed_vecs.T.dot(fixed_vecs)
			lambdaI = np.eye(XTX.shape[0]) * _lambda
			
			for i in xrange(latent_vectors.shape[0]):
				latent_vectors[i, :] = solve((XTX + lambdaI), 
											 ratings[:, i].T.dot(fixed_vecs))
		return latent_vectors
	
	def train(self, n_iter=10):
		""" Train model for n_iter iterations. Can be called multiple times for further training."""
		ctr = 1
		while ctr <= n_iter:
			if ctr % 10 == 0 and self._v:
				print('\tcurrent iteration: {}'.format(ctr))
			if self.learning == 'als':
				self.user_vecs = self.als_step(self.user_vecs, 
											   self.item_vecs, 
											   self.ratings, 
											   self.user_fact_reg, 
											   type='user')
				self.item_vecs = self.als_step(self.item_vecs, 
											   self.user_vecs, 
											   self.ratings, 
											   self.item_fact_reg, 
											   type='item')
			elif self.learning == 'sgd':
				self.training_indices = np.arange(self.n_samples)
				np.random.shuffle(self.training_indices)
				self.sgd()
			ctr += 1

	def sgd(self):
		for idx in self.training_indices:
			u = self.sample_row[idx]
			i = self.sample_col[idx]
		prediction = self.predict(u, i)

		e = (self.ratings[u,i] - prediction) # error
		
		# Update biases
		self.user_bias[u] += (self.learning_rate * (e - self.user_bias_reg * self.user_bias[u]))
		self.item_bias[i] += (self.learning_rate * (e - self.item_bias_reg * self.item_bias[i]))
		
		# Update latent factors
		self.user_vecs[u, :] += (self.learning_rate * (e * self.item_vecs[i, :] - 
								 self.user_fact_reg * self.user_vecs[u,:]))
		self.item_vecs[i, :] += (self.learning_rate *
								(e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i,:]))

	def predict_single(self, u, i):
		""" Single user and item prediction."""
		if self.learning == 'als':
			return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
		elif self.learning == 'sgd':
			prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
			prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
			return prediction

	def predict(self):
		""" Predict ratings for every user and item."""
		predictions = np.zeros((self.user_vecs.shape[0], 
								self.item_vecs.shape[0]))
		for u in xrange(self.user_vecs.shape[0]):
			for i in xrange(self.item_vecs.shape[0]):
				predictions[u, i] = self.predict_single(u, i)
		return predictions

		