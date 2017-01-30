from __future__ import division
from scipy import linalg
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy

__all__ = ['Mean',
			'KNN',
			'NNMF_multiplicative',
			'NNMF_sgd']
__author__ = ["Luke Chang"]
__license__ = "MIT"

# Notes might consider making a ratings data class that can accomodate timeseries and tensors

class BaseCF(object):

	''' Base Collaborative Filtering Class '''

	def __init__(self, ratings, mask=None, n_train_items=None):
		if not isinstance(ratings, pd.DataFrame):
			raise ValueError('ratings must be a pandas dataframe instance')			
		self.ratings = ratings
		self.predicted_ratings = None
		self.is_fit = False
		self.is_predict = False
		if mask is not None:
			self.train_mask = mask
			self.is_mask = True
		elif self.ratings.isnull().any().any():
			self.train_mask = ~self.ratings.isnull()
			self.is_mask = True
		else:
			self.is_mask = False

		if n_train_items is not None:
			self.split_train_test(n_train_items=n_train_items)

	def __repr__(self):
		return '%s(rating=%s)' % (
			self.__class__.__name__,
			self.ratings.shape
			)

	def get_mse(self, data='all'):

		''' Get overall mean squared error for predicted compared to actual for all items and subjects. '''
		
		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		''' Get Mean Squared Error ignoring Missing Values '''
		if data is 'all':
			actual = self.ratings.values.flatten()
			pred = self.predicted_ratings.values.flatten()
			return np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2)
		elif self.is_mask:
			if data is 'test':
				return np.mean((self.predicted_ratings.values[~self.train_mask.values]-self.ratings.values[~self.train_mask.values])**2)
			elif data is 'train':
				return np.mean((self.predicted_ratings.values[self.train_mask.values]-self.ratings.values[self.train_mask.values])**2)
		else:
			raise ValueError('Must run split_train_test() before using this option.')

	def get_corr(self, data='all'):

		''' Get overall correlation for predicted compared to actual for all items and subjects. '''

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		''' Get Correlation ignoring Missing Values '''
		if data is 'all':
			actual = self.ratings.values.flatten()
			pred = self.predicted_ratings.values.flatten()
			return pearsonr(pred[(~np.isnan(actual)) & (~np.isnan(pred))], actual[(~np.isnan(actual)) & (~np.isnan(pred))])[0]
		elif self.is_mask:
			if data is 'test':
				return pearsonr(self.predicted_ratings.values[~self.train_mask.values], self.ratings.values[~self.train_mask.values])[0]
			if data is 'train':
				return pearsonr(self.predicted_ratings.values[self.train_mask.values], self.ratings.values[self.train_mask.values])[0]
		else:
			raise ValueError('Must run split_train_test() before using this option.')

	def get_sub_corr(self, data='all'):

		'''Calculate observed/predicted correlation for each subject in matrix'''

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		r = []
		if data is 'all':
			for i in self.ratings.index:
				r.append(pearsonr(self.ratings.loc[i,:], self.predicted_ratings.loc[i,:])[0])
		elif self.is_mask:
			if data is 'test':
				for i in self.ratings.index:
					r.append(pearsonr(self.ratings.loc[i, ~self.train_mask.loc[i,:]], self.predicted_ratings.loc[i,~self.train_mask.loc[i,:]])[0])
			if data is 'train':
				for i in self.ratings.index:
					r.append(pearsonr(self.ratings.loc[i, self.train_mask.loc[i,:]], self.predicted_ratings.loc[i,self.train_mask.loc[i,:]])[0])
		else:
			raise ValueError('Must run split_train_test() before using this option.')
		return np.array(r)

	def get_sub_mse(self, data='all'):

		'''Calculate observed/predicted mse for each subject in matrix'''

		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')

		mse = []
		if data is 'all':
			for i in self.ratings.index:
				actual = self.ratings.loc[i,:]
				pred = self.predicted_ratings.loc[i,:]
				mse.append(np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2))
		if self.is_mask:
			if data is 'test':
				for i in self.ratings.index:
					actual = self.ratings.loc[i,~self.train_mask.loc[i,:]]
					pred = self.predicted_ratings.loc[i,~self.train_mask.loc[i,:]]
					mse.append(np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2))
			if data is 'train':
				for i in self.ratings.index:
					actual = self.ratings.loc[i,self.train_mask.loc[i,:]]
					pred = self.predicted_ratings.loc[i,self.train_mask.loc[i,:]]
					mse.append(np.mean((pred[(~np.isnan(actual)) & (~np.isnan(pred))] - actual[(~np.isnan(actual)) & (~np.isnan(pred))])**2))
		else:
			raise ValueError('Must run split_train_test() before using this option.')
		return np.array(mse)

	def split_train_test(self, n_train_items=20):
		''' Split ratings matrix into train and test items.  mask indicating training items

		Args:
			n_train_items: (int) number of items for test dictionary or list of specific items

		'''
		
		self.n_train_items = int(n_train_items)
		self.train_mask = self.ratings.copy()
		self.train_mask.loc[:,:] = np.zeros(self. ratings.shape).astype(bool)

		for sub in self.ratings.index:
			sub_train_rating_item =  np.random.choice(self.ratings.columns,replace=False, size=n_train_items)
			self.train_mask.loc[sub, sub_train_rating_item] = True
		self.is_mask = True

	def plot_predictions(self):

		''' Create plot of actual and predicted ratings'''

		import matplotlib.pyplot as plt
		import seaborn as sns		
		if not self.is_fit:
			raise ValueError('You must fit() model first before using this method.')
		if not self.is_predict:
			raise ValueError('You must predict() model first before using this method.')
		
		if self.is_mask:
			f, ax = plt.subplots(nrows=1,ncols=3, figsize=(15,8))
		else:
			f, ax = plt.subplots(nrows=1,ncols=2, figsize=(15,8))

		sns.heatmap(self.ratings,vmax=100,vmin=0,ax=ax[0],square=False,xticklabels=False, yticklabels=False)
		ax[0].set_title('Actual User/Item Ratings')
		ax[0].set_xlabel('Items')
		ax[0].set_ylabel('Users')
		sns.heatmap(self.predicted_ratings,vmax=100,vmin=0,ax=ax[1],square=False,xticklabels=False, yticklabels=False)
		ax[1].set_title('Predicted User/Item Ratings')
		ax[1].set_xlabel('Items')
		ax[1].set_ylabel('Users')

		f.tight_layout()

		if self.is_mask:
			actual = self.ratings.values.flatten()
			pred = self.predicted_ratings.values.flatten()
			mask = self.train_mask.values.flatten()
			pred = pred[~mask]
			actual = actual[~mask]
			ax[2].scatter(actual[(~np.isnan(actual)) & (~np.isnan(pred))],pred[(~np.isnan(actual)) & (~np.isnan(pred))])
			ax[2].set_xlabel('Actual Ratings')
			ax[2].set_ylabel('Predicted Ratings')
			ax[2].set_title('Predicted Ratings')
			r = pearsonr(actual[(~np.isnan(actual)) & (~np.isnan(pred))],pred[(~np.isnan(actual)) & (~np.isnan(pred))])[0]
			print('Correlation: %s' % r)
			return f, r
		else:
			return f

	def downsample(self, sampling_freq=None, target=None, target_type='samples'):

		''' Downsample rating matrix to a new target frequency or number of samples using averaging.
		
			Args:
				sampling_freq:  Sampling frequency of data 
				target: downsampling target
				target_type: type of target can be [samples,seconds,hz]
				
		'''

		if sampling_freq is None:
			raise ValueError('Please specify the sampling frequency of the ratings data.')
		if target is None:
			raise ValueError('Please specify the downsampling target.')
		if target_type is None:
			raise ValueError('Please specify the type of target to downsample to [samples,seconds,hz].')


		def ds(ratings, sampling_freq=sampling_freq, target=None, target_type='samples'):
			if target_type is 'samples':
				n_samples = target
			elif target_type is 'seconds':
				n_samples = target*sampling_freq
			elif target_type is 'hz':
				n_samples = sampling_freq/target
			else:
				raise ValueError('Make sure target_type is "samples", "seconds", or "hz".')

			ratings = ratings.T
			idx = np.sort(np.repeat(np.arange(1,data.shape[0]/n_samples,1),n_samples))
			if ratings.shape[0] > len(idx):
				idx = np.concatenate([idx, np.repeat(idx[-1]+1,data.shape[0]-len(idx))])
			return ratings.groupby(idx).mean().T

		self.ratings = ds(self.ratings, sampling_freq=sampling_freq, target=target, 
			target_type=target_type)

		if self.is_mask:
			self.train_mask = ds(self.train_mask, sampling_freq=sampling_freq, 
				target=target, target_type=target_type)
			self.train_mask.loc[:,:] = self.train_mask>0

		if self.is_predict:
			self.predicted_ratings = ds(self.predicted_ratings, 
				sampling_freq=sampling_freq, target=target, target_type=target_type)

	def to_long_df(self):
	
		''' Create a long format pandas dataframe with observed, predicted, and mask.'''

		observed = pd.DataFrame(columns=['Subject','Item','Rating','Condition'])
		for row in self.ratings.iterrows():
			tmp = pd.DataFrame(columns=observed.columns)
			tmp['Rating'] = row[1]
			tmp['Item'] = self.ratings.columns
			tmp['Subject'] = row[0]
			tmp['Condition'] = 'Observed'
			if self.is_mask:
				tmp['Mask'] = self.train_mask.loc[row[0]]
			observed = observed.append(tmp)

		if self.is_predict:
			predicted = pd.DataFrame(columns=['Subject','Item','Rating','Condition'])
			for row in self.predicted_ratings.iterrows():
				tmp = pd.DataFrame(columns=predicted.columns)
				tmp['Rating'] = row[1]
				tmp['Item'] = self.predicted_ratings.columns
				tmp['Subject'] = row[0]
				tmp['Condition'] = 'Predicted'
				if self.is_mask:
					tmp['Mask'] = self.train_mask.loc[row[0]]
				predicted = predicted.append(tmp)
			observed = observed.append(predicted)
		return observed

	def _conv_ts_mean_overlap(self, sub_rating, n_samples=5):

		'''Dilate each rating by n samples (centered).  If dilated samples are overlapping they will be averaged.
		
			Args:
				sub_rating: vector of ratings for subject
				n_samples:  number of samples to dilate each rating

			Returns:
				sub_rating_conv_mn: subject rating vector with each rating dilated n_samples (centered) with mean of overlapping

		'''

		# Notes:  Could add custom filter input
		if np.any(sub_rating.isnull()):
			sub_rating.fillna(0, inplace=True)
		bin_sub_rating = sub_rating>0
		filt = np.ones(n_samples)
		bin_sub_rating_conv = np.convolve(bin_sub_rating, filt, mode='same')
		sub_rating_conv = np.convolve(sub_rating, filt, mode='same')
		sub_rating_conv_mn = deepcopy(sub_rating_conv)
		sub_rating_conv_mn[bin_sub_rating_conv>1] = (sub_rating_conv_mn[bin_sub_rating_conv>1]/
			bin_sub_rating_conv[bin_sub_rating_conv>1])
		return sub_rating_conv_mn

	def _dilate_ts_rating_samples(self, n_samples=None):

		''' Helper function to dilate sparse time-series ratings by n_samples.  
			Overlapping ratings will be averaged

			Args:
				n_samples:  Number of samples to dilate ratings

			Returns:
				masked_ratings: pandas ratings instance that has been dilated by n_samples
		'''
		
		if n_samples is None:
			raise ValueError('Please specify number of samples to dilate.')
		
		if not self.is_mask:
			raise ValueError('Make sure cf instance has been masked.')

		masked_ratings = self.ratings[self.train_mask]
		return masked_ratings.apply(lambda x: self._conv_ts_mean_overlap(x, n_samples=n_samples), axis=1)

class Mean(BaseCF):

	''' CF using Item Mean across subjects'''

	def __init__(self, ratings, mask=None, n_train_items=None):
		super(Mean, self).__init__(ratings, mask, n_train_items)
		self.mean = None

	def fit(self, dilate_ts_n_samples=None):

		''' Fit collaborative model to training data.  Calculate similarity between subjects across items

		Args:
			metric: type of similarity {"correlation","cosine"}
			dilate_ts_n_samples: will dilate masked samples by n_samples to leverage auto-correlation 
								in estimating time-series ratings

		'''

		if self.is_mask:			
			if dilate_ts_n_samples is None:
				self.mean = self.ratings[self.train_mask].mean(skipna=True, axis=0)
			else:
				self.mean = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples).mean(skipna=True, axis=0)
		else:
			self.mean = self.ratings.mean(skipna=True, axis=0)
		self.is_fit = True

	def predict(self):

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

	def __init__(self, ratings, mask=None, n_train_items=None):
		super(KNN, self).__init__(ratings, mask, n_train_items)
		self.subject_similarity = None

	def fit(self, metric='pearson', dilate_ts_n_samples=None):

		''' Fit collaborative model to training data.  Calculate similarity between subjects across items

		Args:
			metric: type of similarity {"pearson",,"spearman","correlation","cosine"}.  Note pearson and spearman are way faster.
			dilate_ts_n_samples: will dilate masked samples by n_samples to leverage auto-correlation 
								in estimating time-series ratings

		'''

		if self.is_mask:			
			if dilate_ts_n_samples is None:
				ratings = self.ratings[self.train_mask]
			else:
				ratings = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)
		else:
			ratings = deepcopy(self.ratings)

		def cosine_similarity(x,y):
			return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

		if metric in ['pearson','kendall','spearman']:
			sim = self.ratings.T.corr(method=metric)
		elif metric in ['correlation','cosine']:
			sim = pd.DataFrame(np.zeros((ratings.shape[0],ratings.shape[0])))
			sim.columns=ratings.index
			sim.index=ratings.index
			for x in ratings.iterrows():
				for y in ratings.iterrows():
					if metric is 'correlation':
						sim.loc[x[0],y[0]] = pearsonr(x[1][(~x[1].isnull()) & (~y[1].isnull())],y[1][(~x[1].isnull()) & (~y[1].isnull())])[0] 
					elif metric is 'cosine':
						sim.loc[x[0],y[0]] = cosine_similarity(x[1][(~x[1].isnull()) & (~y[1].isnull())],y[1][(~x[1].isnull()) & (~y[1].isnull())]) 
		else:
			raise NotImplementedError("%s is not implemented yet. Try ['pearson','spearman','correlation','cosine']" % metric )
		self.subject_similarity = sim
		self.is_fit = True

	def predict(self, k=None):

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
			top_subjects = top_subjects[~top_subjects.isnull()] # remove nan subjects
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
	
	def __init__(self, ratings, mask=None, n_train_items=None):
		super(NNMF_multiplicative, self).__init__(ratings, mask, n_train_items)
		self.H = None
		self.W = None
	
	def fit(self,
		n_factors = None, 
		max_iterations = 100,
		error_limit = 1e-6, 
		fit_error_limit = 1e-6, 
		verbose = False,
		dilate_ts_n_samples = None):

		''' Fit NNMF collaborative filtering model to training data using multiplicative updating.

		Args:
			n_factors (int): Number of factors or components
			max_iterations (int):  maximum number of interations (default=100)
			error_limit (float): error tolerance (default=1e-6)
			fit_error_limit (float): fit error tolerance (default=1e-6)
			verbose (bool): verbose output during fitting procedure (default=True)
			dilate_ts_n_samples (int): will dilate masked samples by n_samples to leverage auto-correlation 
										in estimating time-series ratings

		'''

		eps = 1e-5

		n_users, n_items = self.ratings.shape
		if n_factors is None:
			n_factors = n_items

		# Initial guesses for solving X ~= WH. H is random [0,1] scaled by sqrt(X.mean() / n_factors)
		avg = np.sqrt(np.nanmean(self.ratings)/n_factors)
		self.H = avg*np.random.rand(n_items, n_factors) # H = Y
		self.W = avg*np.random.rand(n_users, n_factors)	# W = A
		
		if self.is_mask:
			if dilate_ts_n_samples is None:
				masked_X = self.ratings.values * self.train_mask.values
				mask = self.train_mask.values
			else:
				masked_X = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples).values
				mask = masked_X>0
		else:
			masked_X = self.ratings.values
			mask = np.ones(self.ratings.shape)


		X_est_prev = np.dot(self.W, self.H)

		ctr = 1
		while ctr <= max_iterations or fit_residual < fit_error_limit:
		# while ctr <= max_iterations or curRes < error_limit or fit_residual < fit_error_limit:
			# Update W: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
			self.W *= np.dot(masked_X, self.H.T) / np.dot(mask * np.dot(self.W, self.H), self.H.T)
			self.W = np.maximum(self.W, eps)

			# Update H: Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
			self.H *= np.dot(self.W.T, masked_X) / np.dot(self.W.T, mask * np.dot(self.W, self.H))
			self.H = np.maximum(self.H, eps)

			# Evaluate
			X_est = np.dot(self.W, self.H)
			err = mask * (X_est_prev - X_est)
			fit_residual = np.sqrt(np.sum(err ** 2))
			X_est_prev = X_est
			# curRes = linalg.norm(mask * (masked_X - X_est), ord='fro')
			if ctr % 10 == 0 and verbose:
				print('\tCurrent Iteration {}:'.format(ctr))
				print('\tfit residual', np.round(fit_residual, 4))
				# print('\ttotal residual', np.round(curRes, 4))
			ctr += 1
		self.is_fit = True

	def predict(self):

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
	
	def __init__(self, ratings, mask=None, n_train_items=None):
		super(NNMF_sgd, self).__init__(ratings, mask, n_train_items)

	def fit(self, 
		n_factors=None, 
		item_fact_reg=0.0, 
		user_fact_reg=0.0,
		item_bias_reg=0.0,
		user_bias_reg=0.0,
		learning_rate=0.001,
		n_iterations=10,
		verbose=False,
		dilate_ts_n_samples=None):

		''' Fit NNMF collaborative filtering model to training data using stochastic gradient descent.

		Args:
			n_factors (int): Number of factors or components
			max_iterations (int):  maximum number of interations (default=100)
			error_limit (float): error tolerance (default=1e-6)
			fit_error_limit (float): fit error tolerance (default=1e-6)
			verbose (bool): verbose output during fitting procedure (default=True)
			dilate_ts_n_samples (int): will dilate masked samples by n_samples to leverage auto-correlation 
										in estimating time-series ratings

		'''

		# initialize variables
		n_users, n_items = self.ratings.shape
		if n_factors is  None:
			n_factors = n_items

		if self.is_mask:
			if dilate_ts_n_samples is None:
				sample_row, sample_col = self.train_mask.values.nonzero()
				self.global_bias = self.ratings[self.train_mask].mean().mean()
				ratings = self.ratings[self.train_mask]
			else:
				ratings = self._dilate_ts_rating_samples(n_samples=dilate_ts_n_samples)
				sample_row, sample_col = ratings.values.nonzero()
				mask = ratings>0
				self.global_bias = ratings[mask].mean().mean()
		else:
			ratings = self.ratings
			sample_row, sample_col = ratings.values.nonzero()
			self.global_bias = self.ratings[~self.ratings.isnull()].mean().mean()

		# initialize latent vectors		
		self.user_vecs = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
		self.item_vecs = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))

		# Initialize biases
		self.user_bias = np.zeros(n_users)
		self.item_bias = np.zeros(n_items)
		self.item_fact_reg = item_fact_reg
		self.user_fact_reg = user_fact_reg
		self.item_bias_reg = item_bias_reg
		self.user_bias_reg = user_bias_reg

		# train weights
		ctr = 1
		while ctr <= n_iterations:
			if ctr % 10 == 0 and verbose:
				print('\tCurrent Iteration: {}'.format(ctr))

			training_indices = np.arange(len(sample_row))
			np.random.shuffle(training_indices)

			for idx in training_indices:
				u = sample_row[idx]
				i = sample_col[idx]
				prediction = self._predict_single(u,i)

				e = (ratings.iloc[u,i] - prediction) # error
				
				# Update biases
				self.user_bias[u] += (learning_rate * (e - self.user_bias_reg * self.user_bias[u]))
				self.item_bias[i] += (learning_rate * (e - self.item_bias_reg * self.item_bias[i]))
				
				# Update latent factors
				self.user_vecs[u, :] += (learning_rate * (e * self.item_vecs[i, :] - self.user_fact_reg * self.user_vecs[u,:]))
				self.item_vecs[i, :] += (learning_rate * (e * self.user_vecs[u, :] - self.item_fact_reg * self.item_vecs[i,:]))
			ctr += 1
		self.is_fit = True

	def predict(self):

		''' Predict Subject's missing items using NNMF with stochastic gradient descent

			Args:
				ratings: pandas dataframe instance of ratings
				k: number of closest neighbors to use
			Returns:
				predicted_rating: (pd.DataFrame instance) adds field to object instance
		'''
		self.predicted_ratings = self.ratings.copy()
		# self.predicted_ratings = np.zeros((self.user_vecs.shape[0], self.item_vecs.shape[0]))
		for u in range(self.user_vecs.shape[0]):
			for i in range(self.item_vecs.shape[0]):
				self.predicted_ratings.iloc[u, i] = self._predict_single(u, i)
		self.is_predict = True

	def _predict_single(self, u, i):
			""" Single user and item prediction."""
			prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
			prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
			return prediction


