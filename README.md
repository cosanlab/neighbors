## Status
* [![Build Status](https://travis-ci.org/ljchang/emotionCF.svg?branch=master)](https://travis-ci.org/ljchang/emotionCF)
* [![Coverage Status](https://coveralls.io/repos/github/ljchang/emotionCF/badge.svg?branch=master)](https://coveralls.io/github/ljchang/emotionCF?branch=master)

# emotionCF
A python package to perform collaborative filtering on emotion datasets.

## Installation

```
git clone https://github.com/ljchang/emotionCF.git
cd emotionCF
python setup.py install
```

## Example Usage

### Create a subject by item matrix

The emotionCF toolbox operates on pandas dataframes that contain ratings for each subject by each item.  There is a function to easily convert a long format pandas dataframe that contains ['Subject','Item','Rating] as columns, `create_sub_by_item_matrix()`.  These data can be passed onto any `cf` class.

```python
from emotioncf.data import create_sub_by_item_matrix

ratings = create_sub_by_item_matrix(long_format_df)
```

### Intialize a cf instance
Initialize a new cf instance by passing in the subject by item ratings pandas DataFrame

```
from emotioncf.cf import KNN

cf = KNN(ratings)
```

### Split Data into Train and Test
It is easy to split a ratings matrix into training and test items using the `split_train_test()` method.  It creates a binary mask called `.train_mask` field that indicates the training values.

```
cf.split_train_test(n_train_items=50)
```

### Estimate Model
Each model can be estimated using the `fit()` method

```
cf.fit()
```

### Predict New Ratings
Missing data from the matrix can then be filled in using the `predict()` method.  This creates a pandas dataframe of the predicted subject by item matrix in the `predicted_ratings` field.

```
cf.predict()
```

### Mean
An easy control model for collaborative filtering is to demonstrate how well the models perform over simply using the item means.  We initalize a class instance and then the model can be estimated and new ratings predicted.  We can get the overall mean squared error on the predicted ratings.

```
from emotioncf.cf import Mean

cf = Mean(ratings)
cf.fit()
cf.predict()
cf.get_mse('all')
```

### K-Nearest Neighbors
EmotionCF uses a standard API to estimate and predict data.  Though the KNN approach is not technically a model, we still use the fit method to estimate data.  This calculates a similarity matrix between subjects using ['correlation','cosine'] methods.  We can then predict the left out ratings using the top `k` nearest neighbors.  We can evaluate how well the model works for all data points using `get_corr()` and `get_mse()` methods.  We can also get the correlation for each subject's indivdiual data using `get_sub_corr()` method.
```
from emotioncf.cf import KNN

cf = KNN(ratings)
cf.split_train_test(n_train_items=20)
cf.fit(metric='cosine')
cf.predict(k=10)
cf.get_mse('test')
cf.get_corr('test')
cf.get_sub_corr('test')
```

### Non-negative matrix factorization using stochastic gradient descent

Here we initialize a new class instance and split the data into 20 training and 80 test items per subject.  We fit the model using 100 iterations.  Can pass in optional regularization parameters and a learning rate for the update function.  The model is then used to predict the left out ratings.  We can get the overall model MSE and correlation value on the test ratings.  We can also make a quick plot of the results.

```
from emotioncf.cf import NNMF_sgd

cf = NNMF_sgd(ratings)
cf.split_train_test(n_train_items=20)
cf.fit(n_iterations = 100,
       user_fact_reg=0,
       item_fact_reg=0,
       user_bias_reg=0,
       item_bias_reg=0,
       learning_rate=.001)
cf.predict()
cf.get_mse('test')
cf.get_corr('test')
cf.plot_predictions()
```

### Non-negative matrix factorization using multiplicative updating

Similarly, we can fit a different NNMF model that uses multiplicative updating with the `NNMF_multiplicative` class.

```
from emotioncf.cf import NNMF_multiplicative

cf = NNMF_multiplicative(ratings)
cf.split_train_test(n_train_items=20)
cf.fit(max_iterations=200)
cf.predict()
cf.get_mse('test')
cf.get_corr('test')
cf.plot_predictions()
```

