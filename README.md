# Neighbors
[![Build Status](https://github.com/cosanlab/neighbors/actions/workflows/tests_and_docs.yml/badge.svg)](https://github.com/cosanlab/neighbors/actions/workflows/tests_and_docs.yml)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/neighbors/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/neighbors?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

**A Python package for collaborative filtering on social datasets**

## Installation

1. Pip (official releases): `pip install neighbors`
2. Github (bleeding edge): `pip install git+https://github.com/cosanlab/neighbors.git`

## Getting started

The best way to learn how to use the package is by checking out the [ documentation site](https://cosanlab.github.io/neighbors) which contains usage tutorials as well as API documentation for all package functionality.

### Quick Demo Usage

```python  
from neighbors.models import NNMF_sgd
from neighbors.utils create_user_item_matrix, estimate_performance

# Assuming data is 3 column pandas df with 'User', 'Item', 'Rating'
# convert it to a (possibly sparse) user x item matrix
mat = create_user_item_matrix(df)

# Initialize a model
model = NNMF_sgd(mat)

# Fit
model.fit()

# If data are time-series optionally fit model using dilation
# to leverage auto-correlation and improve performance
model.fit(dilate_by_nsamples=60)

# Visualize results
model.plot_predictions()

# Estimate algorithm performance using
# Repeated refitting with random masking (dense data)
# Or cross-validation (sparse data)
group_results, user_results = estimate_performance(NNMF_sgd, mat)
```


## Algorithms

Currently supported algorithms include:  

- `Mean` - a baseline model
- `KNN` - k-nearest neighbors
- `NNMF_mult` - non-negative matrix factorization trained via multiplicative updating
- `NNMF_sgd` - non-negative matrix factorization trained via stochastic gradient descent
