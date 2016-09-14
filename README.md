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

### Initialize a CF instance

First, we need to create an instance of the CF class. This requires creating a subjects by items matrix.  This can easily be done if the data is a pandas dataframe in the long format and contains ['Subject','Item','Rating] as columns.  Then a CF instance can be created by passing in the number of items to use as training.

```python
from emotioncf.core import CF, create_sub_by_item_matrix

rating = create_sub_by_item_matrix(df)

cf = CF(rating, n_train_items=20)
```
