# flake8: noqa
from .base import Base, BaseNMF
from .models import Mean, KNN, NNMF_mult, NNMF_sgd
from .utils import (
    create_sub_by_item_matrix,
    get_size_in_mb,
    get_sparsity,
    nanpdist,
    create_sparse_mask,
    estimate_performance,
    flatten_dataframe,
    unflatten_dataframe,
    split_train_test,
    check_random_state,
)
from .data import load_movielens100k, load_toymat
from .version import __version__
