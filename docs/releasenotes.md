# Release Notes

## 0.2.1
- Fix `NNMF_mult` treating missing (held-out) entries as observed zeros during multiplicative updating. Only the numerators of the update were masked; the denominators used the full reconstruction, so the model was also fitting the zeros in place of missing values and dragging held-out predictions toward zero. Both denominators now use the masked reconstruction following [Zhu (2016)](https://arxiv.org/pdf/1612.06037.pdf), and the training error is computed over observed entries only. **Note:** `NNMF_mult` results will differ (substantially more accurate on held-out data) from previous versions
- Fix `NNMF_mult` and `NNMF_sgd` prediction clipping using the *dilated* training data to compute the rating range when fit with `dilate_by_nsamples`. Dilation averages neighbouring ratings, which shrinks the range and truncated legitimate predictions near the ends of the rating scale. Clip bounds now always come from the raw observed ratings

## 0.2.0
- `NNMF_mult` and `NNMF_sgd` now clip predictions to the observed rating range by default (disable with `fit(clip_predictions=False)`), preventing out-of-range predictions such as negative values caused by unconstrained bias terms ([#47](https://github.com/cosanlab/neighbors/issues/47)). This is the same approach the [Surprise](https://surpriselib.com/) package takes when making predictions
- Fix `estimate_performance` failing with `KeyError: 'user'` when the input dataframe's index was not named exactly "User" ([#38](https://github.com/cosanlab/neighbors/issues/38))
- Center temporal dilation kernels on each observed sample and average (rather than sum) overlapping dilations ([#41](https://github.com/cosanlab/neighbors/issues/41)). **Note:** models fit with `dilate_by_nsamples` will produce numerically different (more accurate) results than previous versions
- Detect and halt SGD training when predictions diverge to NaN, exposed via a new `.error_is_nan` model attribute ([#42](https://github.com/cosanlab/neighbors/issues/42))
- Fix splitting/combining datasets with mixed or non-string column and index names ([#34](https://github.com/cosanlab/neighbors/issues/34), [#36](https://github.com/cosanlab/neighbors/issues/36))
- Support modern numpy (>=1.26) and pandas (>=2.1, including 3.x)
- **Drop support for Python < 3.11**; tested on Python 3.11-3.14
- Modernized tooling: `uv` + `pyproject.toml` for packaging and environments (replacing `setup.py` and requirements files) and `ruff` for linting/formatting (replacing `black` and `pycodestyle`)

## 0.1.0
- **Official pypi public release**
- Package rename

## 0.0.4
- standardize codebase with `black`
- **complete API rewrite**
- new `estimate_performance` function
- all new tests with `pytest` fixtures
- new docs site with `mkdocs`

## 0.0.3
- Fix dilation and convolution issues
- Update tests
- Drop support for Python 2

## 0.0.2
- Fixed pandas `.apply` bug
- Faster `create_sub_by_item_matrix`

## 0.0.1
- Initial internal release