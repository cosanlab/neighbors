# Working with Time-Series Data
A unique feature of this toolbox, is that it has also been designed to work with time-series data.

For example, model instances can be downsampled across items, where items refers to time samples. You must specify the `sampling_freq` of the data, and the `target`, where target must have a `target_type` of `['hz','samples','seconds']`.  Downsampling is performed by averaging over bin windows.  In this example we downsample a dataset from 10Hz to 5Hz.

```python
cf.downsample(sampling_freq=10,target=5, target_type='hz')
```

It is also possible to leverage presumed autocorrelation when training models by using the `dilate_by_nsamples=n_samples` keyword.  This flag will convolve a boxcar kernel of width `n_samples` with each user's rating from `model.train_mask`.  The dilation will be centered on each sample.  The intuition here is that if a subject rates an item at a given time point, say '50', they likely will have rated time points immediately preceding and following similarly (e.g., `[50,50,50]`).  This is due to autocorrelation in the data.  More presumed autocorrelation will likely benefit from a higher number of samples being selected.  This will allow time series that are sparsely sampled to be estimated more accurately.

```python
cf = NNMF_sgd(ratings, n_mask_items=.5)
cf.fit(n_iterations = 100,
     user_fact_reg=1.0,
     item_fact_reg=0.001, 
     user_bias_reg=0,
     item_bias_reg=0,
     learning_rate=.001,
     dilate_ts_n_samples=20)
cf.summary()
```