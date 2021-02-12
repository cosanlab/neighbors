# Emotion CF
![Build Status](https://github.com/cosanlab/emotionCF/workflows/EmotionCF/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/emotionCF/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/emotionCF?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

## **A Python package for collaborative filtering on emotion datasets**

## Installation

```bash
pip install git+https://github.com/cosanlab/emotionCF.git
```  

## Getting started

Checkout the [quick overview](overview.md) for examples to help you get started.  

Or check out the API reference on the left to explore the details of specific models.

A unique feature of this toolbox is its support for [working with time-series data](overview/#working-with-time-series-data).

## Algorithms

Currently supported algorithms include:  

- `Mean` - a baseline model
- `KNN` - k-nearest neighbors
- `NNMF_mult` - non-negative matrix factorization trained via multiplicative updating
- `NNMF_sgd` - non-negative matrix factorization trained via stochastic gradient descent
