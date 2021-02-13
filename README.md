# EmotionCF
[![Build Status](https://github.com/cosanlab/emotionCF/workflows/EmotionCF/badge.svg)](https://github.com/cosanlab/emotionCF/actions?query=workflow%3AEmotionCF)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/emotionCF/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/emotionCF?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)

A python package to perform collaborative filtering on emotion datasets.  Compatible with Python 3 only.

## [Documentation Site](https://cosanlab.github.io/emotionCF)

### Installation

```
pip install git+https://github.com/cosanlab/emotionCF.git
```

---

## Development

To develop this package or its documentation locally you will need to install a few extra dependencies.

### Installation

`pip install -r requirements-dev.txt`

### Testing

To run tests just call `pytest` from the root of this repository. New tests can be added in `emotioncf/tests/`.

### Formatting

Please format your code using black. If you've installed the development dependencies, then you can configure `git` to tell if you any new changes are not formatted by setting up a **pre-commit hook:**  

- `cd .git/hooks`
- Create a new file called `pre-commit` with the following contents:

     ```
     #!/bin/sh
     black --check .
     ```
- Make sure the file is executable `chmod 775 pre-commit`

Now anytime you try to commit new changes, git will automatically run black before the commit and warn you if certain files need to be formatted.

### Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org/) using the [mkdocs material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://pawamoy.github.io/mkdocstrings/) extension. 


#### Live server

After installation above, simply run `mkdocs serve` this the project root to start a hot-reloading server of the documentation at `http://localhost:8000`.  

To alter the layout of the docs site adjust settings in `mkdocs.yml`. To add or edit pages simply create markdown files within the `docs/` folder.

#### Deploying

You can use the `mkdocs gh-deploy` command in order to build and push the documentation site to the [github-pages branch](https://github.com/cosanlab/emotionCF/tree/gh-pages) of this repo.
