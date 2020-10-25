# Development

To develop this package or its documentation locally you will need to install a few extra dependencies.

## Installation

`pip install -r requirements-dev.txt`

## Testing

To run tests just call `pytest` from the root of this repository. New tests can be added in `emotioncf/tests/`.

## Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org/) using the [mkdocs material theme](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://pawamoy.github.io/mkdocstrings/) extension. 


### Live server

After installation above, simply run `mkdocs serve` this the project root to start a hot-reloading server of the documentation at `http://localhost:8000`.  

To alter the layout of the docs site adjust settings in `mkdocs.yml`. To add or edit pages simply create markdown files within the `docs/` folder.

### Deploying

You can use the `mkdocs gh-deploy` command in order to build and push the documentation site to the [github-pages branch](https://github.com/cosanlab/emotionCF/tree/gh-pages) of this repo.
