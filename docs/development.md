# Contributing to development

Each new push or pull-request to the code base for this toolbox will automatically be run through testing and documentation building via github actions.

To develop this package or its documentation locally you will need to install a few extra dependencies.

## Installation

`pip install -r requirements-dev.txt`

## Testing

To run tests just call `pytest` from the root of this repository. New tests can be added in `neighbors/tests/`. To speed up tests you can optionally `pip install pytest-xdist` to parallelize testing and use it with: `pytest -rs -n auto`.

## Formatting

Please format your code using black. If you've installed the development dependencies, then you can configure `git` to tell if you any new changes are not formatted by setting up a **pre-commit hook:**  

- `cd .git/hooks`
- Create a new file called `pre-commit` with the following contents:

     ```
     #!/bin/sh
     black --check .
     ```
- Make sure the file is executable `chmod 775 pre-commit`

Now anytime you try to commit new changes, git will automatically run black before the commit and warn you if certain files need to be formatted.


## Editing continuous integration

To change how the automatic workflow builds are specified, make the relevant edits in `.github/workflows/conda_ci.yml`.

## Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org/) using the [mkdocs material theme](https://squidfunk.github.io/mkdocs-material/), [mkdocstrings](https://pawamoy.github.io/mkdocstrings/), and [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) plugins. 


### Live server

After installation above, simply run `mkdocs serve` this the project root to start a hot-reloading server of the documentation at `http://localhost:8000`.  

To alter the layout of the docs site adjust settings in `mkdocs.yml`. To add or edit pages simply create markdown files within the `docs/` folder.

### Deploying

You can use the `mkdocs gh-deploy` command in order to build and push the documentation site to the [github-pages branch](https://github.com/cosanlab/neighbors/tree/gh-pages) of this repo.

