# Contributing to development

Each new push or pull-request to the code base for this toolbox will automatically be run through testing and documentation building via github actions.

Development uses [uv](https://docs.astral.sh/uv/) to manage the project environment, dependencies, and builds. All project metadata, dependencies, and tool configuration live in `pyproject.toml`.

## Installation

`uv sync --group dev` creates a virtual environment at `.venv/` with the package installed in editable mode along with all development dependencies. Add `--group docs` if you also want to build the documentation locally.

## Testing

To run tests just call `uv run pytest` from the root of this repository. New tests can be added in `neighbors/tests/`. Tests can be parallelized with `uv run pytest -rs -n auto`.

## Linting and formatting

Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/). Before committing, run:

```
uv run ruff check .
uv run ruff format .
```

You can configure `git` to warn you about unlinted or unformatted changes by setting up a **pre-commit hook:**

- `cd .git/hooks`
- Create a new file called `pre-commit` with the following contents:

     ```
     #!/bin/sh
     uv run ruff check . && uv run ruff format --check .
     ```
- Make sure the file is executable `chmod 775 pre-commit`

Now anytime you try to commit new changes, git will automatically run ruff before the commit and warn you if certain files need attention.


## Editing continuous integration

To change how the automatic workflow builds are specified, make the relevant edits in the files within `.github/workflows/`.

## Documentation

Documentation is built with [mkdocs](https://www.mkdocs.org/) using the [mkdocs material theme](https://squidfunk.github.io/mkdocs-material/), [mkdocstrings](https://pawamoy.github.io/mkdocstrings/), and [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) plugins.


### Live server

After installation above, simply run `uv run mkdocs serve` from the project root to start a hot-reloading server of the documentation at `http://localhost:8000`.

To alter the layout of the docs site adjust settings in `mkdocs.yml`. To add or edit pages simply create markdown files within the `docs/` folder.

### Deploying

You can use the `uv run mkdocs gh-deploy` command in order to build and push the documentation site to the [github-pages branch](https://github.com/cosanlab/neighbors/tree/gh-pages) of this repo.
