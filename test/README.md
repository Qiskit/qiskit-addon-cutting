# Test environments

This repository's tests and development automation tasks are organized using [tox], a command-line CI frontend for Python projects.  tox is typically used during local development and is also invoked from this repository's GitHub Actions [workflows](../.github/workflows/).

tox can be installed by running `pip install tox`.

tox is organized around various "environments," each of which is described below.  To run _all_ test environments, run `tox` without any arguments:

```sh
$ tox
```

Environments for this repository are configured in [`tox.ini`] as described below.

## Lint environment

The `lint` environment ensures that the code meets basic coding standards, including

- [_Black_] formatting style
- Style checking with [ruff], [autoflake], and [pydocstyle]
- [mypy] type annotation checker, as configured by [`.mypy.ini`]

The _Black_ and mypy passes are applied also to [Jupyter] notebooks (via [nbqa]).

To run:

```sh
$ tox -e lint
```

## Style environment

The command `tox -e style` will apply automated style fixes.  This includes:

- Automated fixes from [ruff] and [autoflake]
- Reformatting of all files in the repository according to _Black_ style

## Test (py##) environments

The `py##` environments are the main test environments.  tox defines one for each version of Python.  For instance, the following command will run the tests on Python 3.8, Python 3.9, and Python 3.10:

```sh
$ tox -e py38,py39,py310
```

These environments execute all tests using [pytest], which supports its own simple style of tests, in addition to [unittest]-style tests and [doctests] located throughout the project's docstrings.

## Notebook environments

The `notebook` and `py##-notebook` environments invoke [nbmake] to ensure that all Jupyter notebooks in the [`docs/`](/docs/) directory execute successfully.

## Coverage environment

The `coverage` environment uses [Coverage.py] to ensure that the fraction of code tested by pytest is above some threshold (currently set to 80% overall, and 100% in all new modules).  A detailed, line-by-line coverage report can be viewed by navigating to `htmlcov/index.html` in a web browser.

To run:

```sh
$ tox -e coverage
```

## Documentation environment

The `docs` environment builds the [Sphinx] documentation locally.

For the documentation build to succeed, [pandoc](https://pandoc.org/) must be installed.  Pandoc is not available via pip, so must be installed through some other means.  Linux users are encouraged to install it through their package manager (e.g., `sudo apt-get install -y pandoc`), while macOS users are encouraged to install it via [Homebrew](https://brew.sh/) (`brew install pandoc`).  Full instructions are available on [pandoc's installation page](https://pandoc.org/installing.html).

To run this environment:

```sh
$ tox -e docs
```

If the build succeeds, it can be viewed by navigating to `docs/_build/html/index.html` in a web browser.

[tox]: https://github.com/tox-dev/tox
[`tox.ini`]: ../tox.ini
[mypy]: https://mypy.readthedocs.io/en/stable/
[`.mypy.ini`]: ../.mypy.ini
[nbmake]: https://github.com/treebeardtech/nbmake
[_Black_]: https://github.com/psf/black
[ruff]: https://github.com/charliermarsh/ruff
[autoflake]: https://github.com/PyCQA/autoflake
[pydocstyle]: https://www.pydocstyle.org/en/stable/
[pylint]: https://github.com/PyCQA/pylint
[nbqa]: https://github.com/nbQA-dev/nbQA
[Jupyter]: https://jupyter.org/
[doctests]: https://docs.python.org/3/library/doctest.html
[pytest]: https://docs.pytest.org/
[unittest]: https://docs.python.org/3/library/unittest.html
[Coverage.py]: https://coverage.readthedocs.io/
[Sphinx]: https://www.sphinx-doc.org/
