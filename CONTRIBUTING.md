# Developer guide

Development of the `qiskit-addon-cutting` package takes place [on GitHub](https://github.com/Qiskit/qiskit-addon-cutting). The [Contributing to Qiskit](https://github.com/Qiskit/qiskit/blob/main/CONTRIBUTING.md) guide may serve as a useful starting point, as this package builds on [Qiskit].

This package is written in [Python] and uses [tox] as a testing framework.  A description of the available `tox` test environments is located at [`test/README.md`](test/README.md).  These environments are used in the CI workflows, which are described at [`.github/workflows/README.md`](.github/workflows/README.md).

Project configuration, including information about dependencies, is stored in [`pyproject.toml`](pyproject.toml).

We use [Sphinx] for documentation and [reno] for release notes.  We use [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), except we omit the type of each argument, as type information is redundant with Python [type hints](https://docs.python.org/3/library/typing.html).

We require 100% coverage in all new code.  In rare cases where it is not possible to test a code block, we mark it with ``# pragma: no cover`` so that the ``coverage`` tests will pass.

[Qiskit]: https://www.ibm.com/quantum/qiskit
[Python]: https://www.python.org/
[tox]: https://github.com/tox-dev/tox
[Sphinx]: https://www.sphinx-doc.org/
[reno]: https://docs.openstack.org/reno/
