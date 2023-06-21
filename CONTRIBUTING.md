# Developer guide

Development of the Circuit Knitting Toolbox takes place [on GitHub](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox).  The [Contributing to Qiskit](https://qiskit.org/documentation/contributing_to_qiskit.html) guide may serve as a useful starting point, as the toolbox builds on [Qiskit] and is part of the [Qiskit Ecosystem].

The toolbox is written in [Python] and uses [tox] as a testing framework.  A description of the available `tox` test environments is located at [`test/README.md`](test/README.md).  These environments are used in the CI workflows, which are described at [`.github/workflows/README.md`](.github/workflows/README.md).

Project configuration, including information about dependencies, is stored in [`pyproject.toml`](pyproject.toml).

We use [Sphinx] for documentation and [reno] for release notes.  We use [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), except we omit the type of each argument, as type information is redundant when Python [type hints](https://docs.python.org/3/library/typing.html) are given.

We require 100% coverage in all new code.  In rare cases where it is not possible to test a code block, we mark it with ``# pragma: no cover`` so that the ``coverage`` tests will pass.

[Qiskit]: https://qiskit.org/
[Qiskit Ecosystem]: https://qiskit.org/ecosystem/
[Python]: https://www.python.org/
[tox]: https://github.com/tox-dev/tox
[Sphinx]: https://www.sphinx-doc.org/
[reno]: https://docs.openstack.org/reno/
