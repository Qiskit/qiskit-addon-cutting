# Developer guide

Development of the Circuit Knitting Toolbox takes place [on GitHub](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox).  The [Contributing to Qiskit](https://qiskit.org/documentation/contributing_to_qiskit.html) guide may serve as a useful starting point, as the toolbox builds on Qiskit and is part of the [Qiskit Ecosystem].

The toolbox is written in [Python] and uses `tox` as a testing framework.  A description of the available `tox` test environments is located at [`test/README.md`](test/README.md).  These environments are used in the CI workflows, which are described at [`.github/workflows/README.md`](.github/workflows/README.md).

Project configuration, including information about dependencies, is stored in [`pyproject.toml`](pyproject.toml).

We use [Sphinx] for documentation and [reno] for release notes.

[Qiskit]: https://qiskit.org/
[Qiskit Ecosystem]: https://qiskit.org/ecosystem/
[Python]: https://www.python.org/
[Sphinx]: https://www.sphinx-doc.org/
[reno]: https://docs.openstack.org/reno/
