<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-addon-cutting.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-cutting/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-cutting?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.1%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-addon-cutting/)
  [![DOI](https://zenodo.org/badge/543181258.svg)](https://zenodo.org/badge/latestdoi/543181258)
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-cutting?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-cutting.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-cutting/)
  [![Tests](https://github.com/Qiskit/qiskit-addon-cutting/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-cutting/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-addon-cutting/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-addon-cutting?branch=main)

# Qiskit addon: circuit cutting

### Table of Contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [References](#references)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://docs.quantum.ibm.com/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package implements circuit cutting.  In this technique, a handful of gates and/or wires are cut, resulting in smaller circuits that are better suited for execution on hardware.  The result of the original circuit can then be reconstructed; however, the trade-off is that the overall number of shots must be increased by a factor exponential in the number of cuts.

For a more detailed discussion on circuit cutting, check out our [technical guide](https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html#overview-of-circuit-cutting).

----------------------------------------------------------------------------------------------------
  
### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-cutting/.

----------------------------------------------------------------------------------------------------
  
### Installation

We encourage installing this package via ``pip``, when possible.

```bash
pip install 'qiskit-addon-cutting'
```

For information on installing from source, running in a container, and platform support, refer to the [installation instructions](https://qiskit.github.io/qiskit-addon-cutting/install.html) in the documentation.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in [Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the [release notes](https://qiskit.github.io/qiskit-addon-cutting/release-notes.html).

----------------------------------------------------------------------------------------------------

### References

[1] Kosuke Mitarai, Keisuke Fujii, [Constructing a virtual two-qubit gate by sampling single-qubit operations](https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc), New J. Phys. 23 023021.

[2] Kosuke Mitarai, Keisuke Fujii, [Overhead for simulating a non-local channel with local channels by quasiprobability sampling](https://quantum-journal.org/papers/q-2021-01-28-388/), Quantum 5, 388 (2021).

[3] Christophe Piveteau, David Sutter, [Circuit knitting with classical communication](https://arxiv.org/abs/2205.00016), arXiv:2205.00016 [quant-ph].

[4] Lukas Brenner, Christophe Piveteau, David Sutter, [Optimal wire cutting with classical communication](https://arxiv.org/abs/2302.03366), arXiv:2302.03366 [quant-ph].

[5] K. Temme, S. Bravyi, and J. M. Gambetta, [Error mitigation for short-depth quantum circuits](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509), Physical Review Letters, 119(18), (2017).
  
----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
