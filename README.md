<!-- SHIELDS -->
<div align="left">

  [![Stability](https://img.shields.io/badge/Stability-alpha-f4d03f.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/releases)
  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.43.0-6133BD)](https://github.com/Qiskit/qiskit)
  [![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-%E2%89%A5%200.5.2-6133BD)](https://github.com/Qiskit/qiskit-nature)
<br />
  [![DOI](https://zenodo.org/badge/543181258.svg)](https://zenodo.org/badge/latestdoi/543181258)
  [![License](https://img.shields.io/github/license/Qiskit-Extensions/circuit-knitting-toolbox?label=License)](LICENSE.txt)
  [![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)
  [![Tests](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit-Extensions/circuit-knitting-toolbox/badge.svg?branch=main)](https://coveralls.io/github/Qiskit-Extensions/circuit-knitting-toolbox?branch=main)

# Circuit Knitting Toolbox

### Table of Contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [References](#references)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor(s), and then knitting their results into a reconstruction of the original circuit's outcome. Circuit knitting includes techniques such as entanglement forging, circuit cutting, and classical embedding. The Circuit Knitting Toolbox (CKT) is a collection of such tools.

Each tool in the CKT partitions a user's problem into quantum and classical components to enable efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It can assign the execution of "quantum code" to QPUs or QPU simulators and "classical code" to various heterogeneous classical resources such as CPUs, GPUs, and TPUs made available via hybrid cloud, on-prem, data centers, etc. 

The toolbox enables users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

The toolbox currently contains the following tools:
- Entanglement Forging [[1]](#references)
- Circuit Cutting [[2-6]](#references)
  
----------------------------------------------------------------------------------------------------
  
### Documentation

All CKT documentation is available at https://qiskit-extensions.github.io/circuit-knitting-toolbox/.

----------------------------------------------------------------------------------------------------
  
### Installation

We encourage installing CKT via ``pip``, when possible. Users intending to use the entanglement forging tool should install the ``pyscf`` optional dependency. Users intending to use the automatic cut finding functionality in the ``CutQC`` package should install the ``cplex`` optional dependency.

```bash
pip install 'circuit-knitting-toolbox[cplex,pyscf]'
```

For information on installing from source, running CKT in a container, and platform support, refer to the [installation instructions](https://qiskit-extensions.github.io/circuit-knitting-toolbox/install.html) in the CKT documentation.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

This project is meant to evolve rapidly and, as such, does not follow [Qiskit's deprecation policy](https://qiskit.org/documentation/contributing_to_qiskit.html#deprecation-policy).  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the [release notes](https://qiskit-extensions.github.io/circuit-knitting-toolbox/release-notes.html).

----------------------------------------------------------------------------------------------------

### References

[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, [Doubling the size of quantum simulators by entanglement forging](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309), PRX Quantum 3, 010309 (2022).

[2] Kosuke Mitarai, Keisuke Fujii, [Constructing a virtual two-qubit gate by sampling single-qubit operations](https://iopscience.iop.org/article/10.1088/1367-2630/abd7bc), New J. Phys. 23 023021.

[3] Christophe Piveteau, David Sutter, [Circuit knitting with classical communication](https://arxiv.org/abs/2205.00016), arXiv:2205.00016 [quant-ph].

[4] Lukas Brenner, Christophe Piveteau, David Sutter, [Optimal wire cutting with classical communication](https://arxiv.org/abs/2302.03366), arXiv:2302.03366 [quant-ph].

[5] Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, Margaret Martonosi, [CutQC: Using small quantum computers for large quantum circuit evaluations](https://doi.org/10.1145/3445814.3446758), Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. pp. 473 (2021).
  
[6] K. Temme, S. Bravyi, and J. M. Gambetta, [Error mitigation for short-depth quantum circuits](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509), Physical Review Letters, 119(18), (2017).
  
----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
