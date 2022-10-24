<!-- SHIELDS -->
<div align="left">

  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-informational)
  [![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.39.0-6133BD)](https://github.com/Qiskit/qiskit)
  [![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-%E2%89%A5%200.4.4-6133BD)](https://github.com/Qiskit/qiskit-nature)
<br />
  [![License](https://img.shields.io/github/license/qiskit-community/prototype-entanglement-forging?label=License)](LICENSE.txt)
  [![Tests](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml)

# Circuit Knitting Toolbox

<!-- TABLE OF CONTENTS -->
### Table of Contents
* [About](#about)
* [Tutorials](docs/tutorials/)
* [Installation](#installation)
* [References](#references)
* [License](#license)

----------------------------------------------------------------------------------------------------

<!-- ABOUT -->

### About
Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor, then recomposing their results into an estimation of the outcome of the original circuit. Circuit knitting includes techniques such as entanglement forging, circuit cutting, and classical embedding. The Circuit Knitting Toolbox (CKT) is a collection of such tools.

Each tool in the CKT will partition a user's problem into quantum and classical components to optimize efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It will assign the execution of "quantum code" to QPUs or QPU simulators and "classical code" to various heterogeneous classical resources such as CPUs, GPUs, and TPUs made available via hybrid cloud, on-prem, data centers, etc. 

The toolbox will allow users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

The toolbox currently contains the following tools:
- Entanglement Forging [[1]](#references)
- Circuit Cutting [[2]](#references)
  
----------------------------------------------------------------------------------------------------
  
<!-- INSTALLATION -->

### Installation

See [`INSTALL.rst`](INSTALL.rst)

----------------------------------------------------------------------------------------------------

<!-- REFERENCES -->
### References
[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, [Doubling the size of quantum simulators by entanglement forging](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309). PRX Quantum 3, 010309 (2022).

[2] Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, Margaret Martonosi, [CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations](https://doi.org/10.1145/3445814.3446758), Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. pp. 473 (2021).

----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
