<!-- SHIELDS -->
<div align="left">

  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-informational)
  [![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.37.0-6133BD)](https://github.com/Qiskit/qiskit)
  [![Qiskit Nature](https://img.shields.io/badge/Qiskit%20Nature-%E2%89%A5%200.4.2-6133BD)](https://github.com/Qiskit/qiskit-nature)
<br />
  [![License](https://img.shields.io/github/license/qiskit-community/prototype-entanglement-forging?label=License)](LICENSE.txt)
  [![Tests](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/actions/workflows/test_latest_versions.yml)

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
Circuit Knitting is the process of decomposing a large circuit into smaller circuits, executing those smaller circuits on a quantum processor, then recomposing those results to simulate the results of the large circuit. The Circuit Knitting Toolbox (CKT) is a collection of Circuit Knitting tools such as Entanglement Forging, Circuit Cutting, and Classical Embedding. 

Each tool in the CKT will partition a user's problem into quantum and classical components to optimize efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It will assign the execution of "quantum code" to QPUs or QPU simulators and "classical code" to various heterogeneous classical resources such as CPUs, GPUs, and TPUs made available via hybrid cloud, on-prem, data centers, etc. 

The toolbox will allow users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

*Status of current tools:*
- *Entanglement Forging: ready for internal testing*
- *Circuit Cutting (wire-cutting aka CutQC): ready for internal testing*
- *Other Circuit Cutting techniques: TBD*
- *Classical Embedding: TBD*
  
----------------------------------------------------------------------------------------------------
  
<!-- INSTALLATION -->

### Installation

There are two options: installing locally or using within a Docker container.  If you are using Linux, everything should work natively, so we recommend the first option.  If you are using macOS with a version of Python earlier than 3.10 and an Intel chip (i.e., not the new M1 or M2 chips), everything should work natively as well.  All other Mac users, as well as all Windows users, will have to use the toolbox within Docker (the second option) for everything to work as designed.

### Option 1: Local installation

* **OPTIONAL** [Install Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) if you wish to obtain the input files necessary to run the [demo notebook](docs/entanglement_forging/demos/entanglement_forging_quantum_serverless_demo.ipynb).

* **OPTIONAL** If a user wishes to use the circuit cutting tool to automatically find optimized cut points for a circuits too large for the free version of CPLEX, they should acquire a license and install the [full version](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjuq9OM1M75AhVoFNQBHWqGBW4YABAAGgJvYQ&ohost=www.google.com&cid=CAESauD2CglQCoRYTsgQCH50ip7Y_PCiHfnYyojivn_Od4YBaoXY74TyZYrKZNZuL0H9je0pzRNWut7uutUNmRc2x-P0nuTbQLAaC2p2fI3PTD87BbRBI07uzMo0ZTSmkyWQiGb9C3Hkv1bbawk&sig=AOD64_0oLk3SUhEbH-EQ35AWeP5_94a45A&q&adurl&ved=2ahUKEwiA1MmM1M75AhXXrmoFHdAcCVQQ0Qx6BAgEEAE&nis=2).
  
* Enter a Python environment and install the software

```sh
$ git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
$ cd circuit-knitting-toolbox
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install tox notebook -e '.[notebook-dependencies]'
$ jupyter notebook
```

* Navigate to the notebooks in the `docs/` directory to run the tutorials.

#### Option 2: Use within Docker

We have provided a [`Dockerfile`](Dockerfile), which can be used to build a Docker image, as well as a [`docker-compose.yml`](docker-compose.yml) file, which allows one to use the Docker image with just a few simple commands.

```sh
$ git clone git@github.com:Qiskit-Extensions/circuit-knitting-toolbox.git
$ cd circuit-knitting-toolbox
$ docker-compose build
$ docker-compose up
```

Depending on your system configuration, you may need to type `sudo` before each `docker-compose` command.

Once the container is running, you should see a message like this:

```
notebook_1  |     To access the server, open this file in a browser:
notebook_1  |         file:///home/jovyan/.local/share/jupyter/runtime/jpserver-7-open.html
notebook_1  |     Or copy and paste one of these URLs:
notebook_1  |         http://e4a04564eb39:8888/lab?token=00ed70b5342f79f0a970ee9821c271eeffaf760a7dcd36ec
notebook_1  |      or http://127.0.0.1:8888/lab?token=00ed70b5342f79f0a970ee9821c271eeffaf760a7dcd36ec
```

Locate the _last_ URL in your terminal (the one that includes `127.0.0.1`), and navigate to that URL in a web browser to access the Jupyter notebook interface.

----------------------------------------------------------------------------------------------------

<!-- REFERENCES -->
### References
[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, *Doubling the size of quantum simulators by entanglement forging*, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309

[2] Wei Tang, Teague Tomesh, Martin Suchara, Jeffrey Larson, Margaret Martonosi, *CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations*, https://arxiv.org/abs/2012.02333

----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
