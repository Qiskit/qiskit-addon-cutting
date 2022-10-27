######################################
Circuit Knitting Toolbox documentation
######################################

Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor(s), and then knitting their results into a reconstruction of the original circuit's outcome. Circuit knitting includes techniques such as entanglement forging, circuit cutting, and classical embedding. The Circuit Knitting Toolbox (CKT) is a collection of such tools.

Each tool in the CKT partitions a user's problem into quantum and classical components to enable efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It is designed to work seamlessly with the `Quantum Serverless <https://github.com/Qiskit-Extensions/quantum-serverless>`_ framework, which enables users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

The toolbox currently contains the following tools:

- Entanglement Forging
- Circuit Cutting

Contents
--------

.. toctree::
  :maxdepth: 3

  Installation Instructions <install>
  Tutorials <tutorials/index>
  Explanatory Material <explanation/index>
  How-To Guides <how-tos/index>
  API References <apidocs/index>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
