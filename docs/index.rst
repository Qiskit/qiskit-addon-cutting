########################
Circuit Knitting Toolbox
########################

Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor(s), and then knitting their results into a reconstruction of the original circuit's outcome. Circuit knitting includes techniques such as entanglement forging, circuit cutting, and classical embedding. The Circuit Knitting Toolbox (CKT) is a collection of such tools.

Each tool in the CKT partitions a user's problem into quantum and classical components to enable efficient use of resources constrained by scaling limits, i.e. size of quantum processors and classical compute capability. It is designed to work seamlessly with the `Quantum Serverless <https://qiskit-extensions.github.io/quantum-serverless/>`_ framework, which enables users to run parallelized and hybrid (quantum + classical) workloads without worrying about allocating and managing underlying infrastructure.

The toolbox currently contains the following tools:

- Entanglement Forging
- Circuit Cutting

The source code to the toolbox is available `on GitHub <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox>`_.

.. note::

   The `Quantum Serverless <https://qiskit-extensions.github.io/quantum-serverless/>`_ framework is documented separately, as it lives in its own repository.  Check out `Tutorial 2: Forging with Quantum Serverless <./tutorials/entanglement_forging/tutorial_2_forging_with_quantum_serverless.ipynb>`_  and `Tutorial 3: Circuit Cutting with Quantum Serverless <./tutorials/circuit_cutting/tutorial_3_cutting_with_quantum_serverless.ipynb>`_ for examples on how to integrate Quantum Serverless into circuit knitting workflows.

This project is meant to evolve rapidly and, as such, does not follow `Qiskit's deprecation policy <https://qiskit.org/documentation/contributing_to_qiskit.html#deprecation-policy>`_.  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the release notes.

Contents
--------

.. toctree::
  :maxdepth: 3

  Installation Instructions <install>
  Tutorials <tutorials/index>
  Explanatory Material <explanation/index>
  How-To Guides <how-tos/index>
  API References <apidocs/index>
  Release Notes <release-notes>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
