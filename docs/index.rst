#############################
Qiskit addon: circuit cutting
#############################

.. image:: https://img.shields.io/github/stars/Qiskit/qiskit-addon-cutting?style=social
   :alt: GitHub repository star counter badge
   :target: https://github.com/Qiskit/qiskit-addon-cutting

`Qiskit addons <https://docs.quantum.ibm.com/guides/addons>`_ are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package implements circuit cutting.  In this technique, a handful of gates and/or wires are cut, resulting in smaller circuits that are better suited for execution on hardware.  The result of the original circuit can then be reconstructed; however, the trade-off is that the overall number of shots must be increased by a factor exponential in the number of cuts.

For a more detailed discussion on circuit cutting, check out our `technical guide <./circuit_cutting/explanation/index.rst#overview-of-circuit-cutting>`__.

We follow `semantic versioning <https://semver.org/>`__ and are guided by the principles in  `Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`__.  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the :ref:`release notes`.

.. note::
   This package was known as the Circuit Knitting Toolbox prior to September 2024.

Citing this project
-------------------

If you use this package in your research, please cite it according to ``CITATON.bib`` file included in this repository:

.. literalinclude:: ../CITATION.bib
   :language: bibtex

If you are using the entanglement forging tool in Circuit Knitting Toolbox version 0.5 or earlier, please use `an older version of the citation file <https://github.com/Qiskit/qiskit-addon-cutting/blob/stable/0.5/CITATION.bib>`__ which includes the authors of that tool.

If you are using the CutQC tool in Circuit Knitting Toolbox version 0.7 or earlier, please use `an older version of the citation file <https://github.com/Qiskit/qiskit-addon-cutting/blob/stable/0.7/CITATION.bib>`__ which includes the authors of that tool.

Developer guide
---------------

The source code to this package is available `on GitHub <https://github.com/Qiskit/qiskit-addon-cutting>`__.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-cutting/blob/main/CONTRIBUTING.md>`__ in the root of this project's repository.

Contents
--------

.. toctree::
  :maxdepth: 2

  Documentation Home <self>
  Installation Instructions <install>
  Tutorials <tutorials/index>
  Explanatory Material <explanation/index>
  How-To Guides <how-tos/index>
  API References <apidocs/index>
  GitHub <https://github.com/Qiskit/qiskit-addon-cutting>
  Release Notes <release-notes>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
