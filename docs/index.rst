########################
Circuit Knitting Toolbox
########################

.. image:: https://img.shields.io/github/stars/Qiskit-Extensions/circuit-knitting-toolbox?style=social
   :alt: GitHub repository star counter badge
   :target: https://github.com/Qiskit-Extensions/circuit-knitting-toolbox

Circuit Knitting is the process of decomposing a quantum circuit into smaller circuits, executing those smaller circuits on a quantum processor(s), and then knitting their results into a reconstruction of the original circuit's outcome.

The toolbox currently contains the following tools:

- Circuit Cutting

For a more detailed discussion on circuit cutting, check out our `technical guide <https://qiskit-extensions.github.io/circuit-knitting-toolbox/circuit_cutting/explanation/index.html#overview-of-circuit-cutting>`__.

This project is meant to evolve rapidly and, as such, does not follow `Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`__.  We may occasionally make breaking changes in order to improve the user experience.  When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.  Each substantial improvement, breaking change, or deprecation will be documented in the :ref:`release notes`.

Citing this project
-------------------

If you use the Circuit Knitting Toolbox in your research, please cite it according to ``CITATON.bib`` file included in this repository:

.. literalinclude:: ../CITATION.bib
   :language: bibtex

Developer guide
---------------

The source code to the toolbox is available `on GitHub <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox>`__.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/blob/main/CONTRIBUTING.md>`__ in the root of this project's repository.

Contents
--------

.. toctree::
  :maxdepth: 2

  About Circuit Knitting Toolbox <self>
  Installation Instructions <install>

.. toctree::
  :maxdepth: 2
  :caption: Circuit Cutting

  Cutting Tutorials <circuit_cutting/tutorials/index>
  Cutting Explanatory Material <circuit_cutting/explanation/index>
  Cutting How-To Guides <circuit_cutting/how-tos/index>
  CutQC (legacy circuit cutting implementation) <circuit_cutting/cutqc/index>

.. toctree::
  :maxdepth: 2
  :caption: References

  API References <apidocs/index>
  Release Notes <release-notes>
  GitHub <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
