How-To Guides
-------------

- `Generate exact quasi-distributions <how_to_generate_exact_quasi_dists_from_sampler.ipynb>`__:
  Use the :class:`~qiskit_addon_cutting.utils.simulation.ExactSampler` interface to generate
  exact quasi-distributions for circuits containing mid-circuit measurements.
- `Generate exact sampling coefficients <how_to_generate_exact_sampling_coefficients.ipynb>`__:
  Generate exact sampling coefficients and run all unique samples from the distribution.
- `Specify cut wires as a single-qubit instruction <how_to_specify_cut_wires.ipynb>`__:
  Perform wire cutting with a single-qubit `CutWire` instruction, rather than a two-qubit `Move` operation.
- `Perform a cutting workflow with multiple observables <how_to_provide_multiple_observables.ipynb>`__:
  Use the :mod:`~qiskit_addon_cutting.utils.observable_terms` utility functions to gather unique observable terms,
  perform circuit cutting experiments, and reconstruct the desired expectation values.
