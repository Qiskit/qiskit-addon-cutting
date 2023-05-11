########################################################
Explanatory material for the circuit cutting module
########################################################

Overview of circuit cutting
---------------------------
Circuit cutting is a technique which involves decomposing
operations in a quantum circuit into local operations specified
by the decompositions' joint quasiprobability decomposition (QPD),
evaluating those circuits on the backend, and finally reconstructing
the simulated output of the full-sized circuit.

Glossary of commonly used terms
-------------------------------
* **decompose**: Replace an abstracted gate with a more explicit representation. `BaseQPDGate`\ s decompose to a set of local operations, as specified by their `basis_id` field.
* **separate**: The act of pulling quantum circuit bits apart into smaller circuits.
* **sample**: A description of one sample from the decomposed instructions' joint quasiprobability distribution.
    * A set of indices to apply to the `basis_id` field of each `BaseQPDGate` in a circuit. These determine to what local operations each `BaseQPDGate` will be decomposed and are generally sampled from the decompositions' joint quasiprobability distribution.
* **subcircuits**: The set of circuits resulting from decomposing and
  separating the input circuit. These circuits contain
  `SingleQubitQPDGate`\ s and will be used to instantiate
  each unique sample.
* **subexperiments**: A term used to describe the unique circuit samples associated with a subcircuit. These are the circuits sent to the backend for execution.
* **partition**:
    * (noun): A set of qubits to be separated from the circuit by cutting gates and/or wires.
    * (verb): Create qubit partitions by replacing nonlocal gates spanning more than one partition with `TwoQubitQPDGate`\ s.

Scaling
-------

General Considerations
----------------------

Current limitations
---------------------

References
----------

This module is based on the theory described in the
following papers:

[1] Christophe Piveteau, David Sutter, *Circuit knitting with classical communication*,
https://arxiv.org/abs/2205.00016

[2] Kosuke Mitarai, Keisuke Fujii, *Constructing a virtual two-qubit gate by sampling
single-qubit operations*,
https://arxiv.org/abs/1909.07534

[3] Lukas Brenner, Christophe Piveteau, David Sutter, *Optimal wire cutting with
classical communication*,
https://arxiv.org/abs/2302.03366
