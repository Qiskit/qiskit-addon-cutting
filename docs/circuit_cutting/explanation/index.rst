###################################################
Explanatory material for the circuit cutting module
###################################################

Overview of circuit cutting
---------------------------
Circuit cutting is a technique to increase the size of circuits we can run on quantum hardware at the cost of an additional sampling overhead. A larger quantum circuit can be decomposed by cutting its gates and wires into smaller circuits, which can be executed within the constraints of available quantum hardware. The results of these smaller circuits are combined to reconstruct the outcome of the original problem. Circuit cutting can also be used to engineer gates between distant qubits which would otherwise require a large swap overhead.

Glossary of terms
-----------------
* subcircuits: The set of circuits resulting from cutting gates in a `QuantumCircuit` and then separating the disconnected qubit subsets into smaller circuits. These circuits contain `SingleQubitQPDGate`\ s and will be used to instantiate each unique subexperiment.

* subexperiments: A term used to describe the unique circuit samples associated with a subcircuit. These circuits have had their `BaseQPDGate`\ s decomposed into local Qiskit gates and measurements. Subexperiments are the circuits sent to the backend for execution.

Current limitations
-------------------
* No support for wire cutting until no sooner than CKT v0.3.0
* `PauliList` is the only supported observable format until no sooner than CKT v.0.3.0

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
