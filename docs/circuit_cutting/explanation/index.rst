###################################################
Explanatory material for the circuit cutting module
###################################################

Overview of circuit cutting
---------------------------
Circuit cutting is a technique to increase the size of circuits we can run on quantum hardware at the cost of an additional sampling overhead. A larger quantum circuit can be decomposed by cutting its gates and wires into smaller circuits, which can be executed within the constraints of available quantum hardware. The results of these smaller circuits are combined to reconstruct the outcome of the original problem. Circuit cutting can also be used to engineer gates between distant qubits which would otherwise require a large swap overhead.

Quasiprobability Decomposition (QPD)
------------------------------------
Quasiprobability decomposition is a technique which can be used to simulate noise-free quantum gates using only noisy, local operations. This is often referred to as circuit cutting. One could cut a non-local gate and simulate it using only local operations. This is referred to as a "gate cut" or "space-like cut". One could also imagine cutting a single-qubit identity gate and simulating it using local operations on either side of the cut. The other side of the cut would be simulated by introducing a new qubit into the circuit and moving remaining operations after the cut identity gate to the new qubit. This is referred to as a "wire cut" or "space-like cut". The cost of conducting these techniques is an exponential sampling overhead. If no real-time classical communication is available between qubits of the cut gate, gate cuts incur a sampling overhead of O(9\ :sup:`n`), and wire cuts incur a sampling overhead of O(16\ :sup:`n`). If real-time communication is available (i.e. dynamic circuits), the sampling overhead for both gate and wire cuts is reduced to O(4\ :sup:`n`). [`3 <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/tree/cutting-workflow#references>`_] [`4 <https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/tree/cutting-workflow#references>`_]

For more detailed information on this technique, refer to the paper, `Error mitigation for short-depth quantum circuits <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`_. [6]

Key terms
-----------------
* subcircuits: The set of circuits resulting from cutting gates in a `QuantumCircuit` and then separating the disconnected qubit subsets into smaller circuits. These circuits contain `SingleQubitQPDGate`\ s and will be used to instantiate each unique subexperiment.

* subexperiments: A term used to describe the unique circuit samples associated with a subcircuit. These circuits have had their `BaseQPDGate`\ s decomposed into local Qiskit gates and measurements. Subexperiments are the circuits sent to the backend for execution.

* decompose: We try to honor the Qiskit notion of "decompose" in the documentation, which loosely means transforming a gate into a less-abstracted representation. *Ocassionally*, we may use the term "decompose" to refer to the act of inserting `BaseQPDGate` instances into quantum circuits as "decomposing" a gate or wire; however, we try to use terms like "partition" and "cut" when referring to this to avoid ambiguity with Qiskit language.

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
