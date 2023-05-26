###################################################
Explanatory material for the circuit cutting module
###################################################

Overview of circuit cutting
---------------------------
Circuit cutting is a technique to increase the size of circuits we can run on quantum hardware at the cost of an additional sampling overhead. A larger quantum circuit can be decomposed by cutting its gates and/or wires, resulting in smaller circuits which can be executed within the constraints of available quantum hardware. The results of these smaller circuits are combined to reconstruct the outcome of the original problem. Circuit cutting can also be used to engineer gates between distant qubits which would otherwise require a large swap overhead.

Quasiprobability Decomposition (QPD)
------------------------------------
Quasiprobability decomposition is a technique which can be used to simulate noise-free quantum gates using only noisy, local operations (LO). This is often referred to as circuit cutting. One could cut a non-local gate and simulate it using only local operations. This is referred to as a "gate cut" or "space-like cut". One could also imagine cutting a single-qubit identity gate and simulating it using local operations on either side of the cut. The other side of the cut would be simulated by introducing a new qubit into the circuit and moving remaining operations after the cut identity gate to the new qubit. This is referred to as a "wire cut" or "time-like cut". The cost of conducting these techniques is an exponential sampling overhead. If no real-time classical communication is available between qubits of the cut gate or wire, gate cuts incur a sampling overhead of O(9\ :sup:`n`), and wire cuts incur a sampling overhead of O(16\ :sup:`n`), where n is the total number of cuts. If real-time communication is available (i.e. dynamic circuits), the sampling overhead for both gate and wire cuts may be reduced to O(4\ :sup:`n`) [`1 <https://arxiv.org/abs/2205.00016>`__] [`3 <https://arxiv.org/abs/2302.03366>`__]; however, support for circuit cutting with classical communication (LOCC) is not yet supported in CKT.

For more detailed information on this technique, refer to the paper, Error mitigation for short-depth quantum circuits [`4 <https://arxiv.org/abs/1612.02058>`__].

Key terms
-----------------
* subcircuits: The set of circuits resulting from cutting gates in a :class:`QuantumCircuit` and then separating the disconnected qubit subsets into smaller circuits. These circuits contain :class:`SingleQubitQPDGate`\ s and will be used to instantiate each unique subexperiment.

* subexperiments: A term used to describe the unique circuit samples associated with a subcircuit. These circuits have had their :class:`BaseQPDGate`\ s decomposed into local Qiskit gates and measurements. Subexperiments are the circuits sent to the backend for execution.

* decompose: We try to honor the Qiskit notion of "decompose" in the documentation and API, which loosely means transforming a gate into a less-abstracted representation. *Occasionally*, we may use the term "decompose" to refer to the act of inserting :class:`BaseQPDGate` instances into quantum circuits as "decomposing" a gate or wire; however, we try to use terms like "partition" and "cut" when referring to this to avoid ambiguity with Qiskit language.

Current limitations
-------------------
* QPD-based wire cutting will be available no sooner than CKT v0.3.0. The `cutqc <https://qiskit-extensions.github.io/circuit-knitting-toolbox/circuit_cutting/cutqc/index.htmlpackage>`__ package may be used for wire cutting in the meantime.
* ``PauliList`` is the only supported observable format until no sooner than CKT v0.3.0.

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

[4] K. Temme, S. Bravyi, and J. M. Gambetta, *Error mitigation for short-depth quantum circuits*,
https://arxiv.org/abs/1612.02058
