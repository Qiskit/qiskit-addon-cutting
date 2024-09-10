.. _circuit cutting explanation:

####################
Explanatory material
####################

Overview of circuit cutting
---------------------------
Circuit cutting is a technique to increase the size of circuits we can run on quantum hardware at the cost of an additional sampling overhead. A larger quantum circuit can be decomposed by cutting its gates and/or wires, resulting in smaller circuits which can be executed within the constraints of available quantum hardware. The results of these smaller circuits are combined to reconstruct the outcome of the original problem. Circuit cutting can also be used to engineer gates between distant qubits which would otherwise require a large swap overhead.

Key terms
-----------------
* subcircuits: The set of circuits resulting from cutting gates in a :class:`~qiskit.circuit.QuantumCircuit` and then separating the disconnected qubit subsets into smaller circuits. These circuits contain :class:`.SingleQubitQPDGate`\ s and will be used to instantiate each unique subexperiment.

* subexperiments: A term used to describe the unique circuit samples associated with a subcircuit. These circuits have had their :class:`.BaseQPDGate`\ s decomposed into local Qiskit gates and measurements. Subexperiments are the circuits sent to the backend for execution.

* decompose: We try to honor the Qiskit notion of "decompose" in the documentation and API, which loosely means transforming a gate into a less-abstracted representation. *Occasionally*, we may use the term "decompose" to refer to the act of inserting :class:`.BaseQPDGate` instances into quantum circuits as "decomposing" a gate or wire; however, we try to use terms like "partition" and "cut" when referring to this to avoid ambiguity with Qiskit language.

Circuit cutting as a quasiprobability decomposition (QPD)
---------------------------------------------------------
Quasiprobability decomposition is a technique which can be used to simulate quantum circuit executions that go beyond the actual capabilities of current quantum hardware while using that same hardware.  It forms the basis of many error mitigation techniques, which allow simulating a noise-free quantum computer using a noisy one.  Circuit cutting techniques, which allow simulating a quantum circuit using fewer qubits than would otherwise be necessary, can also be phrased in terms of a quasiprobability decomposition.  No matter the goal, the cost of the quasiprobability decomposition is an exponential overhead in the number of circuit executions which must be performed.  In certain cases, this tradeoff is worth it, because it can allow the estimation of quantities that would otherwise be impossible on today's hardware.

There are two types of cuts: gate cuts and wire cuts.  Gate cuts, also known as "space-like" cuts, exist when the cut goes through a gate operating on two (or more) qubits.  Wire cuts, also known as "time-like" cuts, are direct cuts through a qubit wire, essentially a single-qubit identity gate that has been cut into two pieces.  In this package, a wire cut is represented by introducing a new qubit into the circuit and moving remaining operations after the cut identity gate to the new qubit; see :ref:`wire cutting as move`, below.

There are `three settings <https://research.ibm.com/blog/circuit-knitting-with-classical-communication>`__ to consider for circuit cutting.  The first is where only local operations (LO) [i.e., local *quantum* operations] are available.  The other settings introduce classical communication between the circuit executions, which is known in the quantum information literature as LOCC, for `local operations and classical communication <https://en.wikipedia.org/wiki/LOCC>`__.  The LOCC can be either near-time, one-directional communication between the circuit executions (the second setting), or real-time, bi-directional communication (the third setting).

As mentioned above, the cost of any simulation based on quasiprobability distribution is an exponential sampling overhead.
The sampling overhead is the factor by which the overall number of shots must increase for the quasiprobability decomposition to result in the same amount of error, :math:`\epsilon`, as one would get by executing the original circuit.
The overhead of a cut gate depends on which gate is cut; see the final appendix of [`1 <https://arxiv.org/abs/2205.00016>`__] for details.
For instance, a single cut CNOT gate incurs a sampling overhead of 9 [`2 <https://arxiv.org/abs/1909.07534>`__,\ `6 <https://arxiv.org/abs/2312.11638>`__].
A circuit with :math:`n` wire cuts incurs a sampling overhead of O(:math:`16^n`) when classical communication is not available (LO setting); this is reduced to O(:math:`4^n`) when classical communication is available (LOCC setting) [`4 <https://arxiv.org/abs/2302.03366>`__].
However, wire cutting with classical communication (LOCC) is not yet supported by this package (see issue `#264 <https://github.com/Qiskit/qiskit-addon-cutting/issues/264>`__).

The QPD can be given explicitly as follows:

.. math::
   :label: eq:qpd

   \mathcal{U} = \sum_i a_i \mathcal{F}_i ,

where :math:`\mathcal{U}` is the channel implementing the desired operation, and each :math:`a_i` is a real coefficient corresponding to a quantum channel, :math:`\mathcal{F}_i`, that is realizable on hardware.

Note that because this is a sum of channels, *not* a sum of unitaries, expectation values can be evaluated efficiently [`2 <https://arxiv.org/abs/1909.07534>`__].

Results equivalent to original unitary channel can be obtained by a post-processing method, given the coefficients :math:`a_i` and the outcome of each experiment corresponding to the different local (:math:`\mathcal{F}_i`) channels. This post-processing boosts the magnitude of each measurement outcome by :math:`\sum_i \left| a_i \right|`, resulting in a sampling overhead of that quantity squared [`6 <https://arxiv.org/abs/2312.11638>`__]:

.. math::
   :label: eq:sampling-overhead

   \mathrm{Sampling\ overhead} = \left( \sum_i \lvert a_i \rvert \right)^2 .

For more detailed information on the quasiprobability decomposition technique, refer to the paper, Error mitigation for short-depth quantum circuits [`5 <https://arxiv.org/abs/1612.02058>`__].

The essential idea of gate cutting is to replace a two-qubit gate with a linear combination of quantum channels [Eq. :eq:`eq:qpd`] that, when recombined, will allow reconstruction of the result for physically measurable quantities like expectation values.  Note that "global phase" is not something that can be physically measured, so we can disregard it when specifying quasiprobability decompositions.

An example: cutting a :class:`~qiskit.circuit.library.RZZGate`
-------------------------------------------------------------------

As a basic and explicit example, let us consider the decomposition of a cut :class:`~qiskit.circuit.library.RZZGate`.

As shown in [`2 <https://arxiv.org/abs/1909.07534>`__], a quantum circuit which implements an :class:`~qiskit.circuit.library.RZZGate` can be simulated by performing six subexperiments where the :class:`~qiskit.circuit.library.RZZGate` in the original circuit has been replaced with only local (single-qubit) operations [the :math:`\mathcal{F}_i`\ 's in Eq. :eq:`eq:qpd`].  The result is then reconstructed by combining the subexperiment results with certain coefficients [the :math:`a_i`\ 's in Eq. :eq:`eq:qpd`], which can be either positive or negative.  Given the :math:`\theta` parameter of the :class:`~qiskit.circuit.library.RZZGate`, the six subexperiments are as follows:

1. With coefficient :math:`a_1 = \cos^2 (\theta/2)`, do nothing (:math:`I \otimes I`, where :math:`I` is the identity operation on a single qubit).
2. With coefficient :math:`a_2 = \sin^2 (\theta/2)`, perform a :class:`~qiskit.circuit.library.ZGate` on each qubit (:math:`Z \otimes Z`).
3. With coefficient :math:`a_3 = -\sin(\theta)/2`, perform a projective measurement in the Z basis on the first qubit and an :class:`~qiskit.circuit.library.SGate` gate on the second qubit (denote this as :math:`M_z \otimes S`).  If the result of the measurement is 1, flip the sign of that outcome's contribution during reconstruction.
4. With coefficient :math:`a_4 = \sin(\theta)/2`, perform a projective measurement in the Z basis on the first qubit and an :class:`~qiskit.circuit.library.SdgGate` gate on the second qubit (denote this as :math:`M_z \otimes S^\dagger`).  If the result of the measurement is 1, flip the sign of that outcome's contribution during reconstruction.
5. Same as term 3 (:math:`a_5 = a_3`), but swap the qubits (:math:`S \otimes M_z`).
6. Same as term 4 (:math:`a_6 = a_4`), but swap the qubits (:math:`S^\dagger \otimes M_z`).

Equation :eq:`eq:qpd` for :class:`~qiskit.circuit.library.RZZGate` can thus be written as a sum of the six terms listed above.  The following plot shows the magnitude of each coeffient (negative coefficients are in orange) as a function of :math:`\theta`.  The square root of the optimal sampling overhead, denoted by :math:`\gamma`, is given by the sum of the absolute coefficients.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from qiskit.circuit.library import RZZGate
   from qiskit_addon_cutting.qpd import QPDBasis

   theta_values = np.linspace(0, np.pi, 101)
   bases = [QPDBasis.from_instruction(RZZGate(theta)) for theta in theta_values]

   colors = ["#57ffff", "#2B568C", "#007da3", "#ffa502", "#7abaff", "#f2cc86"]
   labels = ['$I \otimes I$ ','$Z \otimes Z$','$M_z \otimes S$','$-M_z \otimes S^\dagger$','$S \otimes M_z$','$-S^\dagger \otimes M_z$']
   plt.stackplot(theta_values, *zip(*[np.abs(basis.coeffs) for basis in bases]), labels=labels, colors=colors)
   plt.axvline(np.pi / 2, c="#aaaaaa", linestyle="dashed")
   plt.axvline(np.pi / 4, c="#aaaaaa", linestyle="dotted")
   plt.axhline(1, c="#aaaaaa", linestyle="solid")
   plt.legend(loc='upper right')
   plt.xlim(0, np.pi)
   plt.ylim(0, 3.6)
   plt.xlabel(r"RZZGate rotation angle $\theta$")
   plt.ylabel("Absolute coefficients, stacked (sum = $\gamma$)")
   plt.title("Quasiprobability decomposition for RZZGate")
   plt.gca().set_xticks(np.linspace(0, np.pi, 5))
   plt.gca().set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
   plt.annotate("CXGate\nfamily", (np.pi / 2, 3), textcoords="offset points", xytext=(-5, 10), ha="right")
   plt.annotate("CSGate\nfamily", (np.pi / 4, 1 + np.sqrt(2)), textcoords="offset points", xytext=(-5, 10), ha="right")

Let's consider some special points in this plot:

- When :math:`\theta = 0`, the gate has no effect, and the sampling overhead is 1.  (Because the overhead is multiplicative, this is equivalent to there being no overhead.)

- When :math:`\theta = \pi`, the gate is equivalent to :math:`Z \otimes Z` up to a global phase, and the sampling overhead is again 1.

- The maximum sampling overhead of :math:`3^2 = 9` is reached at a ZZ Rotation of :math:`\theta=\pi/2`.  We call this point the :class:`~qiskit.circuit.library.CXGate` family because this rotation is equivalent to a CXGate up to (single-qubit) local unitary operations.  This point is also equivalent, up to local unitary operations, to :class:`~qiskit.circuit.library.CZGate`, :class:`~qiskit.circuit.library.CYGate`, :class:`~qiskit.circuit.library.CHGate`, and :class:`~qiskit.circuit.library.ECRGate`.

- The ZZ rotation at :math:`\theta=\pi/4` has sampling overhead of :math:`3+2\sqrt{2} \approx 5.828`.  We call this the :class:`~qiskit.circuit.library.CSGate` family because this rotation is equivalent to a CSGate up to (single-qubit) local operations.  This family also includes :class:`~qiskit.circuit.library.CSdgGate` and :class:`~qiskit.circuit.library.CSXGate`.

- Likewise, :class:`~qiskit.circuit.library.RXXGate`, :class:`~qiskit.circuit.library.RYYGate`, and :class:`~qiskit.circuit.library.RZXGate` are all locally equivalent to :class:`~qiskit.circuit.library.RZZGate`.  The controlled rotations :class:`~qiskit.circuit.library.CRXGate`, :class:`~qiskit.circuit.library.CRYGate`, :class:`~qiskit.circuit.library.CRZGate`, and :class:`~qiskit.circuit.library.CPhaseGate` at an angle of :math:`2\theta` are locally equivalent to :class:`~qiskit.circuit.library.RZZGate` at an angle of :math:`\theta`.

More general cut two-qubit gates via the KAK decomposition
----------------------------------------------------------

We can formalize this notion of local unitary equivalence and expand it to all two-qubit gates using the KAK decomposition, given by

.. math::
   :label: eq:kak

   U = (V_1 \otimes V_2) \exp \left[ i \left( \theta_x \, X \otimes X + \theta_y \, Y \otimes Y + \theta_z \, Z \otimes Z \right) \right] (V_3 \otimes V_4) ,

where :math:`V_1`, :math:`V_2`, :math:`V_3`, and :math:`V_4` are local, single-qubit operations, and the two-qubit portion of the interaction is parametrized entirely by :math:`\vec{\theta} = (\theta_x, \theta_y, \theta_z)`.  By convention, we have chosen :math:`\vec{\theta}` to be in the "Weyl chamber" restricted by :math:`\pi/4 \geq \theta_x \geq \theta_y \geq | \theta_z | \geq 0` [`6 <https://arxiv.org/abs/2312.11638>`__].
For more information on the KAK decomposition, see Ref. [`7 <https://arxiv.org/abs/quant-ph/0209120>`__].

The code that generates subexperiments from the KAK decomposition currently follows Ref. [`3 <https://arxiv.org/abs/2006.11174>`__], which is now known to be non-optimal.  A provably optimal method has been presented in Ref. [`6 <https://arxiv.org/abs/2312.11638>`__], but this newer method has not yet been implemented in this package (see issue `#531 <https://github.com/Qiskit/qiskit-addon-cutting/issues/531>`__).

.. _wire cutting as move:

Wire cutting phrased as a two-qubit :class:`.Move` operation
------------------------------------------------------------

A wire cut is represented fundamentally by this package as a two-qubit :class:`.Move` instruction, which is defined as a reset of the second qubit followed by a swap of both qubits.  Equivalently, the operation is defined as transferring the state of the first qubit wire to the second qubit wire, while simultaneously discarding the state of the second qubit wire (the first qubit ends up in state :math:`\lvert 0 \rangle`).

We have chosen to represent wire cuts in this way primarily because it is consistent with the way one must treat wire cuts when acting on physical qubits: for instance, a wire cut might take the state of physical qubit :math:`n` and continue it as physical qubit :math:`m` after the cut.  Our choice also has the benefit of allowing us to think of "instruction cutting" as a unified framework for considering both wire cuts and gate cuts in the same formalism, being that a wire cut is just a cut :class:`.Move` instruction.

More information on this formalism is given in Sec. 3 of Ref. [`4 <https://arxiv.org/abs/2302.03366>`__]

If you prefer to place cut wires abstractly on a single qubit wire, please see the `how-to guide on placing wire cuts using a single-qubit instruction <../how-tos/how_to_specify_cut_wires.ipynb>`__, which explains how to use the :func:`.cut_wires` function to convert a circuit with :class:`.CutWire` instructions to a circuit with :class:`.Move`\ s on additional qubits.

Sample weights in the Qiskit addon for circuit cutting
------------------------------------------------------
In this package, the number of samples taken from the distribution is generally controlled by a ``num_samples`` argument, and each sample has an associated weight which is used during expectation value reconstruction. Each weight with absolute value above a threshold of 1 / ``num_samples`` will be evaluated exactly.  The remaining low-probability elements -- those in the tail of the distribution -- will then be sampled, resulting in at most ``num_samples`` unique weights. Setting ``num_samples`` to infinity indicates that all weights should be generated rigorously, rather than by sampling from the distribution.

Much of the circuit cutting literature describes a process where we sample from the distribution, take a single shot, then sample from the distribution again and repeat; however, this is not feasible in practice, so we instead perform all sampling upfront.  For now, because of limitations in version 1 of the Qiskit primitives, we take a fixed number of shots for each considered subexperiment and send the subexperiments to the backend(s) in batches. During reconstruction, each subexperiment contributes to the final result with proportion equal to its weight.  One must ensure the number of shots taken is sufficient for the heaviest weighted subexperiment.  In the future, we plan to support passing an individual ``shots`` count with each subexperiment to Qiskit Runtime, so that each subexperiment will be run with a number of shots proportional to that subexperiment's weight in the final result (see issue `#532 <https://github.com/Qiskit/qiskit-addon-cutting/issues/532>`__).  This per-experiment shots count is a new feature enabled by version 2 of the Qiskit primitives.

Sampling overhead reference table
---------------------------------

The below table provides the sampling overhead factor for a variety of two-qubit instructions, provided that only a single instruction is cut.

+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| Instruction(s)                                 | KAK decomposition angles          | Sampling overhead factor                                                |
+================================================+===================================+=========================================================================+
| :class:`~qiskit.circuit.library.CSGate`,       | :math:`(\pi/8, 0, 0)`             | :math:`3+2\sqrt{2} \approx 5.828`                                       |
| :class:`~qiskit.circuit.library.CSdgGate`,     |                                   |                                                                         |
| :class:`~qiskit.circuit.library.CSXGate`       |                                   |                                                                         |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| :class:`~qiskit.circuit.library.CXGate`,       | :math:`(\pi/4, 0, 0)`             | :math:`3^2=9`                                                           |
| :class:`~qiskit.circuit.library.CYGate`,       |                                   |                                                                         |
| :class:`~qiskit.circuit.library.CZGate`,       |                                   |                                                                         |
| :class:`~qiskit.circuit.library.CHGate`,       |                                   |                                                                         |
| :class:`~qiskit.circuit.library.ECRGate`       |                                   |                                                                         |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| :class:`~qiskit.circuit.library.iSwapGate`,    | :math:`(\pi/4, \pi/4, 0)`         | :math:`7^2=49`                                                          |
| :class:`~qiskit.circuit.library.DCXGate`       |                                   |                                                                         |
+------------------------------------------------+-----------------------------------+                                                                         +
| :class:`~qiskit.circuit.library.SwapGate`      | :math:`(\pi/4,\pi/4,\pi/4)`       |                                                                         |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| :class:`~qiskit.circuit.library.RXXGate`,      | :math:`(|\theta/2|, 0, 0)`        | :math:`\left[1 + 2 \left|\sin(\theta)\right| \right]^2`                 |
| :class:`~qiskit.circuit.library.RYYGate`,      |                                   |                                                                         |
| :class:`~qiskit.circuit.library.RZZGate`,      |                                   |                                                                         |
| :class:`~qiskit.circuit.library.RZXGate`       |                                   |                                                                         |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| :class:`~qiskit.circuit.library.CRXGate`,      | :math:`(|\theta/4|, 0, 0)`        | :math:`\left[1 + 2 \left|\sin(\theta/2)\right| \right]^2`               |
| :class:`~qiskit.circuit.library.CRYGate`,      |                                   |                                                                         |
| :class:`~qiskit.circuit.library.CRZGate`,      |                                   |                                                                         |
| :class:`~qiskit.circuit.library.CPhaseGate`    |                                   |                                                                         |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| :class:`~qiskit.circuit.library.XXPlusYYGate`, | :math:`(|\theta/4|,|\theta/4|,0)` | :math:`\left[1+4\left|\sin(\theta/2)\right|+2\sin^2(\theta/2)\right]^2` |
| :class:`~qiskit.circuit.library.XXMinusYYGate` |                                   | (independent of :math:`\beta` parameter)                                |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+
| :class:`.Move` (cut wire) without classical    | not applicable                    | :math:`4^2=16`                                                          |
| communication (i.e., in the LO setting)        |                                   |                                                                         |
+------------------------------------------------+-----------------------------------+-------------------------------------------------------------------------+

Current limitations
-------------------
* The workflow only allows taking the *expectation value* of observables with respect to a circuit.  Limited support for reconstructing an output probability distribution may be added to a future version of this package (see issue `#259 <https://github.com/Qiskit/qiskit-addon-cutting/issues/259>`__).
* Due to current code limitations, some of the generated subexperiments are redundant.  This can result in more subexperiments than expected, particularly when using wire cutting.  This is tracked by issue `#262 <https://github.com/Qiskit/qiskit-addon-cutting/issues/262>`__.

References
----------

This module is based on the theory described in the
following papers:

[1] Christophe Piveteau, David Sutter, *Circuit knitting with classical communication*,
https://arxiv.org/abs/2205.00016

[2] Kosuke Mitarai, Keisuke Fujii, *Constructing a virtual two-qubit gate by sampling
single-qubit operations*,
https://arxiv.org/abs/1909.07534

[3] Kosuke Mitarai, Keisuke Fujii, *Overhead for simulating a non-local channel with local channels by quasiprobability sampling*,
https://arxiv.org/abs/2006.11174

[4] Lukas Brenner, Christophe Piveteau, David Sutter, *Optimal wire cutting with
classical communication*,
https://arxiv.org/abs/2302.03366

[5] K. Temme, S. Bravyi, and J. M. Gambetta, *Error mitigation for short-depth quantum circuits*,
https://arxiv.org/abs/1612.02058

[6] Lukas Schmitt, Christophe Piveteau, David Sutter, *Cutting circuits with multiple two-qubit unitaries*,
https://arxiv.org/abs/2312.11638

[7] Jun Zhang, Jiri Vala, K. Birgitta Whaley, Shankar Sastry, *A geometric theory of non-local two-qubit operations*,
https://arxiv.org/abs/quant-ph/0209120
