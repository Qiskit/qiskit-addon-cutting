# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Automatically find cut locations in a quantum circuit."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

from .optimization_settings import OptimizationSettings
from .quantum_device_constraints import DeviceConstraints
from .circuit_interface import SimpleGateList
from .lo_cuts_optimizer import LOCutsOptimizer
from .cco_utils import QCtoCCOCircuit
from ..instructions import CutWire
from ..cutting_decomposition import cut_gates


def find_cuts(
    circuit: QuantumCircuit,
    optimization: OptimizationSettings | dict[str, str | int],
    constraints: DeviceConstraints | dict[str, int],
) -> QuantumCircuit:
    """
    Find cut locations in a circuit, given optimization settings and QPU constraints.

    Args:
        circuit: The circuit to cut
        optimization: Settings for controlling optimizer behavior. Currently,
            only a best-first optimizer is supported. For a list of supported
            optimization settings, see :class:`.OptimizationSettings`.
        constraints: QPU constraints used to generate the cut location search space.
            For information on how to specify QPU constraints, see :class:`.DeviceConstraints`.

    Returns:
        A circuit containing :class:`.BaseQPDGate` instances. The subcircuits
        resulting from cutting these gates will be runnable on the devices
        specified in ``constraints``.
    """
    circuit_cco = QCtoCCOCircuit(circuit)
    interface = SimpleGateList(circuit_cco)

    if isinstance(optimization, dict):
        opt_settings = OptimizationSettings.from_dict(optimization)
    else:
        opt_settings = optimization

    # Hard-code the optimization type to best-first
    opt_settings.setEngineSelection("CutOptimization", "BestFirst")

    if isinstance(constraints, dict):
        constraint_settings = DeviceConstraints.from_dict(constraints)
    else:
        constraint_settings = constraints

    # Hard-code the optimizer to an LO-only optimizer
    optimizer = LOCutsOptimizer(interface, opt_settings, constraint_settings)

    # Find cut locations
    opt_out = optimizer.optimize()

    wire_cut_actions = []
    gate_ids = []
    for action in opt_out.actions:
        if action[0].getName() == "CutTwoQubitGate":
            gate_ids.append(action[1][0])
        else:
            wire_cut_actions.append(action)

    # First, replace all gates to cut with BaseQPDGate instances.
    # This assumes each gate to cut is replaced 1-to-1 with a QPD gate.
    # This may not hold in the future as we stop treating gate cuts individually.
    circ_out = cut_gates(circuit, gate_ids)[0]

    # Insert all the wire cuts
    counter = 0
    for action in sorted(wire_cut_actions, key=lambda a: a[1][0]):
        if action[0].getName() == "CutTwoQubitGate":
            continue
        inst_id = action[1][0]
        qubit_id = action[2][0][0] - 1
        circ_out.data.insert(
            inst_id + counter,
            CircuitInstruction(CutWire(), [circuit.data[inst_id].qubits[qubit_id]], []),
        )
        counter += 1

    return circ_out
