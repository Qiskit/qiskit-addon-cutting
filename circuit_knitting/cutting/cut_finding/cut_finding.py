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

from .optimization_settings import OptimizationSettings
from .quantum_device_constraints import DeviceConstraints
from .circuit_interface import SimpleGateList
from .lo_cuts_optimizer import LOCutsOptimizer
from .utils import QCtoCCOCircuit


def find_cuts(
    circuit: QuantumCircuit,
    optimization: OptimizationSettings | dict[str, str | int],
    constraints: DeviceConstraints | dict[str, int],
):
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

    optimizer = LOCutsOptimizer(interface, opt_settings, constraint_settings)
    out = optimizer.optimize()

    print(
        " Gamma =",
        None if (out is None) else out.upperBoundGamma(),
        ", Min_gamma_reached =",
        optimizer.minimumReached(),
    )
    if out is not None:
        out.print(simple=True)
    else:
        print(out)

    print(
        "Subcircuits:",
        interface.exportSubcircuitsAsString(name_mapping="default"),
        "\n",
    )
