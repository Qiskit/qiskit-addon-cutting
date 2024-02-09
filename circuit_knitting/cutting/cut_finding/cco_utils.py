# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper functions that are used in the code."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Gate
from .optimization_settings import OptimizationSettings
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cut_optimization import CutOptimizationFuncArgs
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .search_space_generator import SearchFunctions
from .best_first_search import BestFirstSearch
from .circuit_interface import CircuitElement, SimpleGateList
from ..qpd import QPDBasis


def qc_to_cco_circuit(circuit: QuantumCircuit) -> list[str | CircuitElement]:
    """Convert a qiskit quantum circuit object into a circuit list that is
    compatible with the :class:`SimpleGateList`. To conform with the uniformity
    of the design, single and multiqubit (that is, gates acting on more than two
    qubits) are assigned :math:`gamma=None`. In the converted list, a barrier
    across the entire circuit is represented by the string "barrier."
    Everything else is represented by an instance of :class:`CircuitElement`.

    Args:
    circuit: an instance of :class:`qiskit.QuantumCircuit` .

    Returns:
    circuit_list_rep: list of circuit instructions represented in a form that is compatible with
    :class:`SimpleGateList` and can therefore be ingested by the cut finder.

    TODO: Extend this function to allow for circuits with (mid-circuit or other)
    measurements, as needed.
    """
    circuit_list_rep = []
    for inst in circuit.data:
        if inst.operation.name == "barrier" and len(inst.qubits) == circuit.num_qubits:
            circuit_element = "barrier"
        else:
            gamma = None
            if isinstance(inst.operation, Gate) and len(inst.qubits) == 2:
                gamma = QPDBasis.from_instruction(inst.operation).kappa
            name = inst.operation.name
            params = inst.operation.params
            circuit_element = CircuitElement(
                name=name,
                params=params,
                qubits=list(circuit.find_bit(q).index for q in inst.qubits),
                gamma=gamma,
            )
        circuit_list_rep.append(circuit_element)

    return circuit_list_rep


def cco_to_qc_circuit(interface: SimpleGateList) -> QuantumCircuit:
    """Convert the cut circuit outputted by the cut finder into a :class:`qiskit.QuantumCircuit` instance.

    Args:
    interface: An instance of :class:`SimpleGateList` whose attributes carry information about the cut circuit.

    Returns:
    qc_cut: The SimpleGateList converted into a :class:`qiskit.QuantumCircuit` instance.

    TODO: This function only works for instances of LO gate cutting.
    Expand to cover the wire cutting case when needed.
    """
    cut_circuit_list = interface.exportCutCircuit(name_mapping=None)
    num_qubits = interface.getNumWires()
    cut_types = interface.cut_type
    qc_cut = QuantumCircuit(num_qubits)
    for k, op in enumerate([cut_circuit for cut_circuit in cut_circuit_list]):
        if cut_types[k] is None:  # only append gates that are not cut.
            op_name = op.name
            op_qubits = op.qubits
            op_params = op.params
            inst = Instruction(op_name, len(op_qubits), 0, op_params)
            qc_cut.append(inst, op_qubits)
    return qc_cut


def selectSearchEngine(
    stage_of_optimization: str,
    optimization_settings: OptimizationSettings,
    search_space_funcs: SearchFunctions,
    stop_at_first_min: bool = False,
) -> BestFirstSearch:
    """Select the search algorithm to use. At present, only Dijkstra's algorithm
    for best first search is supported.
    """
    engine = optimization_settings.getEngineSelection(stage_of_optimization)

    if engine == "BestFirst":
        return BestFirstSearch(
            optimization_settings,
            search_space_funcs,
            stop_at_first_min=stop_at_first_min,
        )

    else:
        raise ValueError(f"Search engine {engine} is not supported.")


def greedyBestFirstSearch(
    state: DisjointSubcircuitsState,
    search_space_funcs: SearchFunctions,
    *args: CutOptimizationFuncArgs,
) -> None | DisjointSubcircuitsState:
    """Perform greedy best-first search using the input starting state and
    the input search-space functions. The resulting goal state is returned,
    or None if a deadend is reached (no backtracking is performed).  Any
    additional input arguments are passed as additional arguments to the
    search-space functions.
    """

    if search_space_funcs.goal_state_func(state, *args):
        return state

    best = min(
        [
            (search_space_funcs.cost_func(next_state, *args), k, next_state)
            for k, next_state in enumerate(
                search_space_funcs.next_state_func(state, *args)
            )
        ],
        default=(None, None, None),
    )

    if best[-1] is not None:
        return greedyBestFirstSearch(best[-1], search_space_funcs, *args)

    else:
        return None
