# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper functions that are used in the code."""

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from .best_first_search import BestFirstSearch


def QCtoCCOCircuit(circuit: QuantumCircuit):
    """Convert a qiskit quantum circuit object into a circuit list that is compatible with the SimpleGateList.

    Args:
    circuit: QuantumCircuit object.

    Returns:
    circuit_list_rep: list of circuit gates along with qubit numbers associated to each gate, represented in a
    form that is compatible with SimpleGateList and is of the form:

    ['barrier',
    ('barrier', <qubit_name>),
    ((<gate_name> [, <parameters>]), <qubit_name_1> ... <qubit_name_n>)].


    TODO: Extend this function to allow for circuits with (mid-circuit or other) measurements, as needed.
    """

    circuit_list_rep = list()
    num_circuit_instructions = len(circuit.data)

    for i in range(num_circuit_instructions):
        gate_instruction = circuit.data[i]
        instruction_name = gate_instruction.operation.name
        qubit_ref = gate_instruction.qubits
        params = gate_instruction.operation.params
        circuit_element = instruction_name

        if (
            circuit_element == "barrier" and len(qubit_ref) == circuit.num_qubits
        ):  # barrier across all qubits is not assigned to a specific qubit.
            circuit_list_rep.append(circuit_element)
        else:
            circuit_element = (circuit_element,)
            if params:
                circuit_element += tuple(params[i] for i in range(len(params)))
                circuit_element = (circuit_element,)
            for j in range(len(qubit_ref)):
                qubit_index = qubit_ref[j].index
                circuit_element += (qubit_index,)
            circuit_list_rep.append(circuit_element)

    return circuit_list_rep


def CCOtoQCCircuit(interface):
    """Convert the cut circuit outputted by the CircuitCuttingOptimizer into a qiskit.QuantumCircuit object.

    Args:
    interface: A SimpleGateList object whose attributes carry information about the cut circuit.

    Returns:
    qc_cut: The SimpleGateList converted into a qiskit.QuantumCircuit object,
    """
    cut_circuit_list = interface.exportCutCircuit(name_mapping=None)
    num_qubits = interface.getNumWires()
    cut_circuit_list_len = len(cut_circuit_list)
    cut_types = interface.cut_type
    qc_cut = QuantumCircuit(num_qubits)
    for i in range(cut_circuit_list_len):
        op = cut_circuit_list[
            i
        ]  # the operation, including gate names and qubits acted on.
        gate_qubits = len(op) - 1  # number of qubits involved in the operation.
        if (
            cut_types[i] is None
        ):  # only append gates that are not cut to qc_cut. 
            if type(op[0]) is tuple:
                params = [i for i in op[0][1:]]
                gate_name = op[0][0]
            else:
                params = []
                gate_name = op[0]
            inst = Instruction(gate_name, gate_qubits, 0, params)
            qc_cut.append(inst, op[1 : len(op)])
    return qc_cut


def selectSearchEngine(
    stage_of_optimization,
    optimization_settings,
    search_space_funcs,
    stop_at_first_min=False,
):
    engine = optimization_settings.getEngineSelection(stage_of_optimization)

    if engine == "BestFirst":
        return BestFirstSearch(
            optimization_settings,
            search_space_funcs,
            stop_at_first_min=stop_at_first_min,
        )

    else:
        assert False, f"Invalid stage_of_optimization {stage_of_optimization}"


def greedyBestFirstSearch(state, search_space_funcs, *args):
    """Perform greedy best-first search using the input starting state and
    the input search-space functions.  The resulting goal state is returned,
    or None if a deadend is reached (no backtracking is performed).  Any
    additional input arguments are pass as additional arguments to the
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
