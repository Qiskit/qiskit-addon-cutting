# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for conducting the wire cutting on quantum circuits."""

from __future__ import annotations

from typing import Sequence, Any, Dict, cast, no_type_check

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.utils.deprecation import deprecate_func
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime import Options, QiskitRuntimeService

from .wire_cutting_evaluation import run_subcircuit_instances
from .wire_cutting_post_processing import generate_summation_terms, build
from .wire_cutting_verification import generate_reconstructed_output


@deprecate_func(
    since="0.2.0",
    additional_msg=(
        "The circuit_knitting_toolbox.circuit_cutting.wire_cutting package is"
        " deprecated and will be removed no sooner than 1 month after its release. "
        "Users should import from circuit_knitting_toolbox.circuit_cutting.cutqc instead."
    ),
)
def cut_circuit_wires(
    circuit: QuantumCircuit,
    method: str,
    subcircuit_vertices: Sequence[Sequence[int]] | None = None,
    max_subcircuit_width: int | None = None,
    max_subcircuit_cuts: int | None = None,
    max_subcircuit_size: int | None = None,
    max_cuts: int | None = None,
    num_subcircuits: Sequence[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Decompose the circuit into a collection of subcircuits.

    Args:
        - method (str): whether to have the cuts be 'automatically' found, in a
            provably optimal way, or whether to 'manually' specify the cuts
        - subcircuit_vertices (Sequence[Sequence[int]]): the vertices to be used in
            the subcircuits. Note that these are not the indices of the qubits, but
            the nodes in the circuit DAG
        - max_subcircuit_width (int): max number of qubits in each subcircuit
        - max_cuts (int): max total number of cuts allowed
        - num_subcircuits (Sequence[int]): list of number of subcircuits to try
        - max_subcircuit_cuts (int, optional): max number of cuts for a subcircuit
        - max_subcircuit_size (int, optional): max number of gates in a subcircuit
        - verbose (bool, optional): flag for printing output of cutting
    Returns:
        (dict[str, Any]): A dictionary containing information on the cuts,
        including the subcircuits themselves (key: 'subcircuits')
    Raises:
        - ValueError: if the input method does not match the other provided arguments
    """
    cuts = {}
    if method == "automatic":
        if max_subcircuit_width is None:
            raise ValueError(
                "The max_subcircuit_width argument must be set if using automatic cut finding."
            )
        cuts = find_wire_cuts(
            circuit=circuit,
            max_subcircuit_width=max_subcircuit_width,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            max_subcircuit_cuts=max_subcircuit_cuts,
            max_subcircuit_size=max_subcircuit_size,
            verbose=verbose,
        )
    elif method == "manual":
        if subcircuit_vertices is None:
            raise ValueError(
                "The subcircuit_vertices argument must be set if manually specifying cuts."
            )
        cuts = cut_circuit_wire(
            circuit=circuit, subcircuit_vertices=subcircuit_vertices, verbose=verbose
        )
    else:
        ValueError(
            'The method argument for the decompose method should be either "automatic" or "manual".'
        )

    return cuts


@deprecate_func(
    since="0.2.0",
    additional_msg=(
        "The circuit_knitting_toolbox.circuit_cutting.wire_cutting package is"
        " deprecated and will be removed no sooner than 1 month after its release. "
        "Users should import from circuit_knitting_toolbox.circuit_cutting.cutqc instead."
    ),
)
def evaluate_subcircuits(
    cuts: dict[str, Any],
    service: QiskitRuntimeService | None = None,
    backend_names: str | Sequence[str] | None = None,
    options: Options | Sequence[Options] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Evaluate the subcircuits.

    Args:
        - cuts (dict): the results of cutting
        - service (QiskitRuntimeService | None): A service for connecting to Qiskit Runtime Service
        - options (Options | Sequence[Options] | None): Options to use on each backend
        - backend_names (str | Sequence[str] | None): The name(s) of the backend(s) to be used
    Returns:
        (dict): the dictionary containing the results from running
        each of the subcircuits
    """
    # Put backend_names and options in lists to ensure it is unambiguous how to sync them
    backends_list: Sequence[str] = []
    options_list: Sequence[Options] = []
    if backend_names is None or isinstance(backend_names, str):
        if isinstance(options, Options):
            options_list = [options]
        elif isinstance(options, Sequence) and (len(options) != 1):
            options_list = [options[0]]
        if isinstance(backend_names, str):
            backends_list = [backend_names]
    else:
        backends_list = backend_names
        if isinstance(options, Options):
            options_list = [options] * len(backends_list)
        elif options is None:
            options_list = [None] * len(backends_list)
        else:
            options_list = options

    if backend_names:
        if len(backends_list) != len(options_list):
            raise AttributeError(
                f"The list of backend names is length ({len(backends_list)}), but the list of options is length ({len(options_list)}). It is ambiguous how these options should be applied."
            )

    _, _, subcircuit_instances = _generate_metadata(cuts)

    subcircuit_instance_probabilities = _run_subcircuits(
        cuts,
        subcircuit_instances,
        service=service,
        backend_names=backends_list,
        options=options_list,
    )

    return subcircuit_instance_probabilities


@deprecate_func(
    since="0.2.0",
    additional_msg=(
        "The circuit_knitting_toolbox.circuit_cutting.wire_cutting package is"
        " deprecated and will be removed no sooner than 1 month after its release. "
        "Users should import from circuit_knitting_toolbox.circuit_cutting.cutqc instead."
    ),
)
def reconstruct_full_distribution(
    circuit: QuantumCircuit,
    subcircuit_instance_probabilities: dict[int, dict[int, np.ndarray]],
    cuts: dict[str, Any],
    num_threads: int = 1,
) -> np.ndarray:
    """
    Reconstruct the full probabilities from the subcircuit evaluations.

    Args:
        - circuit (QuantumCircuit): the original full circuit
        - subcircuit_instance_probabilities (dict): the probability vectors from each
            of the subcircuit instances, as output by the _run_subcircuits function
        - num_threads (int): the number of threads to use to parallelize the recomposing
    Returns:
        - (np.ndarray): the reconstructed probability vector
    """
    summation_terms, subcircuit_entries, _ = _generate_metadata(cuts)

    subcircuit_entry_probabilities = _attribute_shots(
        subcircuit_entries, subcircuit_instance_probabilities
    )

    unordered_probability, smart_order, overhead = build(
        summation_terms=summation_terms,
        subcircuit_entry_probs=subcircuit_entry_probabilities,
        num_cuts=cuts["num_cuts"],
        num_threads=num_threads,
    )

    reconstructed_probability = generate_reconstructed_output(
        circuit,
        cuts["subcircuits"],
        unordered_probability,
        smart_order,
        cuts["complete_path_map"],
    )

    return reconstructed_probability


def _generate_metadata(
    cuts: dict[str, Any]
) -> tuple[
    list[dict[int, int]],
    dict[int, dict[tuple[str, str], tuple[int, Sequence[tuple[int, int]]]]],
    dict[int, dict[tuple[tuple[str, ...], tuple[Any, ...]], int]],
]:
    """
    Generate metadata used to execute subcircuits and reconstruct probabilities of original circuit.

    Args:
        - cuts (dict[str, Any]): results from the cutting step
    Returns:
        - (tuple): information about the 4^(num cuts) summation terms used to reconstruct original
            probabilities, a dictionary with information on each of the subcircuits, and a dictionary
            containing indexes for each of the subcircuits
    """
    (
        summation_terms,
        subcircuit_entries,
        subcircuit_instances,
    ) = generate_summation_terms(
        subcircuits=cuts["subcircuits"],
        complete_path_map=cuts["complete_path_map"],
        num_cuts=cuts["num_cuts"],
    )
    return summation_terms, subcircuit_entries, subcircuit_instances


def _run_subcircuits(
    cuts: dict[str, Any],
    subcircuit_instances: dict[int, dict[tuple[tuple[str, ...], tuple[Any, ...]], int]],
    service: QiskitRuntimeService | None = None,
    backend_names: Sequence[str] | None = None,
    options: Sequence[Options] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Execute all the subcircuit instances.

    task['subcircuit_instance_probs'][subcircuit_idx][subcircuit_instance_idx] = measured prob

    Args:
        - cuts (dict[str, Any]): results from the cutting step
        - subcircuit_instances (dict): the dictionary containing the index information for each
            of the subcircuit instances
        - service (QiskitRuntimeService | None): the arguments for the runtime service
        - backend_names (Sequence[str] | None): the backend(s) used to run the subcircuits
        - options (Options | None): options for the runtime execution of subcircuits
    Returns:
        - (dict): the resulting probabilities from each of the subcircuit instances
    """
    subcircuit_instance_probs = run_subcircuit_instances(
        subcircuits=cuts["subcircuits"],
        subcircuit_instances=subcircuit_instances,
        service=service,
        backend_names=backend_names,
        options=options,
    )

    return subcircuit_instance_probs


def _attribute_shots(
    subcircuit_entries: dict[
        int, dict[tuple[str, str], tuple[int, Sequence[tuple[int, int]]]]
    ],
    subcircuit_instance_probs: dict[int, dict[int, np.ndarray]],
) -> dict[int, dict[int, np.ndarray]]:
    """
    Attribute the shots into respective subcircuit entries.

    task['subcircuit_entry_probs'][subcircuit_idx][subcircuit_entry_idx] = prob

    Args:
        - subcircuit_entries (dict): dictionary containing information about each of the
            subcircuit instances
        - subcircuit_instance_probs (dict): the probability vectors from each of the subcircuit
            instances, as output by the _run_subcircuits function
    Returns:
        - (dict): a dictionary containing the probability results to each of the appropriate subcircuits
    Raises:
        - ValueError: if each of the kronecker terms are not of size two or if there are no subcircuit
            probs provided
    """
    subcircuit_entry_probs: dict[int, dict[int, np.ndarray]] = {}
    for subcircuit_idx in subcircuit_entries:
        subcircuit_entry_probs[subcircuit_idx] = {}
        for label in subcircuit_entries[subcircuit_idx]:
            subcircuit_entry_idx, kronecker_term = subcircuit_entries[subcircuit_idx][
                label
            ]
            subcircuit_entry_prob: np.ndarray | None = None
            for coefficient, subcircuit_instance_idx in kronecker_term:
                if subcircuit_entry_prob is None:
                    subcircuit_entry_prob = (
                        coefficient
                        * subcircuit_instance_probs[subcircuit_idx][
                            subcircuit_instance_idx
                        ]
                    )
                else:
                    subcircuit_entry_prob += (
                        coefficient
                        * subcircuit_instance_probs[subcircuit_idx][
                            subcircuit_instance_idx
                        ]
                    )

            if subcircuit_entry_prob is None:
                raise ValueError(
                    "Something unexpected happened during shot attribution."
                )
            subcircuit_entry_probs[subcircuit_idx][
                subcircuit_entry_idx
            ] = subcircuit_entry_prob

    return subcircuit_entry_probs


@no_type_check
@deprecate_func(
    since="0.2.0",
    additional_msg=(
        "The circuit_knitting_toolbox.circuit_cutting.wire_cutting package is"
        " deprecated and will be removed no sooner than 1 month after its release. "
        "Users should import from circuit_knitting_toolbox.circuit_cutting.cutqc instead."
    ),
)
def find_wire_cuts(
    circuit: QuantumCircuit,
    max_subcircuit_width: int,
    max_cuts: int | None,
    num_subcircuits: Sequence[int] | None,
    max_subcircuit_cuts: int | None,
    max_subcircuit_size: int | None,
    verbose: bool,
) -> dict[str, Any]:
    """
    Find optimal cuts for the wires.

    Will print if the model cannot find a solution at all, and will print whether
    the found solution is optimal or not.

    Args:
        - circuit (QuantumCircuit): original quantum circuit to be cut into subcircuits
        - max_subcircuit_width (int): max number of qubits in each subcircuit
        - max_cuts (int, optional): max total number of cuts allowed
        - num_subcircuits (list, optional): list of number of subcircuits to try
        - max_subcircuit_cuts (int, optional): max number of cuts for a subcircuit
        - max_subcircuit_size (int, optional): the maximum number of two qubit gates in each
            subcircuit
        - verbose (bool): whether to print information about the cut finding or not
    Returns:
        - (dict): the solution found for the cuts
    """
    stripped_circ = _circuit_stripping(circuit=circuit)
    n_vertices, edges, vertex_ids, id_vertices = _read_circuit(circuit=stripped_circ)
    num_qubits = circuit.num_qubits
    cut_solution = {}
    min_cost = float("inf")

    best_mip_model = None
    for num_subcircuit in num_subcircuits:
        if (
            num_subcircuit * max_subcircuit_width - (num_subcircuit - 1) < num_qubits
            or num_subcircuit > num_qubits
            or max_cuts + 1 < num_subcircuit
        ):
            if verbose:
                print("%d subcircuits : IMPOSSIBLE" % (num_subcircuit))
            continue
        kwargs = dict(
            n_vertices=n_vertices,
            edges=edges,
            vertex_ids=vertex_ids,
            id_vertices=id_vertices,
            num_subcircuit=num_subcircuit,
            max_subcircuit_width=max_subcircuit_width,
            max_subcircuit_cuts=max_subcircuit_cuts,
            max_subcircuit_size=max_subcircuit_size,
            num_qubits=num_qubits,
            max_cuts=max_cuts,
        )

        from .mip_model import MIPModel

        mip_model = MIPModel(**kwargs)
        feasible = mip_model.solve(min_postprocessing_cost=min_cost)
        if not feasible:
            if verbose:
                print("%d subcircuits : NO SOLUTIONS" % (num_subcircuit))
            continue
        else:
            positions = _cuts_parser(mip_model.cut_edges, circuit)
            subcircuits, complete_path_map = _subcircuits_parser(
                subcircuit_gates=mip_model.subcircuits, circuit=circuit
            )
            O_rho_pairs = _get_pairs(complete_path_map=complete_path_map)
            counter = _get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)

            classical_cost = _cost_estimate(counter=counter)
            cost = classical_cost

            if cost < min_cost:
                min_cost = cost
                best_mip_model = mip_model
                cut_solution = {
                    "max_subcircuit_width": max_subcircuit_width,
                    "subcircuits": subcircuits,
                    "complete_path_map": complete_path_map,
                    "num_cuts": len(positions),
                    "counter": counter,
                    "classical_cost": classical_cost,
                }
    if verbose and len(cut_solution) > 0:
        print("-" * 20)
        classical_cost: float = float(cut_solution["classical_cost"])
        # We can remove typing.Dict from this cast statement when py38 is deprecated.
        # https://bugs.python.org/issue45117
        counter = cast(Dict[int, Dict[str, int]], cut_solution["counter"])
        subcircuits: Sequence[Any] = cut_solution["subcircuits"]
        num_cuts: int = cut_solution["num_cuts"]
        _print_cutter_result(
            num_subcircuit=len(subcircuits),
            num_cuts=num_cuts,
            subcircuits=subcircuits,
            counter=counter,
            classical_cost=classical_cost,
        )

        if best_mip_model is None:
            raise ValueError(
                "Something went wrong during cut finding. The best MIP model object was never instantiated."
            )
        print("Model objective value = %.2e" % (best_mip_model.objective), flush=True)  # type: ignore
        print("MIP runtime:", best_mip_model.runtime, flush=True)

        if best_mip_model.optimal:
            print("OPTIMAL, MIP gap =", best_mip_model.mip_gap, flush=True)
        else:
            print("NOT OPTIMAL, MIP gap =", best_mip_model.mip_gap, flush=True)
        print("-" * 20, flush=True)
    return cut_solution


@deprecate_func(
    since="0.2.0",
    additional_msg=(
        "The circuit_knitting_toolbox.circuit_cutting.wire_cutting package is"
        " deprecated and will be removed no sooner than 1 month after its release. "
        "Users should import from circuit_knitting_toolbox.circuit_cutting.cutqc instead."
    ),
)
def cut_circuit_wire(
    circuit: QuantumCircuit, subcircuit_vertices: Sequence[Sequence[int]], verbose: bool
) -> dict[str, Any]:
    """
    Perform the provided cuts.

    Used when cut locations are chosen manually.

    Args:
        - circuit (QuantumCircuit): original quantum circuit to be cut into subcircuits
        - subcircuit_vertices (list): the list of vertices to apply the cuts to
        - verbose (bool): whether to print the details of cutting or not
    Returns:
        - (dict): the solution calculated from the provided cuts
    """
    stripped_circ = _circuit_stripping(circuit=circuit)
    n_vertices, edges, vertex_ids, id_vertices = _read_circuit(circuit=stripped_circ)

    subcircuit_list = []
    for vertices in subcircuit_vertices:
        subcircuit = []
        for vertex in vertices:
            subcircuit.append(id_vertices[vertex])
        subcircuit_list.append(subcircuit)
    if sum([len(subcircuit) for subcircuit in subcircuit_list]) != n_vertices:
        raise ValueError("Not all gates are assigned into subcircuits")

    subcircuit_object = _subcircuits_parser(
        subcircuit_gates=subcircuit_list, circuit=circuit
    )
    if len(subcircuit_object) != 2:
        raise ValueError("subcircuit_object should contain exactly two elements.")
    subcircuits = subcircuit_object[0]
    complete_path_map = subcircuit_object[-1]

    O_rho_pairs = _get_pairs(complete_path_map=complete_path_map)
    counter = _get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)
    classical_cost = _cost_estimate(counter=counter)
    max_subcircuit_width = max([subcirc.width() for subcirc in subcircuits])  # type: ignore

    cut_solution = {
        "max_subcircuit_width": max_subcircuit_width,
        "subcircuits": subcircuits,
        "complete_path_map": complete_path_map,
        "num_cuts": len(O_rho_pairs),
        "counter": counter,
        "classical_cost": classical_cost,
    }

    if verbose:
        print("-" * 20)
        _print_cutter_result(
            num_subcircuit=len(cut_solution["subcircuits"]),
            num_cuts=cut_solution["num_cuts"],
            subcircuits=cut_solution["subcircuits"],
            counter=cut_solution["counter"],
            classical_cost=cut_solution["classical_cost"],
        )
        print("-" * 20)
    return cut_solution


def _print_cutter_result(
    num_subcircuit: int,
    num_cuts: int,
    subcircuits: Sequence[QuantumCircuit],
    counter: dict[int, dict[str, int]],
    classical_cost: float,
) -> None:
    """
    Pretty print the results.

    Args:
        - num_subciruit (int): the number of subcircuits
        - num_cuts (int): the number of cuts
        - subcircuits (list): the list of subcircuits
        - counter (dict): the dictionary containing all meta information regarding
            each of the subcircuits
        - classical_cost (float): the estimated processing cost
    Returns:
        - None
    """
    for subcircuit_idx in range(num_subcircuit):
        print("subcircuit %d" % subcircuit_idx)
        print(
            "\u03C1 qubits = %d, O qubits = %d, width = %d, effective = %d, depth = %d, size = %d"
            % (
                counter[subcircuit_idx]["rho"],
                counter[subcircuit_idx]["O"],
                counter[subcircuit_idx]["d"],
                counter[subcircuit_idx]["effective"],
                counter[subcircuit_idx]["depth"],
                counter[subcircuit_idx]["size"],
            )
        )
        print(subcircuits[subcircuit_idx])
    print("Estimated cost = %.3e" % classical_cost, flush=True)


def _cuts_parser(
    cuts: Sequence[tuple[str]], circ: QuantumCircuit
) -> list[tuple[Qubit, int]]:
    """
    Convert cuts to wires.

    Args:
        - cuts (list): the cuts found by the model (or provided by the user)
        - circ (QuantumCircuit): the quantum circuit the cuts are from
    Returns:
        - (list): the list containing the wires that were cut and the gates
            that are affected by these cuts
    """
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        if len(position) != 2:
            raise ValueError(
                "position variable should be a length 2 sequence: {position}"
            )
        source = position[0]
        dest = position[-1]
        source_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source.split(" ")
        ]
        dest_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest.split(" ")
        ]
        qubit_cut = []
        for source_qarg in source_qargs:
            source_qubit, source_multi_Q_gate_idx = source_qarg
            for dest_qarg in dest_qargs:
                dest_qubit, dest_multi_Q_gate_idx = dest_qarg
                if (
                    source_qubit == dest_qubit
                    and dest_multi_Q_gate_idx == source_multi_Q_gate_idx + 1
                ):
                    qubit_cut.append(source_qubit)
        # if len(qubit_cut)>1:
        #     raise Exception('one cut is cutting on multiple qubits')
        for x in source.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                source_idx = int(x.split("]")[1])
        for x in dest.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                dest_idx = int(x.split("]")[1])
        multi_Q_gate_idx = max(source_idx, dest_idx)

        wire = None
        for qubit in circ.qubits:
            if circ.find_bit(qubit).registers[0][0].name == qubit_cut[0].split("[")[
                0
            ] and circ.find_bit(qubit).index == int(
                qubit_cut[0].split("[")[1].split("]")[0]
            ):
                wire = qubit
        tmp = 0
        all_Q_gate_idx = None
        for gate_idx, gate in enumerate(
            list(dag.nodes_on_wire(wire=wire, only_ops=True))
        ):
            if len(gate.qargs) > 1:
                tmp += 1
                if tmp == multi_Q_gate_idx:
                    all_Q_gate_idx = gate_idx
        if (wire is None) or (all_Q_gate_idx is None):
            raise ValueError("Something unexpected happened while parsing cuts.")
        positions.append((wire, all_Q_gate_idx))
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions


def _subcircuits_parser(
    subcircuit_gates: list[list[str]], circuit: QuantumCircuit
) -> tuple[Sequence[QuantumCircuit], dict[Qubit, list[dict[str, int | Qubit]]]]:
    """
    Convert the subcircuit gates into quantum circuits and path out the DAGs to enable conversion.

    Args:
        - subcircuit_gates (list): the gates in the subcircuits
        - circuit (QuantumCircuit): the original circuit
    Returns:
        - (list): the subcircuits
        - (dict): the paths in the quantum circuit DAGs
    """
    """
    Assign the single qubit gates to the closest two-qubit gates
    """

    def calculate_distance_between_gate(gate_A, gate_B):
        if len(gate_A.split(" ")) >= len(gate_B.split(" ")):
            tmp_gate = gate_A
            gate_A = gate_B
            gate_B = tmp_gate
        distance = float("inf")
        for qarg_A in gate_A.split(" "):
            qubit_A = qarg_A.split("]")[0] + "]"
            qgate_A = int(qarg_A.split("]")[-1])
            for qarg_B in gate_B.split(" "):
                qubit_B = qarg_B.split("]")[0] + "]"
                qgate_B = int(qarg_B.split("]")[-1])
                # print('%s gate %d --> %s gate %d'%(qubit_A,qgate_A,qubit_B,qgate_B))
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        # print('Distance from %s to %s = %f'%(gate_A,gate_B,distance))
        return distance

    dag = circuit_to_dag(circuit)
    qubit_allGate_depths = {x: 0 for x in circuit.qubits}
    qubit_2qGate_depths = {x: 0 for x in circuit.qubits}
    gate_depth_encodings = {}
    # print('Before translation :',subcircuit_gates,flush=True)
    for op_node in dag.topological_op_nodes():
        gate_depth_encoding = ""
        for qarg in op_node.qargs:
            gate_depth_encoding += "%s[%d]%d " % (
                circuit.find_bit(qarg).registers[0][0].name,
                circuit.find_bit(qarg).index,
                qubit_allGate_depths[qarg],
            )
        gate_depth_encoding = gate_depth_encoding[:-1]
        gate_depth_encodings[op_node] = gate_depth_encoding
        for qarg in op_node.qargs:
            qubit_allGate_depths[qarg] += 1
        if len(op_node.qargs) == 2:
            MIP_gate_depth_encoding = ""
            for qarg in op_node.qargs:
                MIP_gate_depth_encoding += "%s[%d]%d " % (
                    circuit.find_bit(qarg).registers[0][0].name,
                    circuit.find_bit(qarg).index,
                    qubit_2qGate_depths[qarg],
                )
                qubit_2qGate_depths[qarg] += 1
            MIP_gate_depth_encoding = MIP_gate_depth_encoding[:-1]
            # print('gate_depth_encoding = %s, MIP_gate_depth_encoding = %s'%(gate_depth_encoding,MIP_gate_depth_encoding))
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break
    # print('After translation :',subcircuit_gates,flush=True)
    subcircuit_op_nodes: dict[int, list[DAGOpNode]] = {
        x: [] for x in range(len(subcircuit_gates))
    }
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    complete_path_map: dict[Qubit, list[dict[str, int | Qubit]]] = {}
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops):
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
                    if len(gate.split(" ")) == 1:
                        # Do not compare against single qubit gates
                        continue
                    else:
                        distance = min(
                            distance,
                            calculate_distance_between_gate(
                                gate_A=gate_depth_encoding, gate_B=gate
                            ),
                        )
                # print('Distance from %s to subcircuit %d = %f'%(gate_depth_encoding,subcircuit_idx,distance))
                if distance < min_distance:
                    min_distance = distance
                    nearest_subcircuit_idx = subcircuit_idx
            assert nearest_subcircuit_idx != -1
            path_element = {
                "subcircuit_idx": nearest_subcircuit_idx,
                "subcircuit_qubit": subcircuit_sizes[nearest_subcircuit_idx],
            }
            if (
                len(complete_path_map[circuit_qubit]) == 0
                or nearest_subcircuit_idx
                != complete_path_map[circuit_qubit][-1]["subcircuit_idx"]
            ):
                # print('{} op #{:d} {:s} encoding = {:s}'.format(circuit_qubit,qubit_op_idx,qubit_op.name,gate_depth_encoding),
                # 'belongs in subcircuit %d'%nearest_subcircuit_idx)
                complete_path_map[circuit_qubit].append(path_element)
                subcircuit_sizes[nearest_subcircuit_idx] += 1

            subcircuit_op_nodes[nearest_subcircuit_idx].append(qubit_op)
    for circuit_qubit in complete_path_map:
        # print(circuit_qubit,'-->')
        for path_element in complete_path_map[circuit_qubit]:
            path_element_qubit = QuantumRegister(
                size=subcircuit_sizes[path_element["subcircuit_idx"]], name="q"
            )[path_element["subcircuit_qubit"]]
            path_element["subcircuit_qubit"] = path_element_qubit
            # print(path_element)
    subcircuits = _generate_subcircuits(
        subcircuit_op_nodes=subcircuit_op_nodes,
        complete_path_map=complete_path_map,
        subcircuit_sizes=subcircuit_sizes,
        dag=dag,
    )
    return subcircuits, complete_path_map


def _generate_subcircuits(
    subcircuit_op_nodes: dict[int, list[DAGOpNode]],
    complete_path_map: dict[Qubit, list[dict[str, int | Qubit]]],
    subcircuit_sizes: Sequence[int],
    dag: DAGCircuit,
) -> Sequence[QuantumCircuit]:
    """
    Generate the subcircuits from given nodes and paths.

    Called in the subcircuit_parser function to convert the found paths and nodes
    into actual quantum circuit objects.

    Args:
        - subcircuit_op_nodes (dict): the nodes of each of the subcircuits
        - complete_path_map (dict): the complete path through the subcircuits
        - subcircuit_sizes (list): the number of qubits in each of the subcircuits
        - dag (DAGCircuit): the dag representation of the input quantum circuit
    Returns:
        - (list): the subcircuits
    """
    qubit_pointers = {x: 0 for x in complete_path_map}
    subcircuits = [QuantumCircuit(x, name="q") for x in subcircuit_sizes]
    for op_node in dag.topological_op_nodes():
        subcircuit_idx_list = list(
            filter(
                lambda x: op_node in subcircuit_op_nodes[x], subcircuit_op_nodes.keys()
            )
        )
        if len(subcircuit_idx_list) != 1:
            raise ValueError("A node cannot belong to more than one subcircuit.")
        subcircuit_idx = subcircuit_idx_list[0]
        # print('{} belongs in subcircuit {:d}'.format(op_node.qargs,subcircuit_idx))
        subcircuit_qargs = []
        for op_node_qarg in op_node.qargs:
            if (
                complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]][
                    "subcircuit_idx"
                ]
                != subcircuit_idx
            ):
                qubit_pointers[op_node_qarg] += 1
            path_element = complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]]
            assert path_element["subcircuit_idx"] == subcircuit_idx
            subcircuit_qargs.append(path_element["subcircuit_qubit"])
        # print('-->',subcircuit_qargs)

        # mypy doesn't recognize QuantumCircuit as being an Iterable, so we ignore
        subcircuits[subcircuit_idx].append(  # type: ignore
            instruction=op_node.op, qargs=subcircuit_qargs, cargs=None
        )
    return subcircuits


def _get_counter(
    subcircuits: Sequence[QuantumCircuit],
    O_rho_pairs: list[tuple[dict[str, int | Qubit], dict[str, int | Qubit]]],
) -> dict[int, dict[str, int]]:
    """
    Create information regarding each of the subcircuit parameters (qubits, width, etc.).

    Args:
        - subcircuits (list): the list of subcircuits
        - O_rho_pairs (list): the pairs for each qubit path as generated in the _get_pairs
            function
    Returns:
        - (dict): the resulting dictionary with all parameter information
    """
    counter = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        counter[subcircuit_idx] = {
            "effective": subcircuit.num_qubits,
            "rho": 0,
            "O": 0,
            "d": subcircuit.num_qubits,
            "depth": subcircuit.depth(),
            "size": subcircuit.size(),
        }
    for pair in O_rho_pairs:
        if len(pair) != 2:
            raise ValueError(f"O_rho_pairs must be length 2: {pair}")
        O_qubit = pair[0]
        rho_qubit = pair[-1]
        counter[O_qubit["subcircuit_idx"]]["effective"] -= 1
        counter[O_qubit["subcircuit_idx"]]["O"] += 1
        counter[rho_qubit["subcircuit_idx"]]["rho"] += 1
    return counter


def _cost_estimate(counter: dict[int, dict[str, int]]) -> float:
    """
    Estimate the cost of processing the subcircuits.

    Args:
        - counter (dict): dictionary containing information for each of the
            subcircuits
    Returns:
        - (float): the estimated cost for classical processing
    """
    num_cuts = sum([counter[subcircuit_idx]["rho"] for subcircuit_idx in counter])
    subcircuit_indices = list(counter.keys())
    num_effective_qubits_list = [
        counter[subcircuit_idx]["effective"] for subcircuit_idx in subcircuit_indices
    ]
    num_effective_qubits, _ = zip(
        *sorted(zip(num_effective_qubits_list, subcircuit_indices))
    )
    classical_cost = 0
    accumulated_kron_len = 2 ** num_effective_qubits[0]
    for effective in num_effective_qubits[1:]:
        accumulated_kron_len *= 2**effective
        classical_cost += accumulated_kron_len
    classical_cost *= 4**num_cuts
    return classical_cost


def _get_pairs(
    complete_path_map: dict[Qubit, list[dict[str, int | Qubit]]]
) -> list[tuple[dict[str, int | Qubit], dict[str, int | Qubit]]]:
    """
    Get all pairs through each path.

    Iterates through the path for each of the qubits and keeps track of the
    each pair of neigbors.

    Args:
        - complete_path_map (dict): the dictionary containing all path information
    Returns:
        - (list): all pairs for each of the qubit paths
    """
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path) > 1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr + 1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs


def _circuit_stripping(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Remove all single qubit and barrier type gates.

    Args:
        - circuit (QuantumCircuit): the circuit to strip
    Returns:
        - (QuantumCircuit): the stripped circuit
    """
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)


def _read_circuit(
    circuit: QuantumCircuit,
) -> tuple[int, list[tuple[int, int]], dict[str, int], dict[int, str]]:
    """
    Read the input circuit to a graph based representation for the MIP model.

    Args:
        - circuit (QuantumCircuit): a stripped circuit to be converted into a
            DAG like representation
    Returns:
        - (int): number of vertices
        - (list): edge list
        - (dict): the dictionary mapping vertices to vertex numbers
        - (dict): the dictionary mapping vertex numbers to vertex information
    """
    dag = circuit_to_dag(circuit)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    qubit_gate_counter = {}
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception("vertex does not have 2 qargs!")
        arg0, arg1 = vertex.qargs
        vertex_name = "%s[%d]%d %s[%d]%d" % (
            circuit.find_bit(arg0).registers[0][0].name,
            circuit.find_bit(arg0).index,
            qubit_gate_counter[arg0],
            circuit.find_bit(arg1).registers[0][0].name,
            circuit.find_bit(arg1).index,
            qubit_gate_counter[arg1],
        )
        qubit_gate_counter[arg0] += 1
        qubit_gate_counter[arg1] += 1
        # print(vertex.op.label,vertex_name,curr_node_id)
        if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            vertex_ids[id(vertex)] = curr_node_id
            curr_node_id += 1

    for u, v, _ in dag.edges():
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
            u_id = vertex_ids[id(u)]
            v_id = vertex_ids[id(v)]
            edges.append((u_id, v_id))

    n_vertices = dag.size()

    return n_vertices, edges, node_name_ids, id_node_names
