# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing all cutting post processing functionality."""

from __future__ import annotations

import itertools
import multiprocessing as mp
from typing import Sequence, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.utils.deprecation import deprecate_func
from qiskit.circuit import Qubit


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def get_cut_qubit_pairs(
    complete_path_map: dict[Qubit, Sequence[dict[str, int | Qubit]]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Get O-rho cut qubit pairs.

    Iterates through the path for each of the qubits and keeps track of the
    each pair of neigbors.

    Args:
        complete_path_map: The dictionary containing all path information

    Returns:
        All pairs for each of the qubit paths
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


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def get_label(label_idx: int, num_cuts: int) -> Sequence[str]:
    """
    Get the basis label for each cut point.

    Args:
        label_idx: The label to be applied for the current basis
        num_cuts: The number of cuts

    Returns:
        The list of the labels
    """
    assert label_idx < 4**num_cuts
    basis = ["I", "X", "Y", "Z"]
    label = []
    for position in range(num_cuts):
        digit = label_idx % 4
        label.append(basis[digit])
        label_idx = label_idx // 4
    return label


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def attribute_label(
    label: Sequence[str],
    O_rho_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    num_subcircuits: int,
) -> dict[int, dict[str, str]]:
    """
    Get the label attributed to each subcircuit.

    subcircuit_label[subcircuit_idx]['init'] = str of init for the cut qubits.
    subcircuit_label[subcircuit_idx]['meas'] = str of meas for the cut qubits.

    Args:
        label: The labels of bases for each cut point, as generated by the
            get_label function
        O_rho_paris: The pairs of cut qubits, as generated by the get_cut_qubit_pairs
            function
        num_subcircuits: The number of subcircuits

    Returns:
        The dictionary containing the labels applied to each of the subcircuits
    """
    subcircuit_label = {
        subcircuit_idx: {"init": "", "meas": ""}
        for subcircuit_idx in range(num_subcircuits)
    }
    for c, pair in zip(label, O_rho_pairs):
        if len(pair) != 2:
            raise ValueError(
                "O_rho_pairs should only contain sequences of length 2: {pair}"
            )
        O_qubit = pair[0]
        rho_qubit: dict[str, int | Qubit] = pair[-1]
        subcircuit_label[O_qubit["subcircuit_idx"]]["meas"] += c
        subcircuit_label[rho_qubit["subcircuit_idx"]]["init"] += c
    return subcircuit_label


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def fill_label(
    subcircuit_idx: int,
    subcircuit: QuantumCircuit,
    subcircuit_label: dict[str, str],
    O_rho_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[str], list[str]]:
    """
    Given a subcircuit, its label and O-rho cut qubit pairs fill the full init, meas strings.

    Args:
        subcircuit_idx: The subcircuit index
        subcircuit: The subcircuit
        subcircuit_label: The dictionary containing the applied labels for the bases
        O_rho_pairs: The list of cut pair qubits

    Returns:
        A tuple containing the initial input bases and the measurements required
    """
    subcircuit_init_label_counter = 0
    subcircuit_meas_label_counter = 0
    init = ["zero" for q in range(subcircuit.num_qubits)]
    meas = ["comp" for q in range(subcircuit.num_qubits)]
    for pair in O_rho_pairs:
        if len(pair) != 2:
            raise ValueError(f"O_rho_pairs should be length 2: {O_rho_pairs}")
        O_qubit = pair[0]
        rho_qubit: dict[str, int | Qubit] = pair[-1]
        if O_qubit["subcircuit_idx"] == subcircuit_idx:
            qubit_idx = subcircuit.qubits.index(O_qubit["subcircuit_qubit"])
            meas[qubit_idx] = subcircuit_label["meas"][subcircuit_meas_label_counter]
            subcircuit_meas_label_counter += 1
        if rho_qubit["subcircuit_idx"] == subcircuit_idx:
            qubit_idx = subcircuit.qubits.index(rho_qubit["subcircuit_qubit"])
            init[qubit_idx] = subcircuit_label["init"][subcircuit_init_label_counter]
            subcircuit_init_label_counter += 1
    return init, meas


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def get_init_meas(
    init_label: Sequence[str], meas_label: Sequence[str]
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """
    Generate the initial measurements.

    Args:
        init_label: The list of initial bases
        meas: The list of measurement bases

    Returns:
        All subcircuit initial measurements required
    """
    init_combinations = []
    for x in init_label:
        if x == "zero":
            init_combinations.append(["zero"])
        elif x == "I":
            init_combinations.append(["+zero", "+one"])
        elif x == "X":
            init_combinations.append(["2plus", "-zero", "-one"])
        elif x == "Y":
            init_combinations.append(["2plusI", "-zero", "-one"])
        elif x == "Z":
            init_combinations.append(["+zero", "-one"])
        else:
            raise Exception("Illegal initilization symbol :", x)
    init_combinations = list(itertools.product(*init_combinations))  # type: ignore

    meas_combinations = []
    for x in meas_label:
        meas_combinations.append([x])
    meas_combinations = list(itertools.product(*meas_combinations))  # type: ignore

    subcircuit_init_meas = []
    for init in init_combinations:
        for meas in meas_combinations:
            subcircuit_init_meas.append((tuple(init), tuple(meas)))
    return subcircuit_init_meas


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def convert_to_physical_init(init: list[str]) -> tuple[int, tuple[str, ...]]:
    """
    Convert the initial measurements to the physical representations.

    Args:
        init: The initial measurements, e.g. ["zero", "2plus"]

    Returns:
        A tuple containing the coefficient of the physical measurements
        and the physical measurement bases
    """
    coefficient = 1
    for idx, x in enumerate(init):
        if x == "zero":
            continue
        elif x == "+zero":
            init[idx] = "zero"
        elif x == "+one":
            init[idx] = "one"
        elif x == "2plus":
            init[idx] = "plus"
            coefficient *= 2
        elif x == "-zero":
            init[idx] = "zero"
            coefficient *= -1
        elif x == "-one":
            init[idx] = "one"
            coefficient *= -1
        elif x == "2plusI":
            init[idx] = "plusI"
            coefficient *= 2
        else:
            raise Exception("Illegal initilization symbol :", x)
    return coefficient, tuple(init)


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def generate_summation_terms(
    subcircuits: Sequence[QuantumCircuit],
    complete_path_map: dict[Qubit, Sequence[dict[str, int | Qubit]]],
    num_cuts: int,
) -> tuple[
    list[dict[int, int]],
    dict[int, dict[tuple[str, str], tuple[int, Sequence[tuple[int, int]]]]],
    dict[int, dict[tuple[tuple[str, ...], tuple[Any, ...]], int]],
]:
    """
    Generate all summation terms for the final reconstructions.

    Final CutQC reconstruction result = Sum(summation_terms).

    summation_terms (list): [summation_term_0, summation_term_1, ...] --> 4^#cuts elements.

    | summation_term[subcircuit_idx] = subcircuit_entry_idx.
    | E.g. summation_term = {0:0,1:13,2:7} = Kron(subcircuit_0_entry_0, subcircuit_1_entry_13, subcircuit_2_entry_7).

    | subcircuit_entries[subcircuit_idx][init_label,meas_label] = subcircuit_entry_idx, kronecker_term.
    | kronecker_term (list): (coefficient, subcircuit_instance_idx).
    | Add coefficient*subcircuit_instance to subcircuit_entry.

    subcircuit_instances[subcircuit_idx][init,meas] = subcircuit_instance_idx.

    Args:
        subcircuits: The list of subcircuits
        complete_path_map: The paths of all the qubits through the circuit DAGs
        num_cuts: The number of cuts

    Returns:
        A tuple containing a summation term dict, a subcircuit entry dict,
        and a subcircuit instances dict
    """
    summation_terms = []
    subcircuit_entries: dict[
        int, dict[tuple[str, str], tuple[int, Sequence[tuple[int, int]]]]
    ] = {subcircuit_idx: {} for subcircuit_idx in range(len(subcircuits))}
    subcircuit_instances: dict[
        int, dict[tuple[tuple[str, ...], tuple[Any, ...]], int]
    ] = {subcircuit_idx: {} for subcircuit_idx in range(len(subcircuits))}
    O_rho_pairs = get_cut_qubit_pairs(complete_path_map=complete_path_map)
    for summation_term_idx in range(4**num_cuts):
        label = get_label(label_idx=summation_term_idx, num_cuts=num_cuts)
        # print('%d/%d summation term:'%(summation_term_idx+1,4**num_cuts),label)
        summation_term = {}
        subcircuit_labels = attribute_label(
            label=label, O_rho_pairs=O_rho_pairs, num_subcircuits=len(subcircuits)
        )
        for subcircuit_idx in range(len(subcircuits)):
            # print('subcircuit %d label :'%subcircuit_idx,subcircuit_labels[subcircuit_idx])
            subcircuit_entry_key = (
                subcircuit_labels[subcircuit_idx]["init"],
                subcircuit_labels[subcircuit_idx]["meas"],
            )
            if subcircuit_entry_key in subcircuit_entries[subcircuit_idx]:
                subcircuit_entry_idx, kronecker_term = subcircuit_entries[
                    subcircuit_idx
                ][subcircuit_entry_key]
                # print('Already in, subcircuit_entry {:d} : {}'.format(subcircuit_entry_idx,kronecker_term))
            else:
                subcircuit_full_label = fill_label(
                    subcircuit_idx=subcircuit_idx,
                    subcircuit=subcircuits[subcircuit_idx],
                    subcircuit_label=subcircuit_labels[subcircuit_idx],
                    O_rho_pairs=O_rho_pairs,
                )
                # print('Full label :',subcircuit_full_label)
                if len(subcircuit_full_label) != 2:
                    raise ValueError(
                        f"subcircuit_full_label variable should be a length-2 tuple: {subcircuit_full_label}"
                    )
                subcircuit_init_meas = get_init_meas(
                    init_label=subcircuit_full_label[0],
                    meas_label=subcircuit_full_label[-1],
                )
                kronecker_term = []
                for init_meas in subcircuit_init_meas:
                    if len(init_meas) != 2:
                        raise ValueError(
                            f"init_meas variable should be a length-2 tuple: {init_meas}"
                        )
                    init: tuple[str, ...] = init_meas[0]
                    meas: tuple[str, ...] = init_meas[-1]
                    coefficient, init = convert_to_physical_init(
                        init=list(init_meas[0])
                    )
                    if (init, meas) in subcircuit_instances[subcircuit_idx]:
                        subcircuit_instance_idx = subcircuit_instances[subcircuit_idx][
                            (init, meas)
                        ]
                    else:
                        subcircuit_instance_idx = len(
                            subcircuit_instances[subcircuit_idx]
                        )
                        subcircuit_instances[subcircuit_idx][
                            (init, meas)
                        ] = subcircuit_instance_idx
                    kronecker_term.append((coefficient, subcircuit_instance_idx))
                subcircuit_entry_idx = len(subcircuit_entries[subcircuit_idx])
                subcircuit_entries[subcircuit_idx][subcircuit_entry_key] = (
                    subcircuit_entry_idx,
                    kronecker_term,
                )
                # print('New subcircuit_entry, {:d} : {}'.format(subcircuit_entry_idx,kronecker_term))
            summation_term[subcircuit_idx] = subcircuit_entry_idx
        summation_terms.append(summation_term)
        # print('summation_term =',summation_term,'\n')
    return summation_terms, subcircuit_entries, subcircuit_instances


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def naive_compute(
    subcircuit_order: Sequence[int],
    summation_terms: Sequence[dict[int, int]],
    subcircuit_entry_probs: dict[int, dict[int, np.ndarray]],
) -> tuple[np.ndarray | None, dict[str, int]]:
    """
    Reconstruct the full probability distribution from the subcircuits.

    This function is called within the build function, meant to be used
    in a multipooling manner.

    Args:
        subcircuit_order: The order of the subcircuit inputs
        summation_terms: The summation terms, as generated
            from generate_summation_terms
        subcircuit_entry_probs: The input probabilities from each of
            the subcircuit executions

    Returns:
        A tuple containing the reconstructed probability distribution
        and the approximate computational overhead of the function
    """
    reconstructed_prob = None
    overhead = {"additions": 0, "multiplications": 0}
    for summation_term in summation_terms:
        summation_term_prob = None
        for subcircuit_idx in subcircuit_order:
            subcircuit_entry_idx = summation_term[subcircuit_idx]
            subcircuit_entry_prob = subcircuit_entry_probs[subcircuit_idx][
                subcircuit_entry_idx
            ]
            if summation_term_prob is None:
                summation_term_prob = subcircuit_entry_prob
            else:
                summation_term_prob = np.kron(
                    summation_term_prob, subcircuit_entry_prob
                )
                overhead["multiplications"] += len(summation_term_prob)
        if reconstructed_prob is None:
            reconstructed_prob = summation_term_prob
        else:
            reconstructed_prob += summation_term_prob
            overhead["additions"] += len(reconstructed_prob)
    return reconstructed_prob, overhead


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def build(
    summation_terms: Sequence[dict[int, int]],
    subcircuit_entry_probs: dict[int, dict[int, np.ndarray]],
    num_cuts: int,
    num_threads: int,
) -> tuple[np.ndarray, list[int], dict[str, int]]:
    """
    Reconstruct the full probability distribution from the subcircuits.

    Args:
        summation_terms: The summation terms used to generate the full
            vector, as generated in generate_summation_terms
        subcircuit_entry_probs: The probabilities vectors from the
            subcircuit executions
        num_cuts: The number of cuts
        num_threads: The number of threads to use for multithreading

    Returns:
        A tuple containing the reconstructed probability distribution of the full
        circuit, the ordering of the distribution, and the computational
        post-processing overhead
    """
    smart_order = sorted(
        list(subcircuit_entry_probs.keys()),
        key=lambda subcircuit_idx: len(subcircuit_entry_probs[subcircuit_idx][0]),
    )
    args = []
    for i in range(num_threads * 5):
        segment_summation_terms = find_process_jobs(
            jobs=summation_terms, rank=i, num_workers=num_threads * 5
        )
        if len(segment_summation_terms) == 0:
            break
        arg = (smart_order, segment_summation_terms, subcircuit_entry_probs)
        args.append(arg)
    # Why "spawn"?  See https://pythonspeed.com/articles/python-multiprocessing/
    with mp.get_context("spawn").Pool(num_threads) as pool:
        results = pool.starmap(naive_compute, args)
    overhead = {"additions": 0, "multiplications": 0}
    reconstructed_prob = None
    for result in results:
        thread_reconstructed_prob, thread_overhead = result
        if reconstructed_prob is None:
            reconstructed_prob = thread_reconstructed_prob
        else:
            reconstructed_prob += thread_reconstructed_prob
        overhead["additions"] += thread_overhead["additions"]
        overhead["multiplications"] += thread_overhead["multiplications"]
    reconstructed_prob /= 2**num_cuts

    if reconstructed_prob is None:
        raise ValueError("Something went wrong during the build.")

    return reconstructed_prob, smart_order, overhead


@deprecate_func(
    removal_timeline="No sooner than CKT v0.8.0",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the circuit_knitting.cutting package. ",
)
def find_process_jobs(
    jobs: Sequence[Any], rank: int, num_workers: int
) -> Sequence[Any]:
    """
    Split up the total jobs into subjobs to be multithreaded.

    Args:
        jobs: All summation terms to be used
        rank: The input thread
        num_workers: 5 * the number of threads

    Returns:
        The sublist of jobs to be executed on the given thread
    """
    count = int(len(jobs) / num_workers)
    remainder = len(jobs) % num_workers
    if rank < remainder:
        jobs_start = rank * (count + 1)
        jobs_stop = jobs_start + count + 1
    else:
        jobs_start = rank * count + remainder
        jobs_stop = jobs_start + (count - 1) + 1
    process_jobs = list(jobs[jobs_start:jobs_stop])
    return process_jobs
