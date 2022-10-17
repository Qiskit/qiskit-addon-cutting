# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains functions for executing subcircuits."""
import itertools, copy
from typing import Dict, Tuple, Sequence, Optional, List, Any, Union

import numpy as np
from nptyping import NDArray

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
from qiskit.primitives import Sampler as TestSampler
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from quantum_serverless import run_qiskit_remote, get

from circuit_knitting_toolbox.utils.conversion import dict_to_array


def run_subcircuit_instances(
    subcircuits: Sequence[QuantumCircuit],
    subcircuit_instances: Dict[int, Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int]],
    service_args: Optional[Dict[str, Any]] = None,
    backend_names: Optional[Sequence[str]] = None,
    options: Optional[Union[Dict, Options]] = None,
) -> Dict[int, Dict[int, NDArray]]:
    """
    Execute all provided subcircuits.

    Using the backend(s) provided, this executes all the subcircuits to generate the
    resultant probability vectors.
    subcircuit_instance_probs[subcircuit_idx][subcircuit_instance_idx] = measured probability

    Args:
        - subcircuits (Sequence[QuantumCircuit]): the list of subcircuits to execute
        - subcircuit_instances (Dict): dictionary containing information about each of the
            subcircuit instances
        - service_args (Dict): the arguments for the runtime service
        - backend_names (Sequence[str]): the backend(s) used to execute the subcircuits
        - options (Options): options for the runtime execution of subcircuits

    Returns:
        - (Dict): the probability vectors from each of the subcircuit instances
    """
    if service_args:
        if backend_names:
            backend_names_repeated: List[Union[str, None]] = [
                backend_names[i % len(backend_names)] for i, _ in enumerate(subcircuits)
            ]
        else:
            backend_names_repeated = ["ibmq_qasm_simulator"] * len(subcircuits)

    else:
        backend_names_repeated = [None] * len(subcircuits)

    subcircuit_instance_probs: Dict[int, Dict[int, NDArray]] = {}
    subcircuit_instance_probs_futures = [
        _run_subcircuit_batch(
            subcircuit_instances[subcircuit_idx],
            subcircuit,
            service_args=service_args,
            backend_name=backend_names_repeated[subcircuit_idx],
            options=options,
        )
        for subcircuit_idx, subcircuit in enumerate(subcircuits)
    ]

    for i, partition_batch_futures in enumerate(subcircuit_instance_probs_futures):
        subcircuit_instance_probs[i] = get(partition_batch_futures)

    return subcircuit_instance_probs


def mutate_measurement_basis(meas: Tuple[str, ...]) -> List[Tuple[Any, ...]]:
    """
    Change of basis for all identity measurements.

    For every identity measurement, it is split into an I and Z measurement.
    I and Z measurement basis correspond to the same logical circuit.

    Args:
        - meas (tuple): the current measurement bases

    Returns:
        - (tuple): the update measurement bases
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas_out = list(itertools.product(*mutated_meas))

        return mutated_meas_out


def modify_subcircuit_instance(
    subcircuit: QuantumCircuit, init: Tuple[str, ...], meas: Tuple[str, ...]
) -> QuantumCircuit:
    """
    Modify the initialization and measurement bases for a given subcircuit.

    Args:
        - subcircuit (QuantumCircuit): the subcircuit to be modified
        - init (tuple): the current initializations
        - meas (tuple): the current measement bases

    Returns:
        - (QuantumCircuit): the updated circuit, modified so the initialziation
            and measurement operators are all in the standard computational basis

    Raises:
        - Exeption: if one of the init's or meas's are not an acceptable string
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)

    return subcircuit_instance_circuit


def run_subcircuits(
    subcircuits: Sequence[QuantumCircuit],
    service_args: Optional[Dict[str, Any]] = None,
    backend_name: Optional[str] = None,
    options: Optional[Union[Dict, Options]] = None,
) -> List[NDArray]:
    """
    Execute the subcircuit(s).

    Args:
        - subcircuit (QuantumCircuit): the subcircuits to be executed
        - service_args (Dict): the arguments for the runtime service
        - backend_name (str): the backend used to execute the subcircuits
        - options (Options): options for the runtime execution of subcircuits

    Returns:
        - (NDArray): the probability distributions
    """
    for subcircuit in subcircuits:
        if subcircuit.num_clbits == 0:
            subcircuit.measure_all()

    service = QiskitRuntimeService(**service_args) if service_args is not None else None
    job_id = None
    if service is not None:
        session = Session(service=service, backend=backend_name)
        sampler = Sampler(session=session, options=options)
    else:
        sampler = TestSampler()

    quasi_dists = sampler.run(circuits=subcircuits).result().quasi_dists

    all_probabilities_out = []
    for i, qd in enumerate(quasi_dists):
        probabilities = qd.nearest_probability_distribution()
        probabilities_out = np.zeros(2 ** subcircuits[i].num_qubits, dtype=float)

        for state in probabilities:
            probabilities_out[state] = probabilities[state]
        all_probabilities_out.append(probabilities_out)

    return all_probabilities_out


def measure_prob(unmeasured_prob: NDArray, meas: Tuple[Any, ...]) -> NDArray:
    """
    Compute the effective probability distribution from the subcircuit distribution.

    Args:
        - unmeasured_prob (Sequence[float]): the outputs of the subcircuit execution
        - meas (tuple): the measurement bases

    Returns:
        - (NDArray): the updated measured probability distribution
    """
    if meas.count("comp") == len(meas):
        return np.array(unmeasured_prob)
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            # TODO: Add states merging here. Change effective_state to merged_bin
            measured_prob[effective_state] += sigma * p

        return measured_prob


def measure_state(full_state: int, meas: Tuple[Any, ...]) -> Tuple[int, int]:
    """
    Compute the corresponding effective_state for the given full_state.

    Measured in basis `meas`. Returns sigma (int), effective_state (int) where sigma = +-1

    Args:
        - full_state (int): the current state (in decimal form)
        - meas (tuple): the measurement bases

    Returns:
        - (tuple): sigma (defined by the parity of non computational basis 1 measurements) and
            the effective state (defined by the measurements in the computational basis)
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0

    return sigma, effective_state


@run_qiskit_remote()
def _run_subcircuit_batch(
    subcircuit_instance: Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int],
    subcircuit: QuantumCircuit,
    service_args: Optional[Dict[str, Any]] = None,
    backend_name: Optional[str] = None,
    options: Optional[Union[Dict, Options]] = None,
):
    """
    Execute a circuit using qiskit runtime and quantum serverless.

    Args:
        - subcircuit_instances (Dict): dictionary containing information about each of the
            subcircuit instances
        - subcircuit (QuantumCircuit): the subcircuit to execute
        - service_args (Dict): the arguments for the runtime service
        - backend_name (str): the backends used to execute the subcircuit
        - options (Options): options for the runtime execution of subcircuit

    Returns:
        - (dict): the measurement probabilities for the subcircuit batch, as calculated from the
            runtime execution
    """
    subcircuit_instance_probs = {}
    circuits_to_run = []

    # For each circuit associated with a given subcircuit
    for init_meas in subcircuit_instance:
        subcircuit_instance_idx = subcircuit_instance[init_meas]

        # Collect all of the circuits we need to evaluate, ensuring we don't have duplicates
        if subcircuit_instance_idx not in subcircuit_instance_probs:
            modified_subcircuit_instance = modify_subcircuit_instance(
                subcircuit=subcircuit,
                init=init_meas[0],
                meas=tuple(init_meas[1]),
            )
            circuits_to_run.append(modified_subcircuit_instance)
            mutated_meas = mutate_measurement_basis(meas=tuple(init_meas[1]))
            for meas in mutated_meas:
                mutated_subcircuit_instance_idx = subcircuit_instance[
                    (init_meas[0], meas)
                ]
                # Set a placeholder in the probability dict to prevent duplicate circuits to the Sampler
                subcircuit_instance_probs[mutated_subcircuit_instance_idx] = np.array(
                    [0.0]
                )

    # Run all of our circuits in one batch
    subcircuit_inst_probs = run_subcircuits(
        circuits_to_run,
        service_args=service_args,
        backend_name=backend_name,
        options=options,
    )

    # Calculate the measured probabilities
    unique_subcircuit_check = {}
    i = 0
    for init_meas in subcircuit_instance:
        subcircuit_instance_idx = subcircuit_instance[init_meas]
        if subcircuit_instance_idx not in unique_subcircuit_check:
            subcircuit_inst_prob = subcircuit_inst_probs[i]
            i = i + 1
            mutated_meas = mutate_measurement_basis(meas=tuple(init_meas[1]))
            for meas in mutated_meas:
                measured_prob = measure_prob(
                    unmeasured_prob=subcircuit_inst_prob, meas=meas
                )
                mutated_subcircuit_instance_idx = subcircuit_instance[
                    (init_meas[0], meas)
                ]
                subcircuit_instance_probs[
                    mutated_subcircuit_instance_idx
                ] = measured_prob
                unique_subcircuit_check[mutated_subcircuit_instance_idx] = True

    return subcircuit_instance_probs
