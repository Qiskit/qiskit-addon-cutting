import itertools, copy, random, psutil
from time import time
from typing import Callable, Dict, Tuple, Sequence, Optional, List, Iterable, Any, cast

import numpy as np
from nptyping import NDArray

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
from qiskit.primitives import Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.sampler import SamplerResultDecoder
from quantum_serverless import run_qiskit_remote, get

from circuit_knitting_toolbox.utils.conversion import dict_to_array


def run_subcircuit_instances(
    subcircuits: Sequence[QuantumCircuit],
    subcircuit_instances: Dict[int, Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int]],
    service_args: Optional[Dict[str, Any]] = None,
    backend_names: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[int, NDArray]]:
    """
    subcircuit_instance_probs[subcircuit_idx][subcircuit_instance_idx] = measured probability
    """
    if service_args:
        if backend_names:
            backend_names_repeated = [
                backend_names[i % len(backend_names)] for i, _ in enumerate(subcircuits)
            ]
        else:
            ValueError("A service was passed but the backend_names argument is None.")

    else:
        backend_names_repeated = [None] * len(subcircuits)

    subcircuit_instance_probs: Dict[int, Dict[int, NDArray]] = {}
    subcircuit_instance_probs_futures = [
        _run_subcircuit_batch(
            subcircuit_instances[subcircuit_idx],
            subcircuit,
            service_args,
            backend_names_repeated[subcircuit_idx],
        )
        for subcircuit_idx, subcircuit in enumerate(subcircuits)
    ]

    for i, partition_batch_futures in enumerate(subcircuit_instance_probs_futures):
        subcircuit_instance_probs[i] = get(partition_batch_futures)

    return subcircuit_instance_probs


def mutate_measurement_basis(meas: Tuple[str, ...]) -> List[Tuple[Any, ...]]:
    """
    I and Z measurement basis correspond to the same logical circuit
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
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
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


def run_subcircuit(
    subcircuit: QuantumCircuit,
    service_args: Optional[Dict[str, Any]] = None,
    backend_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> NDArray:
    """
    Simulate a subcircuit
    """
    if subcircuit.num_clbits == 0:
        subcircuit.measure_all()

    service = QiskitRuntimeService(**service_args) if service_args is not None else None
    job_id = None
    if service is not None:
        shots = 4096
        inputs = {
            "circuits": [subcircuit],
            "circuit_indices": [0],
            "shots": shots,
            "transpilation_options": {"optimization_level": 3},
            "resilience_settings": {"level": 2},
        }
        options = {"backend": backend_name}

        start_session = False
        if session_id is None and backend_name is not "ibmq_qasm_simulator":
            start_session = True

        job = service.run(
            program_id="sampler",
            inputs=inputs,
            options=options,
            result_decoder=SamplerResultDecoder,
            session_id=session_id,
            start_session=start_session,
        )
        job_id = job.job_id
        quasi_dists = job.result().quasi_dists[0]
    else:
        sampler = Sampler()
        quasi_dists = sampler.run(circuits=[subcircuit]).result().quasi_dists[0]

    probabilities = quasi_dists.nearest_probability_distribution()
    probabilities_out = np.zeros(2**subcircuit.num_qubits, dtype=float)

    for state in probabilities:
        probabilities_out[state] = probabilities[state]

    return probabilities_out, job_id


def measure_prob(unmeasured_prob: NDArray, meas: Tuple[Any, ...]) -> NDArray:
    if meas.count("comp") == len(meas):
        return np.array(unmeasured_prob)
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            # TODO: Add states merging here. Change effective_state to merged_bin
            measured_prob[effective_state] += sigma * p

        return measured_prob


def measure_state(full_state: int, meas: Tuple[Any, ...]) -> Tuple[int, int]:
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
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
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))

    return sigma, effective_state


@run_qiskit_remote()
def _run_subcircuit_batch(
    subcircuit_instance: Dict[Tuple[Tuple[str, ...], Tuple[Any, ...]], int],
    subcircuit: QuantumCircuit,
    service_args: Optional[Dict[str, Any]] = None,
    backend_name: Optional[str] = None,
):
    subcircuit_instance_probs = {}
    session_id = None
    for init_meas in subcircuit_instance:
        subcircuit_instance_idx = subcircuit_instance[init_meas]
        if subcircuit_instance_idx not in subcircuit_instance_probs:
            modified_subcircuit_instance = modify_subcircuit_instance(
                subcircuit=subcircuit,
                init=init_meas[0],
                meas=tuple(init_meas[1]),
            )
            subcircuit_inst_prob, job_id = run_subcircuit(
                modified_subcircuit_instance, service_args, backend_name, session_id
            )
            if session_id is None:
                session_id = job_id
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

    return subcircuit_instance_probs
