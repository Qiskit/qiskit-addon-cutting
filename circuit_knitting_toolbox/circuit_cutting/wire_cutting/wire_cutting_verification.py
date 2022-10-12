"""File that contains the function to verify the results of the cut circuits."""
import psutil, copy
from typing import Sequence, Dict, Union, Tuple, List

import numpy as np
from nptyping import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

from circuit_knitting_toolbox.utils.conversion import quasi_to_real
from circuit_knitting_toolbox.utils.metrics import (
    chi2_distance,
    MSE,
    MAPE,
    cross_entropy,
    HOP,
)


def verify(
    full_circuit: QuantumCircuit,
    reconstructed_output: NDArray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare the reconstructed probabilities to the ground truth.

    Executes the original circuit, then measures the distributional differences between this exact
    result (ground truth) and the reconstructed result from the subcircuits.
    Provides a variety of metrics to evaluate the differences in the distributions.

    Args:
        - full_circuit (QuantumCircuit): the original quantum circuit that was cut
        - reconstructed_output (NDArray): the reconstructed probability distribution from the
            execution of the subcircuits

    Returns:
        - (dict): a dictionary containing a variety of distributional difference metrics for the
            ground truth and reconstructed distributions
    """
    ground_truth = _evaluate_circuit(circuit=full_circuit)
    metrics = {}
    for quasi_conversion_mode in ["nearest", "naive"]:
        real_probability = quasi_to_real(
            quasiprobability=reconstructed_output, mode=quasi_conversion_mode
        )

        chi2 = chi2_distance(target=ground_truth, obs=real_probability)
        mse = MSE(target=ground_truth, obs=real_probability)
        mape = MAPE(target=ground_truth, obs=real_probability)
        ce = cross_entropy(target=ground_truth, obs=real_probability)
        hop = HOP(target=ground_truth, obs=real_probability)
        metrics[quasi_conversion_mode] = {
            "chi2": chi2,
            "Mean Squared Error": mse,
            "Mean Absolute Percentage Error": mape,
            "Cross Entropy": ce,
            "HOP": hop,
        }
    return metrics


def generate_reconstructed_output(
    full_circuit: QuantumCircuit,
    subcircuits: Sequence[QuantumCircuit],
    unordered: NDArray,
    smart_order: Sequence[int],
    complete_path_map: Dict[Qubit, Sequence[Dict[str, Union[int, Qubit]]]],
) -> NDArray:
    """
    Reorder the probability distribution.

    Args:
        - full_circuit (QuantumCircuit): the original uncut circuit
        - subcircuits (list): the cut subcircuits
        - unordered (NDArray): the unordered results of the subcircuits
        - smart_order (list): the correct ordering of the subcircuits
        - complete_path_map (dict): the path map of the cuts, as defined from the
            cutting function

    Returns:
        - (NDArray): the reordered and reconstructed probability distribution over the
            full circuit
    """
    subcircuit_out_qubits: Dict[int, List[Qubit]] = {
        subcircuit_idx: [] for subcircuit_idx in smart_order
    }
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        subcircuit_out_qubits[output_qubit["subcircuit_idx"]].append(
            (output_qubit["subcircuit_qubit"], full_circuit.qubits.index(input_qubit))
        )

    for subcircuit_idx in subcircuit_out_qubits:
        subcircuit_out_qubits[subcircuit_idx] = sorted(
            subcircuit_out_qubits[subcircuit_idx],
            key=lambda x: subcircuits[subcircuit_idx].qubits.index(x[0]),
            reverse=True,
        )
        subcircuit_out_qubits[subcircuit_idx] = [
            x[1] for x in subcircuit_out_qubits[subcircuit_idx]
        ]

    unordered_qubit: List[int] = []
    for subcircuit_idx in smart_order:
        unordered_qubit += subcircuit_out_qubits[subcircuit_idx]

    reconstructed_output = np.zeros(len(unordered))

    for unordered_state, unordered_p in enumerate(unordered):
        bin_unordered_state = bin(unordered_state)[2:].zfill(full_circuit.num_qubits)
        _, ordered_bin_state = zip(
            *sorted(zip(unordered_qubit, bin_unordered_state), reverse=True)
        )
        ordered_bin_state_str = "".join([str(x) for x in ordered_bin_state])
        ordered_state = int(ordered_bin_state_str, 2)
        reconstructed_output[ordered_state] = unordered_p

    return np.array(reconstructed_output)


def _evaluate_circuit(circuit: QuantumCircuit) -> Sequence[float]:
    """
    Simulate the given circuit to get the final probability vector.

    Args:
        - circuit (QuantumCircuit): the circuit to simulate

    Returns:
        - (Sequence[float]): the final probability vector of the circuit
    """
    circuit = copy.deepcopy(circuit)
    max_memory_mb = psutil.virtual_memory().total >> 20
    max_memory_mb = int(max_memory_mb / 4 * 3)
    simulator = Aer.get_backend("statevector_simulator")
    result = simulator.run(circuit).result()
    statevector = result.get_statevector(circuit)
    prob_vector = Statevector(statevector).probabilities()

    return prob_vector
