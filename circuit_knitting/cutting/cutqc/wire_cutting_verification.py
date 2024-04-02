# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File that contains the function to verify the results of the cut circuits."""

from __future__ import annotations

import copy
import psutil
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit.utils.deprecation import deprecate_func

from ...utils.conversion import quasi_to_real
from ...utils.metrics import (
    chi2_distance,
    MSE,
    MAPE,
    cross_entropy,
    HOP,
)


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the `circuit_knitting.cutting` package. ",
)
def verify(
    full_circuit: QuantumCircuit,
    reconstructed_output: np.ndarray,
) -> tuple[dict[str, dict[str, float]], Sequence[float]]:
    """
    Compare the reconstructed probabilities to the ground truth.

    Executes the original circuit, then measures the distributional differences between this exact
    result (ground truth) and the reconstructed result from the subcircuits.
    Provides a variety of metrics to evaluate the differences in the distributions.

    Args:
        full_circuit: The original quantum circuit that was cut
        reconstructed_output: The reconstructed probability distribution from the
            execution of the subcircuits

    Returns:
        A tuple containing metrics for the ground truth and reconstructed distributions
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
    return metrics, ground_truth


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the `circuit_knitting.cutting` package. ",
)
def generate_reconstructed_output(
    full_circuit: QuantumCircuit,
    subcircuits: Sequence[QuantumCircuit],
    unordered: np.ndarray,
    smart_order: Sequence[int],
    complete_path_map: dict[Qubit, Sequence[dict[str, int | Qubit]]],
) -> np.ndarray:
    """
    Reorder the probability distribution.

    Args:
        full_circuit: The original uncut circuit
        subcircuits: The cut subcircuits
        unordered: The unordered results of the subcircuits
        smart_order: The correct ordering of the subcircuits
        complete_path_map: The path map of the cuts, as defined from the
            cutting function

    Returns:
        The reordered and reconstructed probability distribution over the
        full circuit
    """
    subcircuit_out_qubits: dict[int, list[Qubit]] = {
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

    unordered_qubit: list[int] = []
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


@deprecate_func(
    removal_timeline="Circuit knitting toolbox 0.8.0 release",
    since="0.7.0",
    package_name="circuit-knitting-toolbox",
    additional_msg="Use the wire cutting or automated cut-finding functionality in the `circuit_knitting.cutting` package. ",
)
def _evaluate_circuit(circuit: QuantumCircuit) -> Sequence[float]:
    """
    Compute exact probability vector of given circuit.

    Args:
        circuit: The circuit to simulate

    Returns:
        The final probability vector of the circuit
    """
    max_memory_mb = psutil.virtual_memory().total >> 20
    max_memory_mb = int(max_memory_mb / 4 * 3)
    simulator = Aer.get_backend(
        "aer_simulator_statevector", max_memory_mb=max_memory_mb
    )
    circuit = copy.deepcopy(circuit)
    circuit.save_state()
    result = simulator.run(circuit).result()
    statevector = result.get_statevector(circuit)
    prob_vector = Statevector(statevector).probabilities()

    return prob_vector
