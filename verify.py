import pickle
import argparse
import glob
import numpy as np

from qiskit_helper_functions.non_ibmq_functions import evaluate_circ
from qiskit_helper_functions.metrics import nearest_probability_distribution

def verify(full_circuit,unordered,complete_path_map,subcircuits,smart_order):
    ground_truth = evaluate_circ(circuit=full_circuit,backend='statevector_simulator')
    subcircuit_out_qubits = {subcircuit_idx:[] for subcircuit_idx in smart_order}
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        subcircuit_out_qubits[output_qubit['subcircuit_idx']].append((output_qubit['subcircuit_qubit'],full_circuit.qubits.index(input_qubit)))
    for subcircuit_idx in subcircuit_out_qubits:
        subcircuit_out_qubits[subcircuit_idx] = sorted(subcircuit_out_qubits[subcircuit_idx],
        key=lambda x:subcircuits[subcircuit_idx].qubits.index(x[0]),reverse=True)
        subcircuit_out_qubits[subcircuit_idx] = [x[1] for x in subcircuit_out_qubits[subcircuit_idx]]
    # print('subcircuit_out_qubits:',subcircuit_out_qubits)
    unordered_qubit = []
    for subcircuit_idx in smart_order:
        unordered_qubit += subcircuit_out_qubits[subcircuit_idx]
    # print('CutQC out qubits:',unordered_qubit)
    unordered, _ = nearest_probability_distribution(quasiprobability=unordered)
    squared_error = 0
    absolute_percentage_error = 0
    reconstructed_output = []
    for unordered_state, unordered_p in enumerate(unordered):
        bin_unordered_state = bin(unordered_state)[2:].zfill(full_circuit.num_qubits)
        _, ordered_bin_state = zip(*sorted(zip(unordered_qubit, bin_unordered_state),reverse=True))
        ordered_bin_state = ''.join([str(x) for x in ordered_bin_state])
        ordered_state = int(ordered_bin_state,2)
        ground_p = ground_truth[ordered_state]
        squared_error += np.power(ground_p-unordered_p,2)
        absolute_percentage_error += abs((ground_p-unordered_p)/ground_p)
        reconstructed_output.append(unordered_p)
    mse = squared_error/len(unordered)
    mape = absolute_percentage_error/len(unordered)
    reconstructed_output = np.array(reconstructed_output)
    return reconstructed_output, mse, mape