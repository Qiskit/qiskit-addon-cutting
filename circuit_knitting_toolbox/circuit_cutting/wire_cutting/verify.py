import pickle
import argparse
import glob
import numpy as np
from qiskit.quantum_info import Statevector

from qiskit_helper_functions.non_ibmq_functions import evaluate_circ
from qiskit_helper_functions.conversions import quasi_to_real
from qiskit_helper_functions.metrics import chi2_distance, MSE, MAPE, cross_entropy, HOP

def verify(full_circuit,unordered,complete_path_map,subcircuits,smart_order):
    ground_truth = evaluate_circ(circuit=full_circuit,backend='statevector_simulator')

    '''
    Reorder the probability distribution
    '''
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
    reconstructed_output = np.zeros(len(unordered))
    for unordered_state, unordered_p in enumerate(unordered):
        bin_unordered_state = bin(unordered_state)[2:].zfill(full_circuit.num_qubits)
        _, ordered_bin_state = zip(*sorted(zip(unordered_qubit, bin_unordered_state),reverse=True))
        ordered_bin_state = ''.join([str(x) for x in ordered_bin_state])
        ordered_state = int(ordered_bin_state,2)
        ground_p = ground_truth[ordered_state]
        reconstructed_output[ordered_state] = unordered_p
    reconstructed_output = np.array(reconstructed_output)

    metrics = {}
    for quasi_conversion_mode in ['nearest','naive']:
        real_probability = quasi_to_real(quasiprobability=reconstructed_output,mode=quasi_conversion_mode)
        
        chi2 = chi2_distance(target=ground_truth,obs=real_probability)
        mse = MSE(target=ground_truth,obs=real_probability)
        mape = MAPE(target=ground_truth,obs=real_probability)
        ce = cross_entropy(target=ground_truth,obs=real_probability)
        hop = HOP(target=ground_truth,obs=real_probability)
        metrics[quasi_conversion_mode] = {
            'chi2':chi2,
            'Mean Squared Error':mse,
            'Mean Absolute Percentage Error':mape,
            'Cross Entropy':ce,
            'HOP':hop
            }
    return reconstructed_output, metrics