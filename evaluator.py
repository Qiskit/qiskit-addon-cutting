import itertools, copy, random
import numpy as np
from time import time
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate

from qiskit_helper_functions.non_ibmq_functions import read_dict, find_process_jobs, evaluate_circ

def generate_subcircuit_instances(subcircuits,complete_path_map):
    '''
    Generate subcircuit instances with different init, meas
    subcircuit_instances[subcircuit_idx][subcircuit_instance_idx] = circuit, init, meas, shots
    subcircuit_instances_idx[subcircuit_idx][init,meas] = subcircuit_instance_idx
    '''
    subcircuit_instances = {}
    subcircuit_instances_idx = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        O_qubits, rho_qubits = find_subcircuit_O_rho_qubits(complete_path_map=complete_path_map,subcircuit_idx=subcircuit_idx)
        combinations = find_init_meas_combinations(O_qubits=O_qubits, rho_qubits=rho_qubits, qubits=subcircuit.qubits)
        subcircuit_instances[subcircuit_idx], subcircuit_instances_idx[subcircuit_idx] = get_one_subcircuit_instances(subcircuit=subcircuit, combinations=combinations)
    return subcircuit_instances, subcircuit_instances_idx

def find_subcircuit_O_rho_qubits(complete_path_map,subcircuit_idx):
    '''
    Find the O and Rho qubits of a subcircuit
    '''
    O_qubits = []
    rho_qubits = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for q in path[:-1]:
                if q['subcircuit_idx'] == subcircuit_idx:
                    O_qubits.append(q)
            for q in path[1:]:
                if q['subcircuit_idx'] == subcircuit_idx:
                    rho_qubits.append(q)
    return O_qubits, rho_qubits

def find_init_meas_combinations(O_qubits, rho_qubits, qubits):
    '''
    Find combinations of init, meas
    for a particular circuit
    '''
    measurement_basis = ['I','X','Y']
    init_states = ['zero','one','plus','plusI']
    # print('\u03C1 qubits :',rho_qubits)
    all_inits = itertools.product(init_states,repeat=len(rho_qubits))
    complete_inits = []
    for init in all_inits:
        complete_init = ['zero' for i in range(len(qubits))]
        for i in range(len(init)):
            complete_init[qubits.index(rho_qubits[i]['subcircuit_qubit'])] = init[i]
        complete_inits.append(complete_init)
    # print('initializations:',complete_inits)

    # print('O qubits:',O_qubits)
    all_meas = itertools.product(measurement_basis,repeat=len(O_qubits))
    complete_meas = []
    for meas in all_meas:
        complete_m = ['comp' for i in range(len(qubits))]
        for i in range(len(meas)):
            complete_m[qubits.index(O_qubits[i]['subcircuit_qubit'])] = meas[i]
        complete_meas.append(complete_m)
    # print('measurement basis:',complete_meas)

    combinations = list(itertools.product(complete_inits,complete_meas))
    return combinations

def mutate_measurement_basis(meas):
    '''
    I and Z measurement basis correspond to the same logical circuit
    '''
    if all(x!='I' for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != 'I':
                mutated_meas.append([x])
            else:
                mutated_meas.append(['I','Z'])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas

def get_one_subcircuit_instances(subcircuit, combinations):
    '''
    Modify the different init, meas for a given subcircuit
    Returns:
    subcircuit_instances[subcircuit_instance_idx] = circuit, init, meas, shots
    subcircuit_instances_idx[init,meas] = subcircuit_instance_idx
    '''
    subcircuit_instances = {}
    subcircuit_instances_idx = {}
    for combination_ctr, combination in enumerate(combinations):
        # print('combination %d/%d :'%(combination_ctr+1,len(combinations)),combination)
        subcircuit_dag = circuit_to_dag(subcircuit)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = subcircuit.qubits[i]
            if x == 'zero':
                continue
            elif x == 'one':
                subcircuit_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus':
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus':
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plusI':
                subcircuit_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minusI':
                subcircuit_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal initialization : ',x)
        for i,x in enumerate(meas):
            q = subcircuit.qubits[i]
            if x == 'I' or x == 'comp':
                continue
            elif x == 'X':
                subcircuit_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                subcircuit_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                subcircuit_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        subcircuit_inst = dag_to_circuit(subcircuit_dag)
        num_shots = max(8192,int(2**subcircuit_inst.num_qubits))
        num_shots = min(8192*10,num_shots)
        mutated_meas = mutate_measurement_basis(meas)
        for idx, meas in enumerate(mutated_meas):
            subcircuit_instance_idx = len(subcircuit_instances)
            if idx==0:
                parent_subcircuit_instance_idx = subcircuit_instance_idx
                shots = num_shots
            else:
                shots = 0
            subcircuit_instances[subcircuit_instance_idx] = {'circuit':subcircuit_inst, 'init':tuple(inits), 'meas':tuple(meas),'shots':shots,'parent':parent_subcircuit_instance_idx}
            subcircuit_instances_idx[(tuple(inits),tuple(meas))] = subcircuit_instance_idx
    return subcircuit_instances, subcircuit_instances_idx

def simulate_subcircuit(key,subcircuit_info,eval_mode):
    '''
    Simulate a subcircuit
    Returns measured_prob (list)
    (int state, probability weightage)
    '''
    tol = 1e-12
    circuit_name, subcircuit_idx, parent_subcircuit_instance_idx = key
    subcircuit = subcircuit_info['circuit']
    shots = subcircuit_info['shots']
    init = subcircuit_info['init']
    meas = subcircuit_info['meas']
    subcircuit_results = {}

    assert shots>0
    if eval_mode=='runtime':
        num_effective_qubits = meas[0].count('comp')
        uniform_p = 1/2**num_effective_qubits
        for m in meas:
            measured_prob = uniform_p
            subcircuit_results[(subcircuit_idx,init,m)] = measured_prob
    else:
        if eval_mode=='sv':
            subcircuit_inst_prob = evaluate_circ(circuit=subcircuit,backend='statevector_simulator')
        elif eval_mode=='qasm':
            subcircuit_inst_prob = evaluate_circ(circuit=subcircuit,backend='noiseless_qasm_simulator',options={'num_shots':shots})
        else:
            raise NotImplementedError
        for m in meas:
            measured_prob = measure_prob(unmeasured_prob=subcircuit_inst_prob,meas=m)
            measured_prob[abs(measured_prob) < tol] = 0.0
            subcircuit_results[(subcircuit_idx,init,m)] = measured_prob
    return circuit_name, subcircuit_results

def measure_prob(unmeasured_prob,meas):
    if meas.count('comp')==len(meas):
        return unmeasured_prob
    else:
        measured_prob = np.zeros(int(2**meas.count('comp')))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state,meas=meas)
            # TODO: Add states merging here. Change effective_state to merged_bin
            measured_prob[effective_state] += sigma*p
        return measured_prob

def measure_state(full_state,meas):
    '''
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    '''
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ''
    for meas_bit, meas_basis in zip(bin_full_state,meas[::-1]):
        if meas_bit=='1' and meas_basis!='I' and meas_basis!='comp':
            sigma*=-1
        if meas_basis=='comp':
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state,2) if bin_effective_state!='' else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state