from time import time
import itertools, copy
import numpy as np
from qiskit_helper_functions.non_ibmq_functions import read_dict

def get_cut_qubit_pairs(complete_path_map):
	'''
	Get O-Rho cut qubit pairs
	'''
	O_rho_pairs = []
	for input_qubit in complete_path_map:
		path = complete_path_map[input_qubit]
		if len(path)>1:
			for path_ctr, item in enumerate(path[:-1]):
				O_qubit_tuple = item
				rho_qubit_tuple = path[path_ctr+1]
				O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
	return O_rho_pairs

def get_label(label_idx, num_cuts):
	'''
	Get the basis label for each cut point
	'''
	assert label_idx<4**num_cuts
	basis = ['I','X','Y','Z']
	label = []
	for position in range(num_cuts):
		digit = label_idx%4
		label.append(basis[digit])
		label_idx = label_idx//4
	return label

def attribute_label(label, O_rho_pairs, num_subcircuits):
	'''
	Get the label attributed to each subcircuit
	subcircuit_label[subcircuit_idx]['init'] = str of init for the cut qubits
	subcircuit_label[subcircuit_idx]['meas'] = str of meas for the cut qubits
	'''
	subcircuit_label = {subcircuit_idx:{'init':'','meas':''} for subcircuit_idx in range(num_subcircuits)}
	for c, pair in zip(label, O_rho_pairs):
		O_qubit, rho_qubit = pair
		subcircuit_label[O_qubit['subcircuit_idx']]['meas'] += c
		subcircuit_label[rho_qubit['subcircuit_idx']]['init'] += c
	return subcircuit_label

def fill_label(subcircuit_idx, subcircuit, subcircuit_label, O_rho_pairs):
	'''
	Given a subcircuit, its label and O-rho cut qubit pairs
	Fill the full init,meas strings
	'''
	subcircuit_init_label_counter = 0
	subcircuit_meas_label_counter = 0
	init = ['zero' for q in range(subcircuit.num_qubits)]
	meas = ['comp' for q in range(subcircuit.num_qubits)]
	for pair in O_rho_pairs:
		O_qubit, rho_qubit = pair
		if O_qubit['subcircuit_idx']==subcircuit_idx:
			qubit_idx = subcircuit.qubits.index(O_qubit['subcircuit_qubit'])
			meas[qubit_idx] = subcircuit_label['meas'][subcircuit_meas_label_counter]
			subcircuit_meas_label_counter += 1
		if rho_qubit['subcircuit_idx']==subcircuit_idx:
			qubit_idx = subcircuit.qubits.index(rho_qubit['subcircuit_qubit'])
			init[qubit_idx] = subcircuit_label['init'][subcircuit_init_label_counter]
			subcircuit_init_label_counter += 1
	return init, meas

def get_init_meas(init_label, meas_label):
	init_combinations = []
	for x in init_label:
		if x == 'zero':
			init_combinations.append(['zero'])
		elif x == 'I':
			init_combinations.append(['+zero','+one'])
		elif x == 'X':
			init_combinations.append(['2plus','-zero','-one'])
		elif x == 'Y':
			init_combinations.append(['2plusI','-zero','-one'])
		elif x == 'Z':
			init_combinations.append(['+zero','-one'])
		else:
			raise Exception('Illegal initilization symbol :',x)
	init_combinations = list(itertools.product(*init_combinations))

	meas_combinations = []
	for x in meas_label:
		meas_combinations.append([x])
	meas_combinations = list(itertools.product(*meas_combinations))

	subcircuit_init_meas = []
	for init in init_combinations:
		for meas in meas_combinations:
			subcircuit_init_meas.append((tuple(init),tuple(meas)))
	return subcircuit_init_meas

def convert_to_physical_init(init):
	coefficient = 1
	for idx, x in enumerate(init):
		if x == 'zero':
			continue
		elif x == '+zero':
			init[idx] = 'zero'
		elif x == '+one':
			init[idx] = 'one'
		elif x == '2plus':
			init[idx] = 'plus'
			coefficient *= 2
		elif x == '-zero':
			init[idx] = 'zero'
			coefficient *= -1
		elif x == '-one':
			init[idx] = 'one'
			coefficient *= -1
		elif x =='2plusI':
			init[idx] = 'plusI'
			coefficient *= 2
		else:
			raise Exception('Illegal initilization symbol :',x)
	return coefficient, tuple(init)

def generate_summation_terms(subcircuits, complete_path_map, num_cuts):
	'''
	Final CutQC reconstruction result = Sum(summation_terms)

	summation_terms (list) : [summation_term_0, summation_term_1, ...] --> 4^#cuts elements

	summation_term[subcircuit_idx] = subcircuit_entry_idx
	E.g. summation_term = {0:0,1:13,2:7} = Kron(subcircuit_0_entry_0, subcircuit_1_entry_13, subcircuit_2_entry_7)

	subcircuit_entries[subcircuit_idx][init_label,meas_label] = subcircuit_entry_idx, kronecker_term
	kronecker_term (list): (coefficient, subcircuit_instance_idx)
	Add coefficient*subcircuit_instance to subcircuit_entry

	subcircuit_instances[subcircuit_idx][init,meas] = subcircuit_instance_idx
	'''
	summation_terms = []
	subcircuit_entries = {subcircuit_idx:{} for subcircuit_idx in range(len(subcircuits))}
	subcircuit_instances = {subcircuit_idx:{} for subcircuit_idx in range(len(subcircuits))}
	O_rho_pairs = get_cut_qubit_pairs(complete_path_map=complete_path_map)
	for summation_term_idx in range(4**num_cuts):
		label = get_label(label_idx=summation_term_idx, num_cuts=num_cuts)
		# print('%d/%d summation term:'%(summation_term_idx+1,4**num_cuts),label)
		summation_term = {}
		subcircuit_labels = attribute_label(label=label,O_rho_pairs=O_rho_pairs,num_subcircuits=len(subcircuits))
		for subcircuit_idx in range(len(subcircuits)):
			# print('subcircuit %d label :'%subcircuit_idx,subcircuit_labels[subcircuit_idx])
			subcircuit_entry_key = (subcircuit_labels[subcircuit_idx]['init'],subcircuit_labels[subcircuit_idx]['meas'])
			if subcircuit_entry_key in subcircuit_entries[subcircuit_idx]:
				subcircuit_entry_idx, kronecker_term = subcircuit_entries[subcircuit_idx][subcircuit_entry_key]
				# print('Already in, subcircuit_entry {:d} : {}'.format(subcircuit_entry_idx,kronecker_term))
			else:
				subcircuit_full_label = fill_label(subcircuit_idx=subcircuit_idx,
				subcircuit=subcircuits[subcircuit_idx],
				subcircuit_label=subcircuit_labels[subcircuit_idx],
				O_rho_pairs=O_rho_pairs)
				# print('Full label :',subcircuit_full_label)
				subcircuit_init_meas = get_init_meas(init_label=subcircuit_full_label[0],meas_label=subcircuit_full_label[1])
				kronecker_term = []
				for init_meas in subcircuit_init_meas:
					init, meas = init_meas
					coefficient, init = convert_to_physical_init(init=list(init_meas[0]))
					if (init,meas) in subcircuit_instances[subcircuit_idx]:
						subcircuit_instance_idx = subcircuit_instances[subcircuit_idx][(init,meas)]
					else:
						subcircuit_instance_idx = len(subcircuit_instances[subcircuit_idx])
						subcircuit_instances[subcircuit_idx][(init,meas)] = subcircuit_instance_idx
					kronecker_term.append((coefficient,subcircuit_instance_idx))
				subcircuit_entry_idx = len(subcircuit_entries[subcircuit_idx])
				subcircuit_entries[subcircuit_idx][subcircuit_entry_key] = subcircuit_entry_idx, kronecker_term
				# print('New subcircuit_entry, {:d} : {}'.format(subcircuit_entry_idx,kronecker_term))
			summation_term[subcircuit_idx] = subcircuit_entry_idx
		summation_terms.append(summation_term)
		# print('summation_term =',summation_term,'\n')
	return summation_terms, subcircuit_entries, subcircuit_instances

def get_naive_overhead(subcircuit_order,num_cuts,counter):
	overhead = {'additions':0,'multiplications':0}
	accumulated_len = None
	for subcircuit_idx in subcircuit_order:
		prob_len = 2**counter[subcircuit_idx]['effective']
		if accumulated_len is None:
			accumulated_len = prob_len
		else:
			accumulated_len *= prob_len
			overhead['multiplications'] += accumulated_len
	overhead['multiplications'] *= 4**num_cuts
	overhead['additions'] = accumulated_len*(4**num_cuts-1)
	return overhead

def naive_compute(subcircuit_order, summation_terms, subcircuit_entry_probs):
	reconstructed_prob = None
	for idx in range(len(summation_terms)):
		summation_term = summation_terms[idx]
		summation_term_prob = None
		for subcircuit_idx in subcircuit_order:
			subcircuit_entry_idx = summation_term[subcircuit_idx]
			subcircuit_entry_prob = subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_idx]
			if summation_term_prob is None:
				summation_term_prob = subcircuit_entry_prob
			else:
				summation_term_prob = np.kron(summation_term_prob,subcircuit_entry_prob)
		if reconstructed_prob is None:
			reconstructed_prob = summation_term_prob
		else:
			reconstructed_prob += summation_term_prob
	return reconstructed_prob

def build(summation_terms, subcircuit_entry_probs, num_cuts, counter, verbose):
	min_overhead = float('inf')
	for subcircuit_order in itertools.permutations(range(len(subcircuit_entry_probs))):
		overhead = get_naive_overhead(subcircuit_order=subcircuit_order,num_cuts=num_cuts,counter=counter)
		if overhead['additions']+overhead['multiplications']<min_overhead:
			smart_order = subcircuit_order
			min_overhead = overhead['additions']+overhead['multiplications']
			if verbose:
				print('subcircuit_order {}. overhead = {}.'.format(subcircuit_order,overhead))
	
	reconstructed_prob = naive_compute(subcircuit_order, summation_terms, subcircuit_entry_probs)
	reconstructed_prob /= 2**num_cuts
	return reconstructed_prob, smart_order