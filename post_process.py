import itertools, copy, pickle
from qiskit_helper_functions.non_ibmq_functions import read_dict

def find_init_meas(combination, O_rho_pairs, subcircuits):
    # print('Finding init_meas for',combination)
    all_init_meas = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        init = ['zero' for q in range(subcircuit.num_qubits)]
        meas = ['comp' for q in range(subcircuit.num_qubits)]
        all_init_meas[subcircuit_idx] = [init,meas]
    for s, pair in zip(combination, O_rho_pairs):
        O_qubit, rho_qubit = pair
        O_qubit_subcircuit_qubits = subcircuits[O_qubit['subcircuit_idx']].qubits
        rho_qubit_subcircuit_qubits = subcircuits[rho_qubit['subcircuit_idx']].qubits
        all_init_meas[rho_qubit['subcircuit_idx']][0][rho_qubit_subcircuit_qubits.index(rho_qubit['subcircuit_qubit'])] = s
        all_init_meas[O_qubit['subcircuit_idx']][1][O_qubit_subcircuit_qubits.index(O_qubit['subcircuit_qubit'])] = s
    # print(all_init_meas)
    for subcircuit_idx in all_init_meas:
        init = all_init_meas[subcircuit_idx][0]
        init_combinations = []
        for idx, x in enumerate(init):
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
        meas = all_init_meas[subcircuit_idx][1]
        meas_combinations = []
        for x in meas:
            meas_combinations.append(['%s'%x])
        meas_combinations = list(itertools.product(*meas_combinations))
        subcircuit_init_meas = []
        for init in init_combinations:
            for meas in meas_combinations:
                subcircuit_init_meas.append((tuple(init),tuple(meas)))
        all_init_meas[subcircuit_idx] = subcircuit_init_meas
    # [print('subcircuit_%d'%subcircuit_idx,all_init_meas[subcircuit_idx]) for subcircuit_idx in all_init_meas]
    return all_init_meas

def get_combinations(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path)>1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr+1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    basis = ['I','X','Y','Z']
    combinations = list(itertools.product(basis,repeat=len(O_rho_pairs)))
    return O_rho_pairs, combinations

def generate_summation_terms(full_circuit, subcircuits, complete_path_map, subcircuit_instances_idx, counter):
    '''
    Final CutQC reconstruction result = Sum(summation_terms)

    summation_terms (list) : [summation_term_0, summation_term_1, ...] --> 4^K elements
    
    summation_term[subcircuit_idx] = subcircuit_entry_idx
    E.g. summation_term = {0:0,1:13,2:7} = Kron(subcircuit_0_entry_0, subcircuit_1_entry_13, subcircuit_2_entry_7)

    subcircuit_entries[subcircuit_idx][subcircuit_entry_idx] = kronecker_term
    subcircuit_entries[subcircuit_idx][kronecker_term] = subcircuit_entry_idx
    kronecker_term : ((coefficient,subcircuit_instance_idx), ...)

    subcircuit_instance_attribution[subcircuit_idx][subcircuit_instance_idx] = [coefficient,subcircuit_entry_idx]
    Add coefficient*subcircuit_instance to subcircuit_entry
    '''
    summation_terms = []
    subcircuit_entries = {subcircuit_idx:{} for subcircuit_idx in range(len(subcircuits))}
    O_rho_pairs, combinations = get_combinations(complete_path_map=complete_path_map)
    smart_order = sorted(range(len(subcircuits)),key=lambda subcircuit_idx:counter[subcircuit_idx]['effective'])
    for i, combination in enumerate(combinations):
        # print('%d/%d combinations:'%(i+1,len(combinations)),combination)
        summation_term = []
        all_init_meas = find_init_meas(combination, O_rho_pairs, subcircuits)
        for subcircuit_idx in smart_order:
            kronecker_term = ()
            for init_meas in all_init_meas[subcircuit_idx]:
                # print('Subcircuit_%d init_meas ='%subcircuit_idx,init_meas)
                coefficient = 1
                init = list(init_meas[0])
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
                meas = list(init_meas[1])
                subcircuit_instance_idx = subcircuit_instances_idx[subcircuit_idx][tuple(init),tuple(meas)]
                kronecker_term += ((coefficient,subcircuit_instance_idx),)
            if kronecker_term in subcircuit_entries[subcircuit_idx]:
                subcircuit_entry_idx = subcircuit_entries[subcircuit_idx][kronecker_term]
            else:
                subcircuit_entry_idx = int(len(subcircuit_entries[subcircuit_idx])/2)
                subcircuit_entries[subcircuit_idx][subcircuit_entry_idx] = kronecker_term
                subcircuit_entries[subcircuit_idx][kronecker_term] = subcircuit_entry_idx
            summation_term.append((subcircuit_idx,subcircuit_entry_idx))
        #     print('subcircuit_{:d} kronecker_term = {} --> record as entry {:d}'.format(subcircuit_idx,kronecker_term,subcircuit_entry_idx))
        # print('summation term =',summation_term)
        # print()
        summation_terms.append(summation_term)
    subcircuit_instance_attribution = {subcircuit_idx:{} for subcircuit_idx in range(len(subcircuits))}
    for subcircuit_idx in subcircuit_entries:
        for subcircuit_entry_idx in subcircuit_entries[subcircuit_idx]:
            if type(subcircuit_entry_idx) is int:
                kronecker_term = subcircuit_entries[subcircuit_idx][subcircuit_entry_idx]
                for item in kronecker_term:
                    coefficient, subcircuit_instance_idx = item
                    if subcircuit_instance_idx in subcircuit_instance_attribution[subcircuit_idx]:
                        subcircuit_instance_attribution[subcircuit_idx][subcircuit_instance_idx].append((coefficient,subcircuit_entry_idx))
                    else:
                        subcircuit_instance_attribution[subcircuit_idx][subcircuit_instance_idx] = [(coefficient,subcircuit_entry_idx)]
    return summation_terms, subcircuit_entries, subcircuit_instance_attribution

def distribute_load(total_load,capacities):
    assert total_load<=sum(capacities)
    loads = [0 for x in capacities]
    for slot_idx, load in reversed(list(enumerate(loads))):
        loads[slot_idx] = int(capacities[slot_idx]/sum(capacities)*total_load)
    total_load -= sum(loads)
    for slot_idx, load in reversed(list(enumerate(loads))):
        while total_load>0 and loads[slot_idx]<capacities[slot_idx]:
            loads[slot_idx] += 1
            total_load -= 1
    # print('distributed loads:',loads)
    assert total_load==0
    return loads

def initialize_dynamic_definition_schedule(counter,recursion_qubit,verbose):
    '''
    schedule[recursion_layer] =  {'smart_order','subcircuit_state','upper_bin'}
    subcircuit_state[subcircuit_idx] = ['0','1','active','merged']
    'active' = -1
    'merged' = -2
    '''
    if verbose:
        print('Initializing first DD recursion, recursion_qubit=%d.'%recursion_qubit,flush=True)
    # print(counter)
    schedule = {}
    smart_order = sorted(list(counter.keys()),key=lambda x:counter[x]['effective'])
    # print('smart_order :',smart_order)
    schedule['smart_order'] = smart_order
    schedule['subcircuit_state'] = {}
    schedule['upper_bin'] = None

    subcircuit_capacities = [counter[subcircuit_idx]['effective'] for subcircuit_idx in smart_order]
    # print('subcircuit_capacities:',subcircuit_capacities)
    if sum(subcircuit_capacities)<=recursion_qubit:
        subcircuit_active_qubits = distribute_load(total_load=sum(subcircuit_capacities),capacities=subcircuit_capacities)
    else:
        subcircuit_active_qubits = distribute_load(total_load=recursion_qubit,capacities=subcircuit_capacities)
    # print('subcircuit_active_qubits:',subcircuit_active_qubits)
    for subcircuit_idx, subcircuit_active_qubit in zip(smart_order,subcircuit_active_qubits):
        num_zoomed = 0
        num_active = subcircuit_active_qubit
        num_merged = counter[subcircuit_idx]['effective'] - num_zoomed - num_active
        schedule['subcircuit_state'][subcircuit_idx] = [-1]*num_active + [-2]*num_merged
    if verbose:
        [print(x,schedule[x],flush=True) for x in schedule]
    return schedule

def next_dynamic_definition_schedule(recursion_layer,schedule,state_idx,recursion_qubit,verbose):
    # if verbose:
    #     print('Get DD recursion %d, recursion_qubit=%d.'%(recursion_layer,recursion_qubit),flush=True)
    num_active = 0
    for subcircuit_idx in schedule['subcircuit_state']:
        num_active += schedule['subcircuit_state'][subcircuit_idx].count(-1)
    bin_state_idx = bin(state_idx)[2:].zfill(num_active)
    # print('bin_state_idx = %s'%(bin_state_idx))
    bin_state_idx_ptr = 0
    for subcircuit_idx in schedule['smart_order']:
        for qubit_ctr, qubit_state in enumerate(schedule['subcircuit_state'][subcircuit_idx]):
            if qubit_state==-1:
                schedule['subcircuit_state'][subcircuit_idx][qubit_ctr] = int(bin_state_idx[bin_state_idx_ptr])
                bin_state_idx_ptr += 1
    schedule['smart_order'] = sorted(schedule['smart_order'],key=lambda x:schedule['subcircuit_state'][x].count(-2))
    subcircuit_capacities = [schedule['subcircuit_state'][subcircuit_idx].count(-2) for subcircuit_idx in schedule['smart_order']]
    # print('subcircuit_capacities:',subcircuit_capacities)
    if sum(subcircuit_capacities)<=recursion_qubit:
        subcircuit_active_qubits = distribute_load(total_load=sum(subcircuit_capacities),capacities=subcircuit_capacities)
    else:
        subcircuit_active_qubits = distribute_load(total_load=recursion_qubit,capacities=subcircuit_capacities)
    # print('subcircuit_active_qubits:',subcircuit_active_qubits)
    for subcircuit_idx, subcircuit_active_qubit in zip(schedule['smart_order'],subcircuit_active_qubits):
        for qubit_ctr, qubit_state in enumerate(schedule['subcircuit_state'][subcircuit_idx]):
            if qubit_state==-2 and subcircuit_active_qubit>0:
                schedule['subcircuit_state'][subcircuit_idx][qubit_ctr] = -1
                subcircuit_active_qubit -= 1
    schedule['upper_bin'] = (recursion_layer,state_idx)
    if verbose:
        [print(x,schedule[x],flush=True) for x in schedule]
    return schedule

def find_max_recursion_layer(curr_recursion_layer,dest_folder):
    max_subgroup_prob = 0
    max_recursion_layer = -1
    for recursion_layer in range(curr_recursion_layer):
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
        build_output = read_dict(filename='%s/build_output.pckl'%(dynamic_definition_folder))
        zoomed_ctr = build_output['zoomed_ctr']
        max_states = build_output['max_states']
        reconstructed_prob = build_output['reconstructed_prob']
        schedule = build_output['dd_schedule']
        # print('layer %d schedule'%recursion_layer,schedule)
        num_merged = 0
        for subcircuit_idx in schedule['subcircuit_state']:
            num_merged += schedule['subcircuit_state'][subcircuit_idx].count(-2)
        if num_merged==0 or zoomed_ctr==len(max_states):
            '''
            num_merged==0 : all qubits have been computed for this DD Layer
            zoomed_ctr==len(max_states) : all bins have been computed for this DD layer
            '''
            continue
        print('Examine recursion_layer %d, zoomed_ctr = %d, max_state = %d, p = %e'%(
            recursion_layer,zoomed_ctr,max_states[zoomed_ctr],reconstructed_prob[max_states[zoomed_ctr]]))
        if reconstructed_prob[max_states[zoomed_ctr]]>max_subgroup_prob and reconstructed_prob[max_states[zoomed_ctr]]>1e-16:
            max_subgroup_prob = reconstructed_prob[max_states[zoomed_ctr]]
            max_recursion_layer = recursion_layer
    return max_recursion_layer

def generate_dd_schedule(recursion_layer,counter,recursion_qubit,dest_folder,verbose):
    if recursion_layer==0:
        dd_schedule = initialize_dynamic_definition_schedule(counter=counter,recursion_qubit=recursion_qubit,verbose=verbose)
        return dd_schedule
    else:
        max_recursion_layer = find_max_recursion_layer(curr_recursion_layer=recursion_layer,dest_folder=dest_folder)
        if max_recursion_layer==-1:
            return None
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,max_recursion_layer)
        build_output = read_dict(filename='%s/build_output.pckl'%(dynamic_definition_folder))
        zoomed_ctr = build_output['zoomed_ctr']
        max_states = build_output['max_states']
        reconstructed_prob = build_output['reconstructed_prob']
        schedule = build_output['dd_schedule']
        print('Zoom in for results of recursion_layer %d'%max_recursion_layer,schedule,flush=True)
        print('state_idx = %d, p = %e'%(max_states[zoomed_ctr],reconstructed_prob[max_states[zoomed_ctr]]),flush=True)
        next_schedule = next_dynamic_definition_schedule(recursion_layer=max_recursion_layer,
        schedule=copy.deepcopy(schedule),state_idx=max_states[zoomed_ctr],recursion_qubit=recursion_qubit,verbose=verbose)
        build_output['zoomed_ctr'] += 1
        pickle.dump(build_output, open('%s/build_output.pckl'%(dynamic_definition_folder),'wb'))
        return next_schedule