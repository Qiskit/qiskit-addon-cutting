from __future__ import annotations

import itertools, copy
import numpy as np
from decimal import *
from typing import Sequence, Any, Dict

from .wire_cutting_post_processing import build
from concurrent.futures import ThreadPoolExecutor

def dd_build(
    #summation_terms: Sequence[dict[int, int]],
    summation_terms: dict[dict[int, int]],
    subcircuit_entry_probs: dict[int, dict[int, np.ndarray]],
    num_cuts: int,
    mem_limit: int,
    recursion_depth: int,
    counter: dict[int, dict[str, int]],
    subcircuit_instances: dict[int, dict[tuple[tuple[str, ...], tuple[Any, ...]], int]],
    num_threads: int
) -> tuple[np.ndarray, list[int], dict[str, int]]:

    """
    Reconstruct the full probabilities from bis of dynamic definition

    Args:
        summation_terms: The summation terms used to generate the full
            vector, as generated in generate_summation_terms
        subcircuit_entry_probs: The probabilities vectors from the
            subcircuit executions
        num_cuts: The number of cuts
        mem_limit: Memory limit size
        recursion_depth: The number of iteration for calculating bin
        counter: The dictionary containing all meta information regarding
            each of the subcircuits
        num_threads: The number of threads to use for multithreading
        circuit: The original full circuit
        cuts: The results of cutting
        subcircuit_instances: The dictionary containing the index information for each
            of the subcircuit instances

    Returns:
        The reconstructed probability vector
        dd_bins: The bin for Dynamic Definition
    """

    dd_bins = {}
    dd_schedule = {}
    overhead = {'additions':0,'multiplications':0}

    num_qubits = sum([count["effective"] for count in counter.values()])
    largest_bins = [] # [{recursion_layer, bin_id}]
    recursion_layer = 0

    while recursion_layer<recursion_depth:
        print('-'*10,'Recursion Layer %d'%(recursion_layer),'-'*10)
        #  Get qubit states
        if recursion_layer==0:
            dd_schedule = initialize_dynamic_definition_schedule(counter, mem_limit)
        elif len(largest_bins)==0:
            break
        else:
            bin_to_expand = largest_bins.pop(0)
            dd_schedule = next_dynamic_definition_schedule(
                recursion_layer=bin_to_expand['recursion_layer'],
                bin_id=bin_to_expand['bin_id'],
                dd_bins=dd_bins, 
                mem_limit=mem_limit)

        merged_subcircuit_entry_probs = merge_states_into_bins(subcircuit_instances, subcircuit_entry_probs, counter, dd_schedule, num_threads)
        reconstructed_prob, smart_order, recursion_overhead = build(
                summation_terms=summation_terms,
                subcircuit_entry_probs=merged_subcircuit_entry_probs,
                num_cuts=num_cuts,
                num_threads=1,
        )

        #  Build from the merged subcircuit entries
        overhead['additions'] += recursion_overhead['additions']
        overhead['multiplications'] += recursion_overhead['multiplications']
            
        dd_bins[recursion_layer] = dd_schedule
        dd_bins[recursion_layer]['smart_order'] = smart_order
        dd_bins[recursion_layer]['bins'] = reconstructed_prob
        dd_bins[recursion_layer]['expanded_bins'] = []
        [print(field,dd_bins[recursion_layer][field]) for field in dd_bins[recursion_layer]]
            
        #  Sort and truncate the largest bins
        has_merged_states = False
        for subcircuit_idx in dd_schedule['subcircuit_state']:
            if 'merged' in dd_schedule['subcircuit_state'][subcircuit_idx]:
                has_merged_states = True
                break
        if recursion_layer<recursion_depth-1 and has_merged_states:
            bin_indices = np.argpartition(reconstructed_prob, -recursion_depth)[-recursion_depth:]
            for bin_id in bin_indices:
                if reconstructed_prob[bin_id]>1/2**num_qubits/10:
                    largest_bins.append({'recursion_layer':recursion_layer,'bin_id':bin_id,'prob':reconstructed_prob[bin_id]})
            largest_bins = sorted(largest_bins,key=lambda bin:bin['prob'],reverse=True)[:recursion_depth]
            print("largest_bins", largest_bins)
        recursion_layer += 1
    return dd_bins
    
def initialize_dynamic_definition_schedule(counter, mem_limit):
    schedule = {}
    schedule['subcircuit_state'] = {}
    schedule['upper_bin'] = None

    subcircuit_capacities = {subcircuit_idx:counter[subcircuit_idx]['effective'] for subcircuit_idx in counter}
    #print("sun capa", subcircuit_capacities)
    subcircuit_active_qubits = distribute_load(capacities=subcircuit_capacities, mem_limit=mem_limit)
    #print('subcircuit_active_qubits:',subcircuit_active_qubits)
    for subcircuit_idx in subcircuit_active_qubits:
        num_zoomed = 0
        num_active = subcircuit_active_qubits[subcircuit_idx]
        num_merged = counter[subcircuit_idx]['effective'] - num_zoomed - num_active
        schedule['subcircuit_state'][subcircuit_idx] = ['active' for _ in range(num_active)] + ['merged' for _ in range(num_merged)]
    #print("sched", schedule)
    return schedule

def next_dynamic_definition_schedule(recursion_layer, bin_id, dd_bins, mem_limit):
    print('Zoom in recursion layer %d bin %d'%(recursion_layer,bin_id))
    num_active = 0
    for subcircuit_idx in dd_bins[recursion_layer]['subcircuit_state']:
        num_active += dd_bins[recursion_layer]['subcircuit_state'][subcircuit_idx].count('active')
    binary_bin_idx = bin(bin_id)[2:].zfill(num_active)
    print('binary_bin_idx = %s'%(binary_bin_idx))
    smart_order = dd_bins[recursion_layer]['smart_order']
    next_dd_schedule = {'subcircuit_state':copy.deepcopy(dd_bins[recursion_layer]['subcircuit_state'])}
    binary_state_idx_ptr = 0
    for subcircuit_idx in smart_order:
        for qubit_ctr, qubit_state in enumerate(next_dd_schedule['subcircuit_state'][subcircuit_idx]):
            if qubit_state=='active':
                next_dd_schedule['subcircuit_state'][subcircuit_idx][qubit_ctr] = int(binary_bin_idx[binary_state_idx_ptr])
                binary_state_idx_ptr += 1
    next_dd_schedule['upper_bin'] = (recursion_layer, bin_id)
    
    subcircuit_capacities = {subcircuit_idx:next_dd_schedule['subcircuit_state'][subcircuit_idx].count('merged') for subcircuit_idx in next_dd_schedule['subcircuit_state']}
    subcircuit_active_qubits = distribute_load(capacities=subcircuit_capacities, mem_limit=mem_limit)
    #print('subcircuit_active_qubits:',subcircuit_active_qubits)
    for subcircuit_idx in next_dd_schedule['subcircuit_state']:
        num_active = subcircuit_active_qubits[subcircuit_idx]
        for qubit_ctr, qubit_state in enumerate(next_dd_schedule['subcircuit_state'][subcircuit_idx]):
            if qubit_state=='merged' and num_active>0:
                next_dd_schedule['subcircuit_state'][subcircuit_idx][qubit_ctr] = 'active'
                num_active -= 1
        assert num_active==0
    return next_dd_schedule

def distribute_load(capacities, mem_limit):
    total_load = min(sum(capacities.values()),mem_limit)
    total_capacity = sum(capacities.values())
    loads = {subcircuit_idx:0 for subcircuit_idx in capacities}
    #print("debug loads", loads)

    for slot_idx in loads:
        #print(capacities[slot_idx], total_capacity, total_load)
        loads[slot_idx] = int(capacities[slot_idx]/total_capacity*total_load)
    total_load -= sum(loads.values())

    for slot_idx in loads:
        while total_load>0 and loads[slot_idx]<capacities[slot_idx]:
            loads[slot_idx] += 1
            total_load -= 1
    print('capacities = {}. total_capacity = {:d}'.format(capacities,total_capacity))
    print('loads = {}. remaining total_load = {:d}'.format(loads,total_load))
    assert total_load==0
    return loads

def merge_prob_vector(unmerged_prob_vector, qubit_states):
    num_active = qubit_states.count('active')
    num_merged = qubit_states.count('merged')
    merged_prob_vector = np.zeros(2**num_active,dtype='float32')
    print('merging with qubit states {}. {:d}-->{:d}'.format(
        qubit_states,
        len(unmerged_prob_vector),len(merged_prob_vector)))
    for active_qubit_states in itertools.product(['0','1'],repeat=num_active):
        #print("active_qubit_states", active_qubit_states)
        if len(active_qubit_states)>0:
            merged_bin_id = int(''.join(active_qubit_states),2)
        else:
            merged_bin_id = 0
        for merged_qubit_states in itertools.product(['0','1'],repeat=num_merged):
            active_ptr = 0
            merged_ptr = 0
            binary_state_id = ''
            for qubit_state in qubit_states:
                if qubit_state=='active':
                    binary_state_id += active_qubit_states[active_ptr]
                    active_ptr += 1
                elif qubit_state=='merged':
                    binary_state_id += merged_qubit_states[merged_ptr]
                    merged_ptr += 1
                else:
                    binary_state_id += '%s'%qubit_state
            state_id = int(binary_state_id,2)
            merged_prob_vector[merged_bin_id] += unmerged_prob_vector[state_id]
    return merged_prob_vector

def find_process_jobs(jobs, rank, num_workers):
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

def merge_state_into_bins_parallel(subcircuit_instances, subcircuit_entry_probs, dd_schedule, num_workers, rank):
    merged_subcircuit_entry_probs = {}
    for subcircuit_idx in subcircuit_instances:
        merged_subcircuit_entry_probs[subcircuit_idx] = {}
        rank_jobs = find_process_jobs(jobs=list(subcircuit_instances[subcircuit_idx].keys()),rank=rank,num_workers=num_workers)
        #print("rank", rank_jobs) 
        for subcircuit_entry_init_meas in rank_jobs:
            subcircuit_entry_id = subcircuit_instances[subcircuit_idx][subcircuit_entry_init_meas]
            print(dd_schedule["subcircuit_state"][subcircuit_idx])
            merged_subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_id] = merge_prob_vector(subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_id], dd_schedule["subcircuit_state"][subcircuit_idx])
    return merged_subcircuit_entry_probs

def merge_states_into_bins(subcircuit_instances, subcircuit_entry_probs, counter, dd_schedule, num_workers):
    #  The first merge of subcircuit probs using the target number of bins
    #  Saves the overhead of writing many states in the first SM recursion

    num_entries = [len(subcircuit_instances[subcircuit_idx]) for subcircuit_idx in subcircuit_instances]
    subcircuit_num_qubits = [counter[subcircuit_idx]['effective'] for subcircuit_idx in counter]

    workers_merge_subcircuit_entry_probs = []
    
    future_prob = []
    executor = ThreadPoolExecutor(max_workers=num_workers)

    for rank in range(num_workers):
        future_prob.append(executor.submit(merge_state_into_bins_parallel, subcircuit_instances, subcircuit_entry_probs, dd_schedule, num_workers, rank))
    for future in future_prob: 
        workers_merge_subcircuit_entry_probs.append(future.result())
    print("workers", workers_merge_subcircuit_entry_probs)


    merged_subcircuit_entry_probs = {}
    for rank in range(num_workers):
        rank_merged_subcircuit_entry_probs = workers_merge_subcircuit_entry_probs[rank]
        for subcircuit_idx in rank_merged_subcircuit_entry_probs:
            if subcircuit_idx not in merged_subcircuit_entry_probs:
                merged_subcircuit_entry_probs[subcircuit_idx] = rank_merged_subcircuit_entry_probs[subcircuit_idx]
            else:
                merged_subcircuit_entry_probs[subcircuit_idx].update(rank_merged_subcircuit_entry_probs[subcircuit_idx])
    #print(merged_subcircuit_entry_probs)
    return merged_subcircuit_entry_probs

def read_dd_bins(subcircuit_out_qubits, dd_bins):
    num_qubits = sum([len(subcircuit_out_qubits[subcircuit_idx]) for subcircuit_idx in subcircuit_out_qubits])
    reconstructed_prob = np.zeros(2**num_qubits,dtype=np.float32)
    print(subcircuit_out_qubits)
    print("read dd bin", dd_bins)
    for recursion_layer in dd_bins:
        print('-'*20,'Verify Recursion Layer %d'%recursion_layer,'-'*20)
        [print(field,dd_bins[recursion_layer][field]) for field in dd_bins[recursion_layer]]
        num_active = sum([dd_bins[recursion_layer]['subcircuit_state'][subcircuit_idx].count('active') for subcircuit_idx in dd_bins[recursion_layer]['subcircuit_state']])
        for bin_id, bin_prob in enumerate(dd_bins[recursion_layer]['bins']):
            if bin_prob>0 and bin_id not in dd_bins[recursion_layer]['expanded_bins']:
                binary_bin_id = bin(bin_id)[2:].zfill(num_active)
                print('dd bin %s'%binary_bin_id)
                binary_full_state = ['' for _ in range(num_qubits)]
                for subcircuit_idx in dd_bins[recursion_layer]['smart_order']:
                    subcircuit_state = dd_bins[recursion_layer]['subcircuit_state'][subcircuit_idx]
                    for subcircuit_qubit_idx, qubit_state in enumerate(subcircuit_state):
                        qubit_idx = subcircuit_out_qubits[subcircuit_idx][subcircuit_qubit_idx]
                        if qubit_state=='active':
                            binary_full_state[qubit_idx] = binary_bin_id[0]
                            binary_bin_id = binary_bin_id[1:]
                        else:
                            binary_full_state[qubit_idx] = '%s'%qubit_state
                print('reordered qubit state = {}'.format(binary_full_state))
                merged_qubit_indices = []
                for qubit, qubit_state in enumerate(binary_full_state):
                    if qubit_state=='merged':
                        merged_qubit_indices.append(qubit)
                num_merged = len(merged_qubit_indices)
                print("bin_prob", bin_prob)
                average_state_prob = bin_prob/2**num_merged
                for binary_merged_state in itertools.product(['0','1'],repeat=num_merged):
                    for merged_qubit_ctr in range(num_merged):
                        binary_full_state[merged_qubit_indices[merged_qubit_ctr]] = binary_merged_state[merged_qubit_ctr]
                    full_state = ''.join(binary_full_state)[::-1]
                    full_state_idx = int(full_state,2)
                    reconstructed_prob[full_state_idx] = average_state_prob
                    print('--> full state {} {:d}. p = {:.3e}'.format(full_state,full_state_idx,average_state_prob))
                print()
    return reconstructed_prob


def get_reconstruction_qubit_order(full_circuit, complete_path_map, subcircuits):
        '''
        Get the output qubit in the full circuit for each subcircuit
        Qiskit orders the full circuit output in descending order of qubits
        '''
        subcircuit_out_qubits = {subcircuit_idx:[] for subcircuit_idx in range(len(subcircuits))}
        for input_qubit in complete_path_map:
                path = complete_path_map[input_qubit]
                output_qubit = path[-1]
                subcircuit_out_qubits[output_qubit['subcircuit_idx']].append((output_qubit['subcircuit_qubit'],full_circuit.qubits.index(input_qubit)))
        for subcircuit_idx in subcircuit_out_qubits:
                subcircuit_out_qubits[subcircuit_idx] = sorted(subcircuit_out_qubits[subcircuit_idx],
                key=lambda x:subcircuits[subcircuit_idx].qubits.index(x[0]),reverse=True)
                subcircuit_out_qubits[subcircuit_idx] = [x[1] for x in subcircuit_out_qubits[subcircuit_idx]]
        return subcircuit_out_qubits
