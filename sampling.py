import numpy as np
import glob
from gurobipy import *

from cutqc.helper_fun import read_prob_from_txt

def dummy_sample(summation_terms):
    '''
    A dummy sampler that samples all summation_terms
    Just to keep the same format of codes with other sampling methods
    '''
    summation_terms_sampled = []
    sampling_prob = 1
    for sample_summation_term_idx, sample_summation_term in enumerate(summation_terms):
        summation_terms_sampled.append({'summation_term_idx':sample_summation_term_idx,'summation_term':sample_summation_term,'sampling_prob':sampling_prob,'frequency':1})
    return summation_terms_sampled

def get_subcircuit_instances_sampled(subcircuit_entries,subcircuit_entry_samples):
    subcircuit_instances_sampled = []
    for subcircuit_entry_sample in subcircuit_entry_samples:
        subcircuit_idx, subcircuit_entry_idx = subcircuit_entry_sample
        kronecker_term = subcircuit_entries[subcircuit_idx][subcircuit_entry_idx]
        for item in kronecker_term:
            coefficient, subcircuit_instance_idx = item
            if (subcircuit_idx,subcircuit_instance_idx) not in subcircuit_instances_sampled:
                subcircuit_instances_sampled.append((subcircuit_idx,subcircuit_instance_idx))
    return subcircuit_instances_sampled

def get_subcircuit_entries_sampled(summation_terms):
    subcircuit_entries_sampled = []
    for summation_term in summation_terms:
        sampling_prob = summation_term['sampling_prob']
        summation_term = summation_term['summation_term']
        for subcircuit_entry in summation_term:
            subcircuit_idx, subcircuit_entry_idx = subcircuit_entry
            if (subcircuit_idx,subcircuit_entry_idx) not in subcircuit_entries_sampled:
                subcircuit_entries_sampled.append((subcircuit_idx,subcircuit_entry_idx))
    return subcircuit_entries_sampled