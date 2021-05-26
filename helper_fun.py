import os
from qiskit.converters import circuit_to_dag
import numpy as np

def check_valid(circuit):
    '''
    If the input circuit is not fully connected, it does not need CutQC to be split into smaller circuits.
    CutQC hence only cuts a circuit if it is fully connected.
    Furthermore, CutQC only supports 2-qubit gates.
    '''
    if circuit.num_unitary_factors()!=1:
        raise ValueError('Input circuit is not fully connected thus does not need cutting. Number of unitary factors = %d'%circuit.num_unitary_factors())
    if circuit.num_clbits>0:
        raise ValueError('Please remove classical bits from the circuit before cutting')
    dag = circuit_to_dag(circuit)
    for op_node in dag.topological_op_nodes():
        if len(op_node.qargs)>2:
            raise ValueError('CutQC currently does not support >2-qubit gates')
        if op_node.op.name=='barrier':
            raise ValueError('Please remove barriers from the circuit before cutting')

def get_dirname(circuit_name,max_subcircuit_qubit,eval_mode,num_threads,mem_limit,field):
    '''
    Directory management for CutQC.
    '''
    if field=='cutter':
        dirname = './cutqc_data/%s/qc_%d'%(circuit_name,max_subcircuit_qubit)
    elif field=='evaluator':
        dirname = './cutqc_data/%s/qc_%d/%s'%(circuit_name,max_subcircuit_qubit,eval_mode)
    elif field=='build':
        dirname = './cutqc_data/%s/qc_%d/%s_%d_%d'%(circuit_name,max_subcircuit_qubit,eval_mode,mem_limit,num_threads)
    elif field=='slurm':
        dirname = './slurm/%s/qc_%d'%(circuit_name,max_subcircuit_qubit)
    elif field=='runtime':
        dirname = './runtime/%s/qc_%d/q_%d_%d'%(circuit_name,max_subcircuit_qubit,mem_limit,num_threads)
    else:
        raise Exception('Illegal field = %s'%field)
    return dirname

def read_prob_from_txt(filename):
    if os.path.isfile(filename):
        txt_file = open(filename,'r')
        lines = txt_file.readlines()
        assert len(lines)==1
        prob = lines[0].rstrip().split(' ')
        prob = np.array(prob,dtype=float)
    else:
        prob = np.array([])
    return prob