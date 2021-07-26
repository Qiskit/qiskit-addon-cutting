from time import time

from cutqc.helper_fun import check_valid
from cutqc.cutter_cplex import find_cuts, cut_circuit
from cutqc.evaluator import run_subcircuit_instances
from cutqc.post_process import generate_summation_terms, build
from cutqc.verify import verify

class CutQC:
    '''
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    '''
    def __init__(self, tasks, verbose):
        '''
        Args:
        tasks (list): the input quantum circuits
        Each element is a dictionary with 'name', 'circuit' and 'kwargs'
        
        verbose: setting verbose to True to turn on logging information.
        Useful to visualize what happens,
        but may produce very long outputs for complicated circuits.
        '''
        self.tasks = tasks
        for task in self.tasks:
            for field in ['name','circuit','kwargs']:
                if field not in task:
                    raise ValueError('Missing %s'%field)
            check_valid(circuit=task['circuit'])
        self.verbose = verbose

    def cut(self):
        '''
        Cut the given circuits
        If use the MIP solver to automatically find cuts, the following are required:
        max_subcircuit_width: max number of qubits in each subcircuit
        The following are optional:
        max_cuts: max total number of cuts allowed
        num_subcircuits: list of subcircuits to try, CutQC returns the best solution found among the trials
        max_subcircuit_cuts: max number of cuts for a subcircuit
        max_subcircuit_size: max number of gates in a subcircuit

        Else supply the subcircuit_vertices manually
        Note that supplying subcircuit_vertices overrides all other arguments
        '''
        for task in self.tasks:
            circuit_name = task['name']
            circuit = task['circuit']
            kwargs = task['kwargs']
            if self.verbose:
                print('*'*20,'Cut %s'%circuit_name,'*'*20)
                print('width = %d depth = %d size = %d -->'%(circuit.num_qubits,circuit.depth(),circuit.size()))
                print(kwargs)
            if 'subcircuit_vertices' not in kwargs:
                if 'max_subcircuit_width' not in kwargs:
                    raise AttributeError('Automatic MIP cut searcher requires users to define max subcircuit width!')
                task.update(find_cuts(**kwargs,circuit=circuit,verbose=self.verbose))
            else:
                task.update(cut_circuit(**kwargs,circuit=circuit,verbose=self.verbose))
    
    def evaluate(self, eval_mode, num_shots_fn, mem_limit, num_threads):
        '''
        eval_mode = qasm: simulate shots
        eval_mode = sv: statevector simulation
        num_shots_fn: a function that gives the number of shots to take for a given circuit
        '''
        if self.verbose:
            print('*'*20,'evaluation mode = %s'%(eval_mode),'*'*20,flush=True)
        
        self._generate_metadata()
        self._run_subcircuits(eval_mode=eval_mode,num_shots_fn=num_shots_fn)
        self._attribute_shots()
        self._build(mem_limit=mem_limit,num_threads=num_threads)

    def verify(self):
        for task in self.tasks:
            print('*'*20,'Verify %s'%task['name'],'*'*20,flush=True)
            reconstructed_output, metrics = verify(full_circuit=task['circuit'],
            unordered=task['unordered_prob'],
            complete_path_map=task['complete_path_map'],
            subcircuits=task['subcircuits'],
            smart_order=task['smart_order'])
            for quasi_conversion_mode in metrics:
                print('Quasi probability conversion mode: %s'%quasi_conversion_mode)
                for metric_name in metrics[quasi_conversion_mode]:
                    print(metric_name,metrics[quasi_conversion_mode][metric_name])
            task['ordered_prob'] = reconstructed_output
            task['metrics'] = metrics
    
    def _generate_metadata(self):
        for task in self.tasks:
            task['summation_terms'], task['subcircuit_entries'], task['subcircuit_instances'] = generate_summation_terms(
                subcircuits=task['subcircuits'],
                complete_path_map=task['complete_path_map'],
                num_cuts=task['num_cuts'])
            # if self.verbose:
            #     print('--> %s subcircuit_instances:'%task['name'],flush=True)
            #     for subcircuit_idx in task['subcircuit_instances']:
            #         for init_meas in task['subcircuit_instances'][subcircuit_idx]:
            #             subcircuit_instance_idx = task['subcircuit_instances'][subcircuit_idx][init_meas]
            #             print('Subcircuit {:d}, {}, instance_idx {:d}'.format(
            #                 subcircuit_idx,init_meas,subcircuit_instance_idx))
            #     print('--> %s subcircuit_entries:'%task['name'],flush=True)
            #     for subcircuit_idx in task['subcircuit_entries']:
            #         for subcircuit_entry_key in task['subcircuit_entries'][subcircuit_idx]:
            #             subcircuit_entry_idx, kronecker_term = task['subcircuit_entries'][subcircuit_idx][subcircuit_entry_key]
            #             print('Subcircuit {:d} {}, entry_idx {:d}, Kronecker term = {}'.format(
            #                 subcircuit_idx,subcircuit_entry_key,subcircuit_entry_idx,kronecker_term))
            #     print('--> %s summation_terms:'%task['name'])
            #     [print(summation_term) for summation_term in task['summation_terms']]

    def _run_subcircuits(self,eval_mode,num_shots_fn):
        '''
        Run all the subcircuit instances
        task['subcircuit_instance_probs'][subcircuit_idx][subcircuit_instance_idx] = measured prob
        '''
        for task in self.tasks:
            if self.verbose:
                print('--> Running Subcircuits %s'%task['name'],flush=True)
            task['subcircuit_instance_probs'] = run_subcircuit_instances(subcircuits=task['subcircuits'],subcircuit_instances=task['subcircuit_instances'],
            eval_mode=eval_mode,num_shots_fn=num_shots_fn)
    
    def _attribute_shots(self):
        '''
        Attribute the shots into respective subcircuit entries
        task['subcircuit_entry_probs'][subcircuit_idx][subcircuit_entry_idx] = prob
        '''
        for task in self.tasks:
            if self.verbose:
                print('--> Attribute shots %s'%task['name'],flush=True)
            attribute_begin = time()
            task['subcircuit_entry_probs'] = {}
            for subcircuit_idx in task['subcircuit_entries']:
                task['subcircuit_entry_probs'][subcircuit_idx] = {}
                for label in task['subcircuit_entries'][subcircuit_idx]:
                    subcircuit_entry_idx, kronecker_term = task['subcircuit_entries'][subcircuit_idx][label]
                    # print('Subcircuit {:d} entry {:d} kronecker_term {}'.format(
                    #     subcircuit_idx, subcircuit_entry_idx, kronecker_term
                    # ))
                    subcircuit_entry_prob = None
                    for term in kronecker_term:
                        coefficient, subcircuit_instance_idx = term
                        if subcircuit_entry_prob is None:
                            subcircuit_entry_prob = coefficient * task['subcircuit_instance_probs'][subcircuit_idx][subcircuit_instance_idx]
                        else:
                            subcircuit_entry_prob += coefficient * task['subcircuit_instance_probs'][subcircuit_idx][subcircuit_instance_idx]
                    task['subcircuit_entry_probs'][subcircuit_idx][subcircuit_entry_idx] = subcircuit_entry_prob
            attribute_time = time()-attribute_begin
            if self.verbose:
                print('%s attribute took %.3e seconds'%(task['name'],attribute_time),flush=True)
    
    def _build(self, mem_limit, num_threads):
        for task in self.tasks:
            if self.verbose:
                print('--> Build %s'%task['name'],flush=True)
                [print(summation_term,flush=True) for summation_term in task['summation_terms'][:10]]
                print('... Total %d summation terms\n'%len(task['summation_terms']),flush=True)

            build_begin = time()
            reconstructed_prob, smart_order = build(
                summation_terms=task['summation_terms'],
                subcircuit_entry_probs=task['subcircuit_entry_probs'],
                num_cuts=task['num_cuts'],
                counter=task['counter'],
                verbose=self.verbose)
            build_time = time()-build_begin
            task['unordered_prob'] = reconstructed_prob
            task['build_time'] = build_time
            task['smart_order'] = smart_order

            if self.verbose:
                print('%s build took %.3e seconds'%(task['name'],build_time),flush=True)