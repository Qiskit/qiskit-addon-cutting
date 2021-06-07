import os, subprocess, pickle, glob, random, time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

from qiskit_helper_functions.non_ibmq_functions import evaluate_circ, read_dict, find_process_jobs
from qiskit_helper_functions.schedule import Scheduler

from cutqc.helper_fun import check_valid, get_dirname
from cutqc.cutter import find_cuts, cut_circuit
from cutqc.evaluator import generate_subcircuit_instances, simulate_subcircuit, measure_prob
from cutqc.sampling import dummy_sample, get_subcircuit_instances_sampled, get_subcircuit_entries_sampled
from cutqc.post_process import generate_summation_terms
from cutqc.verify import verify

class CutQC:
    '''
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    '''
    def __init__(self, circuit_name, circuit, verbose):
        '''
        Args:
        circuit: the input quantum circuit
        circuit_name: can be arbitrary
        verbose: setting verbose to True to turn on logging information.
        Useful to visualize what happens,
        but may produce very long outputs for complicated circuits.
        '''
        check_valid(circuit=circuit)
        self.circuit_name = circuit_name
        self.circuit = circuit
        self.verbose = verbose
    
    def cut(self,
    max_subcircuit_qubit=None, max_cuts=None, num_subcircuits=None,
    subcircuit_vertices=None):
        '''
        Cut the given circuit

        If use the MIP solver to automatically find cuts, supply
        max_subcircuit_qubit: max number of qubits in each subcircuit
        max_cuts: max number of cuts allowed
        num_subcircuits: list of subcircuits to try, CutQC returns the best solution found among trials

        Else supply subcircuit_vertices manually
        Note that subcircuit_vertices override all other arguments
        '''
        if self.verbose:
            print('*'*20,'Cut','*'*20)
            print('%s : width = %d, depth = %d, size = %d'%(
                self.circuit_name,
                self.circuit.num_qubits,
                self.circuit.depth(),
                self.circuit.size()))
        
        if subcircuit_vertices is None:
            if max_subcircuit_qubit is None or max_cuts is None or num_subcircuits is None:
                raise AttributeError('Check the specifications requirement of the automatic MIP cut searcher!')
            cut_solution = find_cuts(circuit=self.circuit,
            max_subcircuit_qubit=max_subcircuit_qubit,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            verbose=self.verbose)
        else:
            cut_solution = cut_circuit(circuit=self.circuit,subcircuit_vertices=subcircuit_vertices,verbose=self.verbose)

        if len(cut_solution) > 0:
            source_folder = get_dirname(circuit_name=self.circuit_name,max_subcircuit_qubit=cut_solution['max_subcircuit_qubit'],
            eval_mode=None,num_threads=None,mem_limit=None,field='cutter')
            if os.path.exists(source_folder):
                subprocess.run(['rm','-r',source_folder])
            os.makedirs(source_folder)
            cut_solution['circuit_name'] = self.circuit_name
            pickle.dump(cut_solution, open('%s/cut_solution.pckl'%(source_folder),'wb'))
            self._generate_subcircuits(source_folder=source_folder,cut_solution=cut_solution)
            return source_folder
        else:
            return None
    
    def evaluate(self,source_folders,eval_mode,mem_limit,num_nodes,num_threads,ibmq):
        if self.verbose:
            print('*'*20,'evaluation mode = %s'%(eval_mode),'*'*20,flush=True)
        self.source_folders = source_folders
        
        subprocess.run(['rm','./cutqc/build'])
        build_command = 'gcc ./cutqc/build.c -L /opt/intel/oneapi/mkl/latest/lib/intel64 -I /opt/intel/oneapi/mkl/2021.2.0/include -I /opt/intel/mkl/include/ -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -DMKL_ILP64 -m64 -o ./cutqc/build'
        subprocess.run(build_command.split(' '))

        circ_dict, all_subcircuit_entries_sampled = self._gather_subcircuits(eval_mode=eval_mode)
        subcircuit_results = self._run_subcircuits(circ_dict=circ_dict,eval_mode=eval_mode,ibmq=ibmq)
        self._attribute_shots(subcircuit_results=subcircuit_results,eval_mode=eval_mode,all_subcircuit_entries_sampled=all_subcircuit_entries_sampled)
        dest_folders = self._build(eval_mode=eval_mode,mem_limit=mem_limit,num_nodes=num_nodes,num_threads=num_threads)
        return dest_folders

    def verify(self, source_folders, dest_folders):
        print('*'*20,'Verify','*'*20,flush=True)
        row_format = '{:<20} {:<10} {:<30}'
        print(row_format.format('Circuit Name','QPU','Error'),flush=True)
        for source_folder, dest_folder in zip(source_folders,dest_folders):
            cut_solution = read_dict(filename='%s/cut_solution.pckl'%source_folder)
            circuit = cut_solution['circuit']
            circuit_name = cut_solution['circuit_name']
            complete_path_map = cut_solution['complete_path_map']
            subcircuits = cut_solution['subcircuits']
            max_subcircuit_qubit = cut_solution['max_subcircuit_qubit']
            summation_terms = pickle.load(open('%s/summation_terms.pckl'%source_folder,'rb'))
            smart_order = [x[0] for x in summation_terms[0]]

            build_output = read_dict(filename='%s/build_output.pckl'%dest_folder)
            reconstructed_prob = build_output['reconstructed_prob']
            eval_mode = build_output['eval_mode']
            
            squared_error = verify(full_circuit=circuit,unordered=reconstructed_prob,complete_path_map=complete_path_map,subcircuits=subcircuits,smart_order=smart_order)
            print(row_format.format(circuit_name,eval_mode,'%.1e'%squared_error),flush=True)
    
    def _generate_subcircuits(self,source_folder,cut_solution):
        '''
        Generate subcircuit variations and the summation terms
        '''
        full_circuit = cut_solution['circuit']
        subcircuits = cut_solution['subcircuits']
        complete_path_map = cut_solution['complete_path_map']
        counter = cut_solution['counter']

        subcircuit_instances, subcircuit_instances_idx = generate_subcircuit_instances(subcircuits=subcircuits,complete_path_map=complete_path_map)
        summation_terms, subcircuit_entries, subcircuit_instance_attribution = generate_summation_terms(full_circuit=full_circuit,subcircuits=subcircuits,complete_path_map=complete_path_map,subcircuit_instances_idx=subcircuit_instances_idx,counter=counter)

        pickle.dump(subcircuit_instances, open('%s/subcircuit_instances.pckl'%(source_folder),'wb'))
        pickle.dump(subcircuit_instances_idx, open('%s/subcircuit_instances_idx.pckl'%(source_folder),'wb'))
        pickle.dump(subcircuit_instance_attribution, open('%s/subcircuit_instance_attribution.pckl'%(source_folder),'wb'))
        pickle.dump(summation_terms, open('%s/summation_terms.pckl'%(source_folder),'wb'))
        pickle.dump(subcircuit_entries, open('%s/subcircuit_entries.pckl'%(source_folder),'wb'))

        if self.verbose:
            print('--> %s subcircuit_instances:'%self.circuit_name,flush=True)
            row_format = '{:<15} {:<15} {:<10} {:<30} {:<30}'
            print(row_format.format('subcircuit','instance_idx','#shots','init','meas'),flush=True)
            for subcircuit_idx in subcircuit_instances:
                for subcircuit_instance_idx in subcircuit_instances[subcircuit_idx]:
                    circuit = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['circuit']
                    shots = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['shots']
                    init = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['init']
                    meas = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['meas']
                    print(row_format.format(subcircuit_idx,subcircuit_instance_idx,shots,str(init)[:30],str(meas)[:30]))
                print('-'*20,flush=True)

            print('--> %s subcircuit_entries:'%self.circuit_name,flush=True)
            row_format = '{:<30} {:<30}'
            for subcircuit_idx in subcircuit_entries:
                print(row_format.format('subcircuit_%d_entry_idx'%subcircuit_idx,'kronecker term (coeff, instance)'),flush=True)
                ctr = 0
                for subcircuit_entry_idx in subcircuit_entries[subcircuit_idx]:
                    if type(subcircuit_entry_idx) is int:
                        ctr += 1
                        if ctr<=10:
                            print(row_format.format(subcircuit_entry_idx,str(subcircuit_entries[subcircuit_idx][subcircuit_entry_idx])[:30]))
                print('... Total %d subcircuit entries\n'%ctr,flush=True)
        
            print('--> %s subcircuit_instance_attribution:'%self.circuit_name,flush=True)
            row_format = '{:<30} {:<50}'
            for subcircuit_idx in subcircuit_instance_attribution:
                print(row_format.format('subcircuit_%d_instance_idx'%subcircuit_idx,'coefficient, subcircuit_entry_idx'),flush=True)
                ctr = 0
                for subcircuit_instance_idx in subcircuit_instance_attribution[subcircuit_idx]:
                    ctr += 1
                    if ctr>10:
                        break
                    print(row_format.format(subcircuit_instance_idx,str(subcircuit_instance_attribution[subcircuit_idx][subcircuit_instance_idx])[:50]),flush=True)
                print('... Total %d subcircuit instances to attribute\n'%len(subcircuit_instance_attribution[subcircuit_idx]))
        
            print('--> %s summation_terms:'%self.circuit_name)
            row_format = '{:<10}'*len(subcircuits)
            for summation_term in summation_terms[:10]:
                row = []
                for subcircuit_entry in summation_term:
                    subcircuit_idx, subcircuit_entry_idx = subcircuit_entry
                    row.append('%d,%d'%(subcircuit_idx,subcircuit_entry_idx))
                print(row_format.format(*row))
            print('... Total %d summations\n'%len(summation_terms),flush=True)

    def _gather_subcircuits(self,eval_mode):
        circ_dict = {}
        all_subcircuit_entries_sampled = {}
        for source_folder in self.source_folders:
            cut_solution = read_dict(filename='%s/cut_solution.pckl'%source_folder)
            subcircuit_instances = read_dict(filename='%s/subcircuit_instances.pckl'%source_folder)
            summation_terms = pickle.load(open('%s/summation_terms.pckl'%source_folder,'rb'))
            subcircuit_entries = read_dict(filename='%s/subcircuit_entries.pckl'%source_folder)
            max_subcircuit_qubit = cut_solution['max_subcircuit_qubit']
            circuit_name = cut_solution['circuit_name']
            
            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            num_threads=None,eval_mode=eval_mode,mem_limit=None,field='evaluator')
            if os.path.exists(eval_folder):
                subprocess.run(['rm','-r',eval_folder])
            os.makedirs(eval_folder)
            
            summation_terms_sampled = dummy_sample(summation_terms=summation_terms)
            subcircuit_entries_sampled = get_subcircuit_entries_sampled(summation_terms=summation_terms_sampled)
            
            all_subcircuit_entries_sampled[circuit_name] = subcircuit_entries_sampled
            subcircuit_instances_sampled = get_subcircuit_instances_sampled(subcircuit_entries=subcircuit_entries,subcircuit_entry_samples=subcircuit_entries_sampled)
            for subcircuit_instance in subcircuit_instances_sampled:
                subcircuit_idx, subcircuit_instance_idx = subcircuit_instance
                parent_subcircuit_instance_idx = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['parent']
                circuit = subcircuit_instances[subcircuit_idx][parent_subcircuit_instance_idx]['circuit']
                shots = subcircuit_instances[subcircuit_idx][parent_subcircuit_instance_idx]['shots']
                init = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['init']
                meas = subcircuit_instances[subcircuit_idx][subcircuit_instance_idx]['meas']
                circ_dict_key = (circuit_name,subcircuit_idx,parent_subcircuit_instance_idx)
                if circ_dict_key in circ_dict:
                    assert circ_dict[circ_dict_key]['init'] == init
                    circ_dict[circ_dict_key]['meas'].append(meas)
                else:
                    circ_dict[circ_dict_key] = {
                        'circuit':circuit,
                        'shots':shots,
                        'init':init,
                        'meas':[meas]}
            pickle.dump(summation_terms_sampled, open('%s/summation_terms_sampled.pckl'%(eval_folder),'wb'))
        return circ_dict, all_subcircuit_entries_sampled
    
    def _run_subcircuits(self,circ_dict,eval_mode,ibmq):
        '''
        Run all the subcircuits
        '''
        if self.verbose:
            print('--> Running Subcircuits',flush=True)
            print('%d total'%len(circ_dict),flush=True)
        # print('circ_dict:')
        # for key in circ_dict:
        #     print(key,circ_dict[key])
        if eval_mode=='sv' or eval_mode=='qasm' or eval_mode=='runtime':
            subcircuit_results = {}
            for key in circ_dict:
                circuit_name, subcircuit_result = simulate_subcircuit(key=key,subcircuit_info=circ_dict[key],eval_mode=eval_mode)
                if circuit_name in subcircuit_results:
                    subcircuit_results[circuit_name].update(subcircuit_result)
                else:
                    subcircuit_results[circuit_name] = subcircuit_result
        elif 'ibmq' in eval_mode:
            subcircuit_results = {}
            scheduler = Scheduler(circ_dict=circ_dict,verbose=True)
            # scheduler.add_ibmq(token=ibmq['token'],hub=ibmq['hub'],group=ibmq['group'],project=ibmq['project'])
            # scheduler.submit_ibmq_jobs(device_names=[eval_mode],transpilation=True,real_device=False)
            # scheduler.retrieve_jobs(force_prob=True,save_memory=False,save_directory=None)
            # NOTE: use noiseless simulation for demonstration
            scheduler.run_simulation_jobs(device_name='noiseless')
            for key in scheduler.circ_dict:
                subcircuit_info=scheduler.circ_dict[key]
                circuit_name, subcircuit_idx, parent_subcircuit_instance_idx = key
                init = subcircuit_info['init']
                meas = subcircuit_info['meas']
                counts = subcircuit_info['noiseless|sim']
                for m in meas:
                    measured_prob = measure_prob(unmeasured_prob=counts,meas=m)
                    if circuit_name in subcircuit_results:
                        subcircuit_results[circuit_name][(subcircuit_idx,init,m)] = measured_prob
                    else:
                        subcircuit_results[circuit_name] = {(subcircuit_idx,init,m): measured_prob}
        else:
            raise NotImplementedError
        # print('results returned:')
        # for circuit_name in subcircuit_results:
        #     print('circuit_name =',circuit_name)
        #     for key in subcircuit_results[circuit_name]:
        #         print(key,'probability output =',subcircuit_results[circuit_name][key])
        return subcircuit_results
    
    def _attribute_shots(self,subcircuit_results,eval_mode,all_subcircuit_entries_sampled):
        '''
        Attribute the shots into respective subcircuit entries
        '''
        row_format = '{:<15} {:<15} {:<25} {:<30}'
        if self.verbose:
            print('--> Attribute shots',flush=True)
            print(row_format.format('circuit_name','subcircuit_idx','subcircuit_instance_idx','coefficient, subcircuit_entry_idx'),flush=True)
        for source_folder in self.source_folders:
            ctr = 0
            subcircuit_entry_probs = {}
            cut_solution = read_dict(filename='%s/cut_solution.pckl'%source_folder)
            subcircuit_instances_idx = read_dict(filename='%s/subcircuit_instances_idx.pckl'%source_folder)
            subcircuit_instance_attribution = read_dict(filename='%s/subcircuit_instance_attribution.pckl'%source_folder)
            max_subcircuit_qubit = cut_solution['max_subcircuit_qubit']
            circuit_name = cut_solution['circuit_name']
            subcircuit_entries_sampled = all_subcircuit_entries_sampled[circuit_name]
            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            eval_mode=eval_mode,num_threads=None,mem_limit=None,field='evaluator')

            for key in subcircuit_results[circuit_name]:
                ctr += 1
                subcircuit_idx, init, meas = key
                subcircuit_instance_idx = subcircuit_instances_idx[subcircuit_idx][(init,meas)]
                subcircuit_instance_prob = subcircuit_results[circuit_name][key]
                attributions = subcircuit_instance_attribution[subcircuit_idx][subcircuit_instance_idx]
                if self.verbose and ctr<=10:
                    print(row_format.format(circuit_name,subcircuit_idx,subcircuit_instance_idx,str(attributions)[:30]),flush=True)

                for item in attributions:
                    coefficient, subcircuit_entry_idx = item
                    if (subcircuit_idx,subcircuit_entry_idx) not in subcircuit_entries_sampled:
                        continue
                    subcircuit_entry_prob_key = (eval_folder,subcircuit_idx,subcircuit_entry_idx)
                    if subcircuit_entry_prob_key in subcircuit_entry_probs:
                        subcircuit_entry_probs[subcircuit_entry_prob_key] += coefficient*subcircuit_instance_prob
                    else:
                        subcircuit_entry_probs[subcircuit_entry_prob_key] = coefficient*subcircuit_instance_prob
            for subcircuit_entry_prob_key in subcircuit_entry_probs:
                subcircuit_entry_prob = subcircuit_entry_probs[subcircuit_entry_prob_key]
                eval_folder,subcircuit_idx,subcircuit_entry_idx = subcircuit_entry_prob_key
                subcircuit_entry_file = open('%s/%d_%d.txt'%(eval_folder,subcircuit_idx,subcircuit_entry_idx),'w')
                [subcircuit_entry_file.write('%e '%x) for x in subcircuit_entry_prob]
                subcircuit_entry_file.close()
            if self.verbose:
                print('... Total %d subcircuit results attributed\n'%ctr,flush=True)
    
    def _build(self, eval_mode, mem_limit, num_nodes, num_threads):
        if self.verbose:
            print('--> Build')
            row_format = '{:<15} {:<20} {:<30}'
            print(row_format.format('circuit_name','summation_term_idx','summation_term'))
        dest_folders = []
        for source_folder in self.source_folders:
            cut_solution = read_dict(filename='%s/cut_solution.pckl'%source_folder)
            max_subcircuit_qubit = cut_solution['max_subcircuit_qubit']
            circuit_name = cut_solution['circuit_name']
            summation_terms = pickle.load(open('%s/summation_terms.pckl'%source_folder,'rb'))
            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            eval_mode=eval_mode,num_threads=None,mem_limit=None,field='evaluator')
            summation_terms_sampled = pickle.load(open('%s/summation_terms_sampled.pckl'%eval_folder,'rb'))
            
            if self.verbose:
                [print(row_format.format(circuit_name,x['summation_term_idx'],str(x['summation_term'])[:30])) for x in summation_terms_sampled[:10]]
                print('... Total %d summation terms sampled\n'%len(summation_terms_sampled))
            full_circuit = cut_solution['circuit']
            subcircuits = cut_solution['subcircuits']
            complete_path_map = cut_solution['complete_path_map']
            counter = cut_solution['counter']

            dest_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            eval_mode=eval_mode,num_threads=num_threads,mem_limit=mem_limit,field='build')
            dest_folders.append(dest_folder)
            if os.path.exists(dest_folder):
                subprocess.run(['rm','-r',dest_folder])
            os.makedirs(dest_folder)

            '''
            TODO: handle the new summation_terms_sampled format
            1. Get rid of repeated summation term computations
            '''
            _build_begin = time.time()
            num_samples = 1
            child_processes = []
            for rank in range(num_threads):
                rank_summation_terms = find_process_jobs(jobs=summation_terms_sampled,rank=rank,num_workers=num_threads)
                build_command = './cutqc/build %d %s %s %d %d %d %d %d'%(
                    rank,eval_folder,dest_folder,int(2**full_circuit.num_qubits),cut_solution['num_cuts'],len(rank_summation_terms),len(subcircuits),num_samples)
                build_command_file = open('%s/build_command_%d.txt'%(dest_folder,rank),'w')
                for rank_summation_term in rank_summation_terms:
                    build_command_file.write('%e '%rank_summation_term['sampling_prob'])
                    build_command_file.write('%d '%rank_summation_term['frequency'])
                    for item in rank_summation_term['summation_term']:
                        subcircuit_idx, subcircuit_entry_idx = item
                        build_command_file.write('%d %d %d '%(subcircuit_idx,subcircuit_entry_idx,int(2**counter[subcircuit_idx]['effective'])))
                build_command_file.close()
                p = subprocess.Popen(args=build_command.split(' '))
                child_processes.append(p)
            for rank in range(num_threads):
                cp = child_processes[rank]
                cp.wait()
            _build_time = time.time()-_build_begin
            
            time.sleep(1)
            elapsed = []
            reconstructed_prob = None
            for rank in range(num_threads):
                rank_logs = open('%s/rank_%d_summary.txt'%(dest_folder,rank), 'r')
                lines = rank_logs.readlines()
                assert lines[-2].split(' = ')[0]=='Total build time' and lines[-1] == 'DONE\n'
                elapsed.append(float(lines[-2].split(' = ')[1]))

                fp = open('%s/build_%d.txt'%(dest_folder,rank), 'r')
                for i, line in enumerate(fp):
                    rank_reconstructed_prob = line.split(' ')[:-1]
                    rank_reconstructed_prob = np.array(rank_reconstructed_prob)
                    rank_reconstructed_prob = rank_reconstructed_prob.astype(np.float)
                    if i>0:
                        raise Exception('C build_output should not have more than 1 line')
                fp.close()
                subprocess.run(['rm','%s/build_%d.txt'%(dest_folder,rank)])
                if isinstance(reconstructed_prob,np.ndarray):
                    reconstructed_prob += rank_reconstructed_prob
                else:
                    reconstructed_prob = rank_reconstructed_prob
            elapsed = np.array(elapsed)
            if self.verbose:
                print('%s _build took %.3e seconds'%(circuit_name,_build_time),flush=True)
                print('Sampled %d/%d summation terms'%(len(summation_terms_sampled),len(summation_terms)))
            pickle.dump(
                {'reconstructed_prob':reconstructed_prob,
                'eval_mode':eval_mode,
                'num_summation_terms_sampled':len(summation_terms_sampled),
                'num_summation_terms':len(summation_terms)
                },open('%s/build_output.pckl'%(dest_folder),'wb'))
        return dest_folders
