#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tigramite causal discovery for time series: Parallization script implementing 
the PCMCI method based on mpi4py. 

Parallelization is done across variables j for both the PC condition-selection
step and the MCI step.
"""

# Author: Jakob Runge <jakobrunge@posteo.de>
#
# License: GNU General Public License v3.0


from mimetypes import init
from turtle import back

from attr import attr
from tabnanny import verbose
from mpi4py import MPI
import numpy as np
import pandas as pd
import os, sys, pickle
import statistics
import pprint
import time

from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.toymodels import structural_causal_processes as toys
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb

import src.event_processing as evt_proc
import src.background_generator as bk_generator
import src.causal_evaluation as causal_eval
import src.security_guard as security_guard

"""Parameter Settings"""

# Default communicator
COMM = MPI.COMM_WORLD
NORMAL = 0
ABNORMAL = 1

TEST_PARAM_SETTING = True
PARAM_SETTING = True
DATA_PREPROCESSING = True
BACKGROUND_GENERATION = True
partition_config = int(sys.argv[1])
apply_bk = int(sys.argv[2])

if TEST_PARAM_SETTING:
    single_frame_test_flag = 1 # JC TEST: Test for single dataframe
    skip_skeleton_estimation_flag = 0 # JC TEST: Skip the causal discovery algorithm
    skip_bayesian_fitting_flag = 0
    num_anomalies = 0
    max_prop_length = 1

if PARAM_SETTING:
    dataset = 'hh101'
    stable_only = 1
    cond_ind_test = CMIsymb()
    tau_max = 1; tau_min = 1
    verbosity = -1 # -1: No debugging information; 0: Debugging information in this module; 2: Debugging info in PCMCI class; 3: Debugging info in CIT implementations
    ## For stable-pc
    max_n_edges = 5
    pc_alpha = 0.1
    max_conds_dim = 5
    maximum_comb = 1
    ## For MCI
    alpha_level = 0.01
    max_conds_px = 5; max_conds_py= 5
    ## For anomaly detection
    sig_level = 0.9
    ## Resulting dict
    pc_result_dict = {}; mci_result_dict = {}
    # For evaluations
    record_count_list =[]
    pc_time_list = []; mci_time_list = []

def _split(container, count):
    """
    Simple function splitting a range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]

def _run_pc_stable_parallel(j, dataframe, cond_ind_test, selected_links,\
    tau_min=1, tau_max=1, pc_alpha = 0.1,verbosity=0,\
    maximum_comb = None, max_conds_dim=None):
    """Wrapper around PCMCI.run_pc_stable estimating the parents for a single 
    variable j.

    Parameters
    ----------
    j : int
        Variable index.

    Returns
    -------
    j, pcmci_of_j, parents_of_j : tuple
        Variable index, PCMCI object, and parents of j
    """

    pcmci_of_j = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    parents_of_j = pcmci_of_j.run_pc_stable(
        selected_links=selected_links[j],
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_combinations=maximum_comb,
        max_conds_dim=max_conds_dim
    )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j

def _run_mci_parallel(j, pcmci_of_j, all_parents, selected_links,\
    tau_min=1, tau_max=1, alpha_level = 0.01,\
    max_conds_px = None, max_conds_py=None):
    """Wrapper around PCMCI.run_mci step.
    Parameters
    ----------
    j : int
        Variable index.

    pcmci_of_j : object
        PCMCI object for variable j. This may contain pre-computed results 
        (such as residuals or null distributions).

    all_parents : dict
        Dictionary of parents for all variables. Needed for MCI independence
        tests.

    Returns
    -------
    j, results_in_j : tuple
        Variable index and results dictionary containing val_matrix, p_matrix,
        and optionally conf_matrix with non-zero entries only for
        matrix[:,j,:].
    """
    results_in_j = pcmci_of_j.run_mci(
        selected_links=selected_links[j],
        tau_min=tau_min,
        tau_max=tau_max,
        parents=all_parents,
        alpha_level=alpha_level,
        max_conds_px=max_conds_px,
        max_conds_py = max_conds_py
    )

    return j, results_in_j

def _lag_name(attr:'str', lag:'int'):
    assert(lag >= 0)
    new_name = '{}({})'.format(attr, -1 * lag) if lag > 0 else '{}'.format(attr)
    return new_name

class BayesianFitter:

    def __init__(self, frame, tau_max, link_dict) -> None:
        self.frame = frame
        self.tau_max = tau_max
        self.dataframe = self.frame['training-data']; self.var_names = self.dataframe.var_names; self.n_vars = len(self.var_names)
        self.expanded_var_names, self.expanded_causal_graph, self.expanded_data_array =\
                     self._transform_materials(tau_max, link_dict)
        self.n_expanded_vars = len(self.expanded_var_names)
        self.model = None
    
    def _transform_materials(self, tau_max, link_dict):
        """
        This function transforms the original N variables into N(tau_max + 1) variables
        As a result,
            1. the resulting number of variables equals to N(tau_max + 1)
            2. The resulting causal graph is a N(tau_max + 1) * N(tau_max + 1) binary array
            3. The resulting data array is of shape (T - tau_max) * N(tau_max + 1)
        Args:
            dataframe (_type_): The original data frame
            tau_max (_type_): User-specified maximum time lag
            link_dict (_type_): Results returned by pc-stable algorithm, which is a dict recording the edges
        Returns:
            expanded_var_names (list[str]): Record the name of expanded variables (starting from t-tau_max to t)
        """ 
        expanded_var_names = []; expanded_causal_graph = None; expanded_data_array = None
        for tau in range(0, tau_max + 1): # Construct expanded_var_names
            expanded_var_names = [*[_lag_name(x, tau) for x in self.var_names], *expanded_var_names]
        expanded_causal_graph = np.zeros(shape=(len(expanded_var_names), len(expanded_var_names)), dtype=np.uint8)
        for outcome, cause_list in link_dict.items(): # Construct expanded causal graph (a binary array)
            for (cause, lag) in cause_list:
                expanded_causal_graph[expanded_var_names.index(_lag_name(cause, abs(lag))), expanded_var_names.index(outcome)] = 1
        expanded_data_array = np.zeros(shape=(self.dataframe.T - tau_max, len(expanded_var_names)), dtype=np.uint8)
        for i in range(0, self.dataframe.T - tau_max): # Construct expanded data array
            expanded_data_array[i] = np.concatenate([self.dataframe.values[i+tau] for tau in range(0, tau_max+1)])
        return expanded_var_names, expanded_causal_graph, expanded_data_array

    def construct_bayesian_model(self):
        """Construct a parameterized causal graph (i.e., a bayesian model)

        Returns:
            model: A pgmpy.model object
        """
        edge_list = [(self.expanded_var_names[i], self.expanded_var_names[j])\
                        for (i, j), x in np.ndenumerate(self.expanded_causal_graph) if x == 1]
        in_degrees = [sum(self.expanded_causal_graph[:, i]) for i in range(0, self.n_expanded_vars)]; max_degree = max(in_degrees); corrs_attr = self.expanded_var_names[in_degrees.index(max_degree)]
        if max_degree > 10:
            print("[Bayesian Fitting] ALERT! The variable {} owns the maximum in-degree {} (larger than 10). This variable may slow down the fitting process!".format(corrs_attr, max_degree))
        self.model = BayesianNetwork(edge_list)
        df = pd.DataFrame(data=self.expanded_data_array, columns=self.expanded_var_names)
        #cpd = MaximumLikelihoodEstimator(self.model, df).estimate_cpd(corrs_attr) # JC TEST: The bayesian fitting consumes much time. Let's test the exact consumed time here..
        #print(cpd)
        self.model.fit(df, estimator= BayesianEstimator) 
    
    def predict_attr_state(self, attr, parent_state_dict, verbose=0):
        """ Predict the value of the target attribute given its parent states, i.e., E[attr|par(attr)]

        Args:
            attr (str): The target attribute
            parent_state_dict (dict[str, int]): The dictionary recording the name and state for each parent of attr

        Returns:
            val: The estimated state of the attribute
        """
        val = 0.0
        if verbose == 1:
            print(self.model.get_cpds(attr))
        phi = self.model.get_cpds(attr).to_factor()
        state_dict = parent_state_dict.copy()
        for possible_val in [0, 1]: # In our case, each attribute is a binary variable. Therefore the state space is [0, 1]
            state_dict[attr] = possible_val
            val += possible_val * phi.get_value(**state_dict) * 1.0
        return val

    def get_expanded_parent_indices(self, expanded_attr_index: 'int'):
        return list(np.where(self.expanded_causal_graph[:,expanded_attr_index] == 1)[0])
        #return {index: self.expanded_var_names[i] for index in par_indices}

    def analyze_discovery_statistics(self):
        print("[Bayesian Fitting] Analyzing discovery statistics.")
        incoming_degree_dict = {var_name: sum(self.expanded_causal_graph[:,self.expanded_var_names.index(var_name)])\
                                for var_name in self.var_names}
        outcoming_degree_dict = {}
        for var_name in self.var_names:
            outcoming_degree = 0
            for tau in range(1, self.tau_max + 1):
                outcoming_degree += sum(self.expanded_causal_graph[self.expanded_var_names.index(_lag_name(var_name, tau))])
            outcoming_degree_dict[var_name] = outcoming_degree
        
        nointeraction_attr_list = []
        for var_name in self.var_names:
            var_index =  self.expanded_var_names.index(var_name)
            parents_index = [k for k in range(self.n_expanded_vars) if self.expanded_causal_graph[k, var_index] > 0]
            if all([ (p - var_index) % self.n_vars == 0 for p in parents_index]):
                nointeraction_attr_list.append(var_name)

        
        isolated_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] + outcoming_degree_dict[var_name] == 0]
        exogenous_attr_list = [var_name for (var_name, count) in incoming_degree_dict.items() if count == 0 and var_name not in isolated_attr_list]
        stop_attr_list = [var_name for (var_name, count) in outcoming_degree_dict.items() if count == 0 and var_name not in isolated_attr_list]
        
        outcoming_degree_list = list(outcoming_degree_dict.values()); incomming_degree_list = list(incoming_degree_dict.values())

        str = " * isolated attrs, #: {}, {}\n".format(isolated_attr_list, len(isolated_attr_list))\
            + " * stop attrs, #: {}, {}\n".format(stop_attr_list, len(stop_attr_list))\
            + " * exogenous attrs, #: {}, {}\n".format(exogenous_attr_list, len(exogenous_attr_list))\
            + " * no-interaction attrs, #: {}, {}\n".format(nointeraction_attr_list, len(nointeraction_attr_list))\
            + " * # edges: {}\n".format(np.sum(self.expanded_causal_graph))\
            + " * (max, mean, min) for outcoming degrees: ({}, {}, {})\n".format(max(outcoming_degree_list),\
                        sum(outcoming_degree_list)*1.0/(self.n_vars - len(isolated_attr_list)), min(outcoming_degree_list))\
            + " * (max, mean, min) for incoming degrees: ({}, {}, {})\n".format(max(incomming_degree_list),\
                        sum(incomming_degree_list)*1.0/(self.n_vars - len(isolated_attr_list)), min(incomming_degree_list))
        print(str)

        self.nointeraction_attr_list = nointeraction_attr_list
        self.isolated_attr_list = isolated_attr_list
        self.exogenous_attr_list = exogenous_attr_list
        self.stop_attr_list = stop_attr_list

"""Event preprocessing"""
if DATA_PREPROCESSING:
    event_preprocessor = evt_proc.Hprocessor(dataset)
    attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config, training_ratio=0.8)

"""Background Generator"""
if BACKGROUND_GENERATION:
    background_generator = bk_generator.BackgroundGenerator(dataset, event_preprocessor, partition_config, tau_max)

"""Interaction Miner and Runtime Security Monitor"""
for frame_id in range(event_preprocessor.frame_count):
    frame = event_preprocessor.frame_dict[frame_id]; dataframe = frame['training-data']
    """Causal Graph Skeleton Construction."""
    start = time.time()
    if not skip_skeleton_estimation_flag:
        T = dataframe.T; N = dataframe.N
        record_count_list.append(T)
        selected_variables = list(range(N))
        splitted_jobs = None
        results = []
        selected_links = background_generator.generate_candidate_interactions(apply_bk, frame_id, N) # Get candidate interactions
        # 1. Paralleled Discovery Engine
        if COMM.rank == 0: # Assign selected_variables into whatever cores are available.
            splitted_jobs = _split(selected_variables, COMM.size)
        scattered_jobs = COMM.scatter(splitted_jobs, root=0)
        pc_start = time.time()
        for j in scattered_jobs: # Each process calls stable-pc algorithm to infer the edges
            (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links,\
                                                                tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                                max_conds_dim=max_conds_dim, verbosity=verbosity, maximum_comb=maximum_comb)
            filtered_parents_of_j = parents_of_j.copy()
            n_edges = min(len(filtered_parents_of_j[j]), max_n_edges) # Only select top 5 causal edges with maximum statistic values (MCI)
            if n_edges > 0:
                filtered_parents_of_j[j] = filtered_parents_of_j[j][0: n_edges]
            results.append((j, pcmci_of_j, filtered_parents_of_j))
            #results.append((j, pcmci_of_j, parents_of_j))
        results = MPI.COMM_WORLD.gather(results, root=0)
        pc_end = time.time()
        if COMM.rank == 0: # The root node gathers the result and generate the interaction graph.
            all_parents = {}
            pcmci_objects = {}
            for res in results:
                for (j, pcmci_of_j, parents_of_j) in res:
                    all_parents[j] = parents_of_j[j]
                    pcmci_objects[j] = pcmci_of_j
            all_parents_with_name = {}
            for outcome_id, cause_list in all_parents.items():
                all_parents_with_name[attr_names[outcome_id]] = [(attr_names[cause_id],lag) for (cause_id, lag) in cause_list]
            pc_result_dict[frame_id] = all_parents_with_name; pc_time_list.append((pc_end - pc_start) * 1.0 / 60)
            if verbosity > -1:
                print("##\n## PC-stable discovery for frame {} finished. Consumed time: {} mins\n##".format(frame_id, (pc_end - pc_start) * 1.0 / 60))
        if stable_only == 0: # Each process further calls the MCI procedure (if needed)
            if COMM.rank == 0: # First distribute the gathered pc results to each process
                for i in range(1, COMM.size):
                    COMM.send((all_parents, pcmci_objects), dest=i)
            else:
                (all_parents, pcmci_objects) = COMM.recv(source=0)
            scattered_jobs = COMM.scatter(splitted_jobs, root=0)
            results = []
            mci_start = time.time()
            """Each process calls MCI algorithm."""
            for j in scattered_jobs:
                (j, results_in_j) = _run_mci_parallel(j, pcmci_objects[j], all_parents, selected_links=selected_links,\
                                                tau_min=tau_min, tau_max=tau_max, alpha_level=alpha_level,\
                                                max_conds_px = max_conds_px, max_conds_py=max_conds_py)
                results.append((j, results_in_j))
            # The root node merge the result and generate final results
            results = MPI.COMM_WORLD.gather(results, root=0)
            if COMM.rank == 0:
                all_results = {}
                for res in results:
                    for (j, results_in_j) in res:
                        for key in results_in_j.keys():
                            if results_in_j[key] is None:  
                                all_results[key] = None
                            else:
                                if key not in all_results.keys():
                                    if key == 'p_matrix':
                                        all_results[key] = np.ones(results_in_j[key].shape)
                                    else:
                                        all_results[key] = np.zeros(results_in_j[key].shape, dtype=results_in_j[key].dtype)
                                all_results[key][:, j, :] = results_in_j[key][:, j, :]
                p_matrix = all_results['p_matrix']
                val_matrix = all_results['val_matrix']
                conf_matrix = all_results['conf_matrix']
 
                sig_links = (p_matrix <= alpha_level)
                sorted_links_with_name = {}
                for j in selected_variables:
                    links = dict([((p[0], -p[1]), np.abs(val_matrix[p[0], j, abs(p[1])]))
                                for p in zip(*np.where(sig_links[:, j, :]))])
                    sorted_links = sorted(links, key=links.get, reverse=True)
                    sorted_links_with_name[attr_names[j]] = []
                    for p in sorted_links:
                        sorted_links_with_name[attr_names[j]].append((attr_names[p[0]], p[1]))
                mci_end = time.time()
                mci_result_dict[frame_id] = sorted_links_with_name; mci_time_list.append((mci_end - mci_start) * 1.0 / 60)
                if verbosity > -1:
                    print("##\n## MCI for frame {} finished. Consumed time: {} mins\n##".format(frame_id, (mci_end - mci_start) * 1.0 / 60))
    else:
        pc_result_dict[frame_id] = {'D001': [], 'D002': [('D002', -1), ('M001', -1)], 'LS001': [('LS001', -1), ('D002', -1), ('M008', -1), ('M001', -1)], 'LS002': [('M008', -1), ('M002', -1)], 'LS003': [('M008', -1)], 'LS004': [('LS016', -1), ('M008', -1)], 'LS005': [('M008', -1)], 'LS006': [('LS006', -1), ('M006', -1), ('M003', -1)], 'LS007': [], 'LS008': [('LS013', -1)], 'LS009': [('M008', -1)], 'LS010': [('LS001', -1),      ('M001', -1)], 'LS011': [('M001', -1), ('M008', -1)], 'LS012': [('M008', -1)], 'LS013': [('LS013', -1), ('M008', -1)], 'LS014': [('LS014', -1), ('LS009', -1), ('M009', -1), ('M008', -1)], 'LS015':        [('M011', -1)], 'LS016': [('M002', -1), ('M008', -1)], 'M001': [('D002', -1), ('M001', -1), ('M010', -1), ('M005', -1), ('LS001', -1)], 'M002': [('M002', -1), ('M004', -1), ('LS016', -1), ('LS002', -1),  ('M003', -1)], 'M003': [('LS006', -1), ('LS003', -1), ('M006', -1), ('M003', -1), ('M002', -1), ('M007', -1), ('M004', -1)], 'M004': [('M004', -1), ('M002', -1), ('M008', -1), ('M003', -1), ('LS016', -   1)], 'M005': [('M005', -1), ('M008', -1), ('M001', -1), ('M004', -1), ('M010', -1)], 'M006': [('M006', -1), ('LS006', -1), ('M003', -1)], 'M007': [('M007', -1), ('M003', -1)], 'M008': [('M008', -1),('M004', -1), ('T104', -1), ('M005', -1), ('LS013', -1), ('LS008', -1), ('LS005', -1)], 'M009': [('M009', -1), ('M012', -1), ('LS014', -1), ('LS009', -1)], 'M010': [('M010', -1), ('M001', -1), ('D002', -1), ('M005', -1)], 'M011': [('M011', -1), ('LS015', -1), ('M009', -  1), ('M001', -1)], 'M012': [('M009', -1), ('M012', -1), ('LS014', -1)], 'T101': [('T101', -1)], 'T102': [], 'T103': [], 'T104': [('T104', -1), ('M008', -1)], 'T105': [('T105', -1), ('M008', -1)]}

    """Causal Graph Parameterization."""
    if COMM.rank == 0:
        print("Skeleton construction completes. Consumed time: {} mins.".format((time.time() - start)*1.0/60))
        start = time.time()
        interaction_graph = pc_result_dict[frame_id] if stable_only == 1 else mci_result_dict[frame_id]
        pprint.pprint(interaction_graph)
        print("\n********** Initiate Bayesian Fitting. **********")
        bayesian_fitter = BayesianFitter(frame, tau_max, interaction_graph)
        bayesian_fitter.analyze_discovery_statistics()
        if not skip_bayesian_fitting_flag:
            bayesian_fitter.construct_bayesian_model()
        print("Bayesian fitting complete. Consumed time: {} mins.".format((time.time() - start)*1.0/60))
    
    """Security Guard."""
    if COMM.rank == 0:
        print("\n********** Initiate Security Guarding. **********")
        security_guard = security_guard.SecurityGuard(frame=frame, bayesian_fitter=bayesian_fitter, sig_level=sig_level)
        evaluator = causal_eval.Evaluator(dataset=dataset, event_processor=event_preprocessor, background_generator=background_generator,\
                                             bayesian_fitter = bayesian_fitter, tau_max=tau_max)
        print("[Security guarding] Testing log starting positions {} with score threshold {}.".format(frame['testing-start-index'] + 1, security_guard.score_threshold))
        # 1. Inject device anomalies
        testing_event_sequence, anomaly_positions, stable_states_dict = evaluator.simulate_malicious_control(frame=frame, n_anomaly=num_anomalies, maximum_length=max_prop_length)
        # 2. Initiate anomaly detection
        start = time.time()
        event_id = 0
        while event_id < len(testing_event_sequence):
            event = testing_event_sequence[event_id]
            report_to_user = False
            if event_id < tau_max:
                security_guard.initialize(event_id, event, frame['testing-data'].values[event_id])
            else:
                report_to_user = security_guard.score_anomaly_detection(event_id=event_id, event=event)

            # JC NOTE: Here we simulate a user involvement, which handles the reported anomalies as soon as it is reported.
            if event_id in anomaly_positions or report_to_user is True:
                security_guard.calibrate(event_id, stable_states_dict)
            event_id += 1

        print("[Security guarding] Anomaly detection completes for {} runtime events. Consumed time: {} mins.".format(event_id, (time.time() - start)*1.0/60))
        # 3. Evaluate the detection accuracy.
        print("[Security guarding] Evaluating the detection accuracy for state transition violations")
        violation_count_dict = {}; violation_event_ids = list(security_guard.violation_dict.keys())
        for violation_event_id, violation_point in security_guard.violation_dict.items():
            #print(" * Violation (event id, interaction, score) = ({}, {}, {})".format(violation_event_id, violation_point['interaction'], violation_point['anomaly-score']))
            violation_count_dict[violation_point['attr']] = 1 if violation_point['attr'] not in violation_count_dict.keys() else violation_count_dict[violation_point['attr']] + 1
            violation_count_dict = dict(sorted(violation_count_dict.items(), key=lambda item: item[1]))
        pprint.pprint(violation_count_dict)

        evaluator.evaluate_detection_accuracy(anomaly_positions, violation_event_ids)
        #print("[Security guarding] Evaluating the detection accuracy for state transition violations")
        #violation_interaction_dict = {}; violation_count_dict = {}
        #violation_event_ids = list(security_guard.violation_dict.keys())
        #for violation_event_id, violation_point in security_guard.violation_dict.items():
        #    #print(" * Violation (event id, interaction, score) = ({}, {}, {})".format(violation_event_id, violation_point['interaction'], violation_point['anomaly-score']))
        #    violation_count_dict[violation_point['interaction'][1]] = 0 if violation_point['interaction'][1] not in violation_count_dict.keys() else  violation_count_dict[violation_point['interaction'][1]] + 1
        #    violation_interaction_dict['->'.join(violation_point['interaction'])] = 1 if '->'.join(violation_point['interaction']) not in violation_interaction_dict.keys() else violation_interaction_dict['->'.join(violation_point['interaction'])] + 1
        #    violation_count_dict = dict(sorted(violation_count_dict.items(), key=lambda item: item[1]))
        #pprint.pprint(violation_count_dict)
        #pprint.pprint(violation_interaction_dict)
        #evaluator.evaluate_detection_accuracy(anomaly_starting_positions, violation_event_ids)

    frame_id += 1

    if single_frame_test_flag == 1:
        break
    if frame_id == event_preprocessor.frame_count - 1:
        break

"""Evaluate discovery accuracy and compare with ARM."""
#if COMM.rank == 0:
#    pc_avg_truth_count, pc_avg_precision, pc_avg_recall = evaluator.estimate_average_discovery_accuracy(1, pc_result_dict)
#    str = "**** Discovery results for partition_config = {}, bk = {} ****".format(partition_config, apply_bk) \
#          + "\n Algorithm parameters:"\
#          + "\n * # frames: {}".format(frame_id) \
#          + "\n * independence test = %s" % cond_ind_test.measure \
#          + "\n * tau_min = {}, tau_max = {}".format(tau_min, tau_max) \
#          + "\n * pc_alpha = {}, max_conds_dim = {}, max_comb = {}".format(pc_alpha, max_conds_dim, maximum_comb) \
#          + "\n * alpha_level = {}, max_conds_px = {}, max_conds_py = {}".format(alpha_level, max_conds_px, max_conds_py) \
#          + "\n Average counts of records: {}".format(statistics.mean(record_count_list)) \
#          + "\nstablePC evaluations: average time, truth-count, precision, recall = {}, {}, {}, {}".format(statistics.mean(pc_time_list), pc_avg_truth_count, pc_avg_precision, pc_avg_recall)
#    if stable_only == 0: # If the PCMCI is also initiated, show the evaluation result.
#        mci_avg_truth_count, mci_avg_precision, mci_avg_recall = evaluator.estimate_average_discovery_accuracy(1, mci_result_dict)
#        str += "\nMCI evaluations: average time, truth-count, precision, recall = {}, {}, {}, {}".format(statistics.mean(mci_time_list), mci_avg_truth_count, mci_avg_precision, mci_avg_recall)
#    print(str)