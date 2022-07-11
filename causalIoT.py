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


import os, sys, pickle
import statistics
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from time import time
from collections import defaultdict
from mpi4py import MPI

from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.toymodels import structural_causal_processes as toys
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb
from src.tigramite.tigramite import plotting as tp

import src.event_processing as evt_proc
import src.background_generator as bk_generator
import src.security_guard as security_guard
from src.event_processing import Hprocessor
from src.bayesian_fitter import BayesianFitter
from src.causal_evaluation import Evaluator
from src.genetic_type import DataFrame, AttrEvent, DevAttribute

"""Parameter Settings"""

# Default communicator
COMM = MPI.COMM_WORLD
NORMAL = 0
ABNORMAL = 1

TEST_PARAM_SETTING = True
PARAM_SETTING = True
dataset = sys.argv[1]
partition_config = int(sys.argv[2])
apply_bk = int(sys.argv[3])

if TEST_PARAM_SETTING:
    single_frame_test_flag = 1 # Whether only testing single dataframe
    autocorrelation_flag = True # Whether consider autocorrelations in structure identification
    skip_skeleton_estimation_flag = False # Whether skip the causal structure identification process (For speedup and testing)
    skip_bayesian_fitting_flag = 0
    num_anomalies = 0
    max_prop_length = 1

if PARAM_SETTING:
    training_ratio = 0.9
    stable_only = 1
    cond_ind_test = CMIsymb()
    tau_max = 4; tau_min = 1
    verbosity = -1 # -1: No debugging information; 0: Debugging information in this module; 2: Debugging info in PCMCI class; 3: Debugging info in CIT implementations
    ## For stable-pc
    max_n_edges = 50
    pc_alpha = 0.001
    max_conds_dim = 5
    maximum_comb = 10
    ## For MCI
    alpha_level = 0.01
    max_conds_px = 5; max_conds_py= 5
    ## For anomaly detection
    sig_level = 0.95
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

"""Data loading"""
event_preprocessor:'Hprocessor' = Hprocessor(dataset)
frame_dict:'dict[DataFrame]' = event_preprocessor.data_loading(partition_config=partition_config, training_ratio=training_ratio)

"""Background Generator"""
background_generator = bk_generator.BackgroundGenerator(dataset, event_preprocessor, partition_config, tau_max)

"""Causal Evaluator"""
evaluator = Evaluator(dataset, event_preprocessor, background_generator, None, tau_max)
evaluator.construct_golden_standard(filter_threshold=partition_config)
exit()

for frame_id in range(event_preprocessor.frame_count):

    """Parallel Interaction Miner"""
    # Result variables
    all_parents = {}
    pcmci_objects = {}
    pc_result_dict = {}

    # Auxillary variables
    frame: 'DataFrame' = event_preprocessor.frame_dict[frame_id]
    dataframe:pp.DataFrame = frame.training_dataframe; attr_names = frame.var_names

    start = time()
    if not skip_skeleton_estimation_flag:
        T = dataframe.T; N = dataframe.N
        record_count_list.append(T)
        selected_variables = list(range(N))
        splitted_jobs = None
        selected_links = background_generator.generate_candidate_interactions(apply_bk, frame_id, N, autocorrelation_flag=autocorrelation_flag) # Get candidate interactions
        results = []

        if COMM.rank == 0: # Assign selected_variables into whatever cores are available.
            splitted_jobs = _split(selected_variables, COMM.size)
        
        scattered_jobs = COMM.scatter(splitted_jobs, root=0)
        pc_start = time()
        for j in scattered_jobs: # Each process calls stable-pc algorithm to infer the edges
            (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links,\
                                                                tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                                max_conds_dim=max_conds_dim, verbosity=verbosity, maximum_comb=maximum_comb)
            filtered_parents_of_j = parents_of_j.copy()
            n_edges = min(len(filtered_parents_of_j[j]), max_n_edges) # Only select top max_n_edges causal edges with maximum MCI
            if n_edges > 0:
                filtered_parents_of_j[j] = filtered_parents_of_j[j][0: n_edges]
            results.append((j, pcmci_of_j, filtered_parents_of_j))
        results = MPI.COMM_WORLD.gather(results, root=0)
        pc_end = time()

        if COMM.rank == 0: # The root node gathers the result and generate the interaction graph.
            all_parents = {}
            for res in results:
                for (j, pcmci_of_j, parents_of_j) in res:
                    all_parents[j] = parents_of_j[j]
                    pcmci_objects[j] = pcmci_of_j
            all_parents_with_name = {}
            for outcome_id, cause_list in all_parents.items():
                all_parents_with_name[attr_names[outcome_id]] = [(attr_names[cause_id],lag) for (cause_id, lag) in cause_list]
            pc_result_dict[frame_id] = all_parents_with_name; pc_time_list.append((pc_end - pc_start) * 1.0 / 60)

        if stable_only == 0: # Each process further calls the MCI procedure (if needed)
            if COMM.rank == 0: # First distribute the gathered pc results to each process
                for i in range(1, COMM.size):
                    COMM.send((all_parents, pcmci_objects), dest=i)
            else:
                (all_parents, pcmci_objects) = COMM.recv(source=0)
            scattered_jobs = COMM.scatter(splitted_jobs, root=0)
            results = []
            mci_start = time()
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
                mci_end = time()
                mci_result_dict[frame_id] = sorted_links_with_name; mci_time_list.append((mci_end - mci_start) * 1.0 / 60)
                if verbosity > -1:
                    print("##\n## MCI for frame {} finished. Consumed time: {} mins\n##".format(frame_id, (mci_end - mci_start) * 1.0 / 60))
    else:
        pc_result_dict[frame_id] = {}
    end = time()

    if COMM.rank == 0 and not skip_skeleton_estimation_flag : # Plot the graph if the PC procedure is not skipped.
        print("Parallel Interaction Mining finished. Consumed time: {} minutes".format((end - start)*1.0/60))
        answer_shape = (dataframe.N, dataframe.N, tau_max + 1)
        graph = np.zeros(answer_shape, dtype='<U3'); val_matrix = np.zeros(answer_shape); p_matrix = np.zeros(answer_shape)
        for j in scattered_jobs:
            pcmci_object:'PCMCI' = pcmci_objects[j]
            local_val_matrix = pcmci_object.results['val_matrix']; local_p_matrix = pcmci_object.results['p_matrix']; local_graph_matrix = pcmci_object.results['graph']
            print("Job {}'s p-matrix: {}".format(j, local_p_matrix))
            assert(all([x == 0 for x in val_matrix[local_val_matrix > 0]])); val_matrix += local_val_matrix
            assert(all([x == 0 for x in p_matrix[local_p_matrix > 0]])); p_matrix += local_p_matrix
            assert(all([x == '' for x in graph[local_graph_matrix != '']])); graph[local_graph_matrix != ''] = local_graph_matrix[local_graph_matrix != '']
            print("Current summary p-matrix: {}".format(p_matrix))
        tp.plot_time_series_graph(
            figsize=(6, 4),
            val_matrix=val_matrix,
            graph=graph,
            var_names= dataframe.var_names,
            link_colorbar_label='MCI'
        )
        plt.savefig("temp/image/{}cmi_test_tau{}.pdf".format(dataset, tau_max))
    exit()

    """CPT Estimator."""
    if COMM.rank == 0:
        print("Skeleton construction completes. Consumed time: {} mins.".format((time() - start)*1.0/60))
        start = time()
        interaction_graph = pc_result_dict[frame_id] if stable_only == 1 else mci_result_dict[frame_id]
        pprint.pprint(interaction_graph)
        print("\n********** Initiate Bayesian Fitting. **********")
        bayesian_fitter = BayesianFitter(frame, tau_max, interaction_graph)
        bayesian_fitter.analyze_discovery_statistics()
        if not skip_bayesian_fitting_flag:
            bayesian_fitter.construct_bayesian_model()
        print("Bayesian fitting complete. Consumed time: {} mins.".format((time() - start)*1.0/60))
    
    """Security Guard."""
    if COMM.rank == 0:
        print("\n********** Initiate Security Guarding. **********")
        security_guard = security_guard.SecurityGuard(frame=frame, bayesian_fitter=bayesian_fitter, sig_level=sig_level)
        evaluator = Evaluator(dataset=dataset, event_processor=event_preprocessor, background_generator=background_generator,\
                                             bayesian_fitter = bayesian_fitter, tau_max=tau_max)
        print("[Security guarding] Testing log starting positions {} with score threshold {}.".format(frame['testing-start-index'] + 1, security_guard.score_threshold))
        # 1. Inject device anomalies
        testing_event_sequence, anomaly_events, anomaly_positions, stable_states_dict = evaluator.simulate_malicious_control(frame=frame, n_anomaly=num_anomalies, maximum_length=max_prop_length)
        assert(len(anomaly_events) == len(anomaly_positions))
        for i in range(len(anomaly_positions)):
            assert(testing_event_sequence[anomaly_positions[i]] == anomaly_events[i])
        # 2. Initiate anomaly detection
        start = time()
        event_id = 0
        while event_id < len(testing_event_sequence):
            event = testing_event_sequence[event_id]
            report_to_user = False
            if event_id < tau_max:
                security_guard.initialize(event_id, event, frame['testing-data'].values[event_id])
            else:
                report_to_user = security_guard.score_anomaly_detection(event_id=event_id, event=event, debugging_id_list=anomaly_positions)
            # JC NOTE: Here we simulate a user involvement, which handles the reported anomalies as soon as it is reported.
            if event_id in anomaly_positions or report_to_user is True:
                security_guard.calibrate(event_id, stable_states_dict)
            event_id += 1

        print("[Security guarding] Anomaly detection completes for {} runtime events. Consumed time: {} mins.".format(event_id, (time() - start)*1.0/60))
        # 3. Evaluate the detection accuracy.
        print("[Security guarding] Evaluating the false positive for state transition violations")
        security_guard.print_debugging_dict(fp_flag=True)
        #violation_count_dict = {}; violation_event_ids = list(security_guard.violation_dict.keys())
        #for violation_event_id, violation_point in security_guard.violation_dict.items():
        #    #print(" * Violation (event id, interaction, score) = ({}, {}, {})".format(violation_event_id, violation_point['interaction'], violation_point['anomaly-score']))
        #    violation_count_dict[violation_point['attr']] = 1 if violation_point['attr'] not in violation_count_dict.keys() else violation_count_dict[violation_point['attr']] + 1
        #    violation_count_dict = dict(sorted(violation_count_dict.items(), key=lambda item: item[1]))
        #pprint.pprint(violation_count_dict)
        print("[Security guarding] Evaluating the false negative for state transition violations")
        security_guard.print_debugging_dict(fp_flag=False)

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