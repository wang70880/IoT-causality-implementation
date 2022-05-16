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
from mpi4py import MPI
import numpy as np
import os, sys, pickle
import statistics
import time

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.toymodels import structural_causal_processes as toys
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb

import src.event_processing as evt_proc
import src.background_generator as bk_generator
import src.causal_evaluation as causal_eval
import src.causal_monitor as causal_monitor
from src.bayesian_predictor import BayesianPredictor

# Default communicator
COMM = MPI.COMM_WORLD

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

"""Parameter Settings"""
partition_config = int(sys.argv[1])
apply_bk = int(sys.argv[2])

dataset = 'hh101'
stable_only = 1
cond_ind_test = CMIsymb()
tau_max = 1; tau_min = 1
verbosity = -1 # -1: No debugging information; 0: Debugging information in this module; 2: Debugging info in PCMCI class; 3: Debugging info in CIT implementations
test_flag = 1
## For stable-pc
pc_alpha = 0.1
max_conds_dim = 5
maximum_comb = 1
## For MCI
alpha_level = 0.01
max_conds_px = 5; max_conds_py= 5
## Resulting dict
pc_result_dict = {}; mci_result_dict = {}
# For evaluations
record_count_list =[]
pc_time_list = []; mci_time_list = []

"""Preprocess the data. Construct background knowledge and golden standard"""
event_preprocessor = evt_proc.Hprocessor(dataset)
attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
background_generator = bk_generator.BackgroundGenerator(dataset, event_preprocessor, partition_config, tau_max)
evaluator = causal_eval.Evaluator(dataset=dataset, event_processor=event_preprocessor, background_generator=background_generator, tau_max=tau_max)
frame_id = 0

"""Initiate Causal Discovery algorithm."""
for dataframe in dataframes:
    T = dataframe.T; N = dataframe.N
    record_count_list.append(T)
    selected_variables = list(range(N))
    splitted_jobs = None
    results = []
    """Apply background knowledge to prune some edges in advance."""
    selected_links = {n: {m: [(i, -t) for i in range(N) for \
            t in range(tau_min, tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}
    if apply_bk >= 1:
        selected_links = background_generator.apply_background_knowledge(selected_links, 'heuristic-temporal', frame_id)
    if apply_bk >= 2:
        selected_links = background_generator.apply_background_knowledge(selected_links, 'spatial', frame_id)
        selected_links = background_generator.apply_background_knowledge(selected_links, 'functionality', frame_id)
    
    """[Numba] Scatter jobs given the avaliable processes."""
    if COMM.rank == 0:
        splitted_jobs = _split(selected_variables, COMM.size) # Split selected_variables into however many cores are available.
    scattered_jobs = COMM.scatter(splitted_jobs, root=0)
    
    """Each process calls stable-pc"""
    pc_start = time.time()
    for j in scattered_jobs:
        (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links,\
                                                            tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                            max_conds_dim=max_conds_dim, verbosity=verbosity, maximum_comb=maximum_comb)
        results.append((j, pcmci_of_j, parents_of_j))
    
    """Gather stable-pc results on rank 0, and evaluate the accuracy."""
    results = MPI.COMM_WORLD.gather(results, root=0)
    if COMM.rank == 0: 
        all_parents = {}
        pcmci_objects = {}
        for res in results:
            for (j, pcmci_of_j, parents_of_j) in res:
                all_parents[j] = parents_of_j[j]
                pcmci_objects[j] = pcmci_of_j
        all_parents_with_name = {}
        for outcome_id, cause_list in all_parents.items():
            all_parents_with_name[attr_names[outcome_id]] = [(attr_names[cause_id],lag) for (cause_id, lag) in cause_list]
        pc_end = time.time()
        pc_result_dict[frame_id] = all_parents_with_name; pc_time_list.append((pc_end - pc_start) * 1.0 / 60)
        if verbosity > -1:
            print("##\n## PC-stable discovery for frame {} finished. Consumed time: {} mins\n##".format(frame_id, (pc_end - pc_start) * 1.0 / 60))
        for i in range(1, COMM.size):
            COMM.send((all_parents, pcmci_objects), dest=i)
    else:
        (all_parents, pcmci_objects) = COMM.recv(source=0)
    
    """If MCI procedure is needed, run it."""
    if stable_only == 0:
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
    
    if test_flag == 1:  # JC TODO: Remove ad-hoc testing codes here or update the test_flag!
        bayesian_predictor = BayesianPredictor(dataframe, tau_max, pc_result_dict[0])
        bayesian_predictor._construct_bayesian_model()
        break

    frame_id += 1
    if frame_id == len(dataframes) - 1: # JC NOTE: Skip the last frame in case that the number of records is not enough.
        break

"""Evaluate discovery accuracy and compare with ARM."""
if COMM.rank == 0:
    pc_avg_truth_count, pc_avg_precision, pc_avg_recall = evaluator.estimate_average_discovery_accuracy(1, pc_result_dict)
    str = "**** Discovery results for partition_config = {}, bk = {} ****".format(partition_config, apply_bk) \
          + "\n Algorithm parameters:"\
          + "\n Number of frames: {}".format(frame_id) \
          + "\n * independence test = %s" % cond_ind_test.measure \
          + "\n * tau_min = {}, tau_max = {}".format(tau_min, tau_max) \
          + "\n * pc_alpha = {}, max_conds_dim = {}, max_comb = {}".format(pc_alpha, max_conds_dim, maximum_comb) \
          + "\n * alpha_level = {}, max_conds_px = {}, max_conds_py = {}".format(alpha_level, max_conds_px, max_conds_py) \
          + "\n Average counts of records: {}".format(statistics.mean(record_count_list)) \
          + "\nstablePC evaluations: average time, truth-count, precision, recall = {}, {}, {}, {}".format(statistics.mean(pc_time_list), pc_avg_truth_count, pc_avg_precision, pc_avg_recall)
    if stable_only == 0: # If the PCMCI is also initiated, show the evaluation result.
        mci_avg_truth_count, mci_avg_precision, mci_avg_recall = evaluator.estimate_average_discovery_accuracy(1, mci_result_dict)
        str += "\nMCI evaluations: average time, truth-count, precision, recall = {}, {}, {}, {}".format(statistics.mean(mci_time_list), mci_avg_truth_count, mci_avg_precision, mci_avg_recall)
    print(str)
    # JC TODO: Comparisons with ARM (It seems that ARM is executed so slow..)

"""State prediction and anomaly detections."""
if COMM.rank == 0:
    # JC TODO: Prepare to initiate bayesian prediction and implement the phantom machine for predictions.
    pass