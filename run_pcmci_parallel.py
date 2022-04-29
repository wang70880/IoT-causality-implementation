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
from mpi4py import MPI
import numpy
import os, sys, pickle
import time

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.toymodels import structural_causal_processes as toys
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb
from src.causal_evaluation import Evaluator

import src.event_processing as evt_proc

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

# Parameter Settings
dataset = 'hh101'
stable_only = 0
partition_config = (1, 10)
cond_ind_test = CMIsymb()
tau_max = 1; tau_min = 1
verbosity = 2  # -1: No debugging information; 0: Debugging information in this module; 2: Debugging info in PCMCI class; 3: Debugging info in CIT implementations
## For stable-pc
pc_alpha = 0.1
max_conds_dim = 5
maximum_comb = 1
## For MCI
alpha_level = 0.001
max_conds_px = 5; max_conds_py= 5

pcmci_links_dict = {}; stable_links_dict = {}

event_preprocessor = evt_proc.Hprocessor(dataset)
attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
frame_id = 0
for dataframe in dataframes:
    T = dataframe.T; N = dataframe.N
    selected_variables = list(range(N))
    # selected_variables = [attr_names.index('M012')] # JC TODO: Remove ad-hoc codes here
    selected_links = {n: {m: [(i, -t) for i in range(N) for \
                t in range(tau_min, tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}
    # Scatter jobs given the avaliable processes
    splitted_jobs = None
    if COMM.rank == 0:
        if verbosity > -1:
            print("## Running Parallelized Tigramite PC algorithm\n##"
                  "\nParameters:")
            print(  "\nframe_id = %d" % frame_id
                  + "\nindependence test = %s" % cond_ind_test.measure
                  + "\nn_records = %d" % T 
                  + "\npartition_days = %d" % partition_config[1]
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\nmax_conds_dim = %s" % max_conds_dim)
        splitted_jobs = _split(selected_variables, COMM.size) # Split selected_variables into however many cores are available.
        if verbosity > -1:
            print("splitted selected_variables = ", splitted_jobs)
    scattered_jobs = COMM.scatter(splitted_jobs, root=0)

    # Each process calls stable-pc
    results = []
    for j in scattered_jobs:
        pc_start = time.time()
        (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links,\
                                                            tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                            max_conds_dim=max_conds_dim, verbosity=verbosity, maximum_comb=maximum_comb)
        results.append((j, pcmci_of_j, parents_of_j))
        pc_end = time.time()
        if verbosity > -1:
            print("[Rank {}] Finish pc-discovery for variable {}, consumed time: {} mins".format(COMM.rank, attr_names[j], (pc_end - pc_start) * 1.0 / 60))

    # Gather stable-pc results on rank 0, and distribute the gathered result to each slave node.
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
        stable_links_dict[frame_id] = all_parents_with_name
        for i in range(1, COMM.size):
            COMM.send((all_parents, pcmci_objects), dest=i)
    else:
        (all_parents, pcmci_objects) = COMM.recv(source=0)

    # If the function requires to run PCMCI instead of single stablePC
    if stable_only == 0:
        if COMM.rank == 0 and verbosity > -1:
            print("\n\n## Running Parallelized MCI algorithm\n##"
                  "\nParameters:")
            print(  "\nframe_id = %d" % frame_id
                  + "\nindependence test = %s" % cond_ind_test.measure
                  + "\nn_records = %d" % T 
                  + "\npartition_days = %d" % partition_config[1]
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\nalpha_level = %s" % alpha_level)
        scattered_jobs = COMM.scatter(splitted_jobs, root=0)
        results = []
        # Each process calls MCI algorithm
        for j in scattered_jobs:
            mci_start = time.time()
            (j, results_in_j) = _run_mci_parallel(j, pcmci_objects[j], all_parents, selected_links=selected_links,\
                                            tau_min=tau_min, tau_max=tau_max, alpha_level=alpha_level,\
                                            max_conds_px = max_conds_px, max_conds_py=max_conds_py)
            results.append((j, results_in_j))
            mci_end = time.time()
            print("[Rank {}] Finish MCI algorithm for variable {}, consumed time: {} mins".format(COMM.rank, attr_names[j], (mci_end - mci_start) * 1.0 / 60))
 
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
                                    all_results[key] = numpy.ones(results_in_j[key].shape)
                                else:
                                    all_results[key] = numpy.zeros(results_in_j[key].shape, dtype=results_in_j[key].dtype)
                            all_results[key][:, j, :] = results_in_j[key][:, j, :]
            p_matrix = all_results['p_matrix']
            val_matrix = all_results['val_matrix']
            conf_matrix = all_results['conf_matrix']
 
            sig_links = (p_matrix <= alpha_level)
            sorted_links_with_name = {}
            for j in selected_variables:
                links = dict([((p[0], -p[1]), numpy.abs(val_matrix[p[0], j, abs(p[1])]))
                            for p in zip(*numpy.where(sig_links[:, j, :]))])
                sorted_links = sorted(links, key=links.get, reverse=True)
                sorted_links_with_name[attr_names[j]] = []
                for p in sorted_links:
                    sorted_links_with_name[attr_names[j]].append((attr_names[p[0]], p[1]))
                #if verbosity > -1:
                #    print("Variable {} has {} links.".format(attr_names[j], len(links)))
                #    for p in sorted_links:
                #        print(" ({}, {}): pval = {}, val = {}.".format(attr_names[p[0]], p[1], p_matrix[p[0], j, abs(p[1])], val_matrix[p[0], j, abs(p[1])]))
            pcmci_links_dict[frame_id] = sorted_links_with_name
 
    frame_id += 1
    if frame_id >= 1:
        break

# Initiate the evasluation
if COMM.rank == 0:
    evaluator = Evaluator(dataset=dataset, partition_config=partition_config, tau_max=tau_max)
    results_to_be_evaluted = stable_links_dict if stable_only == 1 else pcmci_links_dict
    print("Results for frame {}, tau {}: {}".format(0, tau_max, results_to_be_evaluted[0]))
    evaluator._adhoc_estimate_single_discovery_accuracy(0, tau_max, results_to_be_evaluted[0])
