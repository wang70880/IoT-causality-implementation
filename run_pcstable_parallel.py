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


from mpi4py import MPI
import numpy
import os, sys, pickle
import time

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.toymodels import structural_causal_processes as toys
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb

import src.event_processing as evt_proc

# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]


def run_pc_stable_parallel(j, dataframe, cond_ind_test, selected_links, tau_min=1, tau_max=1, pc_alpha = 0.1, verbosity=0, maximum_comb = None, max_conds_dim=None):
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


def run_mci_parallel(j, pcmci_of_j, all_parents):
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
        max_conds_px=max_conds_px,
    )

    return j, results_in_j

# Parameter setting
dataset = 'hh101'
partion_config = (1, 10)
verbosity = 2
pc_alpha = 0.001; alpha_level = 0.05
tau_max = 1; tau_min = 1
max_conds_dim = 5; max_conds_px = None
maximum_comb = 10
cond_ind_test = CMIsymb()

# Data frame construction
#print("* Initiate data preprocessing.")
event_preprocessor = evt_proc.Hprocessor(dataset)
attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partion_config)
#print("* Data preprocessing finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))

# Prepare to initiate the task
frame_id = 0
for dataframe in dataframes:
    int_start = time.time()
    T = dataframe.T; N = dataframe.N
    selected_variables = list(range(N))
    selected_variables = [attr_names.index('D002')] # JC TODO: Remove ad-hoc codes here
    selected_links = {n: {m: [(i, -t) for i in range(N) for \
                          t in range(tau_min, tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}

    # Start the script
    if COMM.rank == 0:
        # Only the master node (rank=0) runs this
        if verbosity > -1:
            #print("\n##\n## Running Parallelized Tigramite PC algorithm\n##"
            #      "\n\nParameters:")
            print(  "\nframe_id = %d" % frame_id
                  + "\nindependence test = %s" % cond_ind_test.measure
                  + "\nn_records = %d" % T 
                  + "\npartition_days = %d" % partion_config[1]
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\nmax_conds_dim = %s" % max_conds_dim)

        # Split selected_variables into however many cores are available.
        splitted_jobs = split(selected_variables, COMM.size)
        if verbosity > -1:
            print("splitted selected_variables = ", splitted_jobs)
    else:
        splitted_jobs = None

    scattered_jobs = COMM.scatter(splitted_jobs, root=0)
    results = []
    for j in scattered_jobs:
        # Estimate conditions
        start = time.time()
        (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links,\
                                                            tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                            max_conds_dim=max_conds_dim, verbosity=verbosity, maximum_comb=maximum_comb)
        results.append((j, pcmci_of_j, parents_of_j))
        end = time.time()
        if verbosity > 0:
            print("Rank {} finishes processing variable {}, consumed time: {} mins".format(COMM.rank, j, (end - int_start) * 1.0 / 60))
            print("Parents of variable {}: {}".format(j, parents_of_j))

    # Gather results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0:
        # Collect all results in dictionaries and send results to workers
        all_parents = {}
        pcmci_objects = {}
        for res in results:
            for (j, pcmci_of_j, parents_of_j) in res:
                all_parents[j] = parents_of_j[j]
                pcmci_objects[j] = pcmci_of_j
        
        all_parents_with_name = {}
        for outcome_id, cause_list in all_parents.items():
            all_parents_with_name[attr_names[outcome_id]] = [(attr_names[cause_id],lag) for (cause_id, lag) in cause_list]

        if verbosity > -1:
            print("\n\n## pc_results = {}".format(all_parents))

            print("## pc_results_with_name = {}".format(all_parents_with_name))
            print("Number of links: {}".format(sum([len(x) for x in all_parents_with_name.values()])))
            end = time.time()
            print("\n\n##Elapsed time: {} mins".format((end - int_start) * 1.0 / 60))

    else:
        if verbosity > 0:
            print("Slave node %d: Receiving all_parents and pcmci_objects..."
                  "" % COMM.rank)
        (all_parents, pcmci_objects) = COMM.recv(source=0)

    frame_id += 1
    if frame_id >= 1:
        break