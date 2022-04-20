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
from src.tigramite.tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

import src.event_processing as evt_proc

# Default communicator
COMM = MPI.COMM_WORLD


def split(container, count):
    """
    Simple function splitting a range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]


def run_pc_stable_parallel(j, dataframe, cond_ind_test, selected_links, tau_min=1, tau_max=1, pc_alpha = 0.1, verbosity=0):
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

    # CondIndTest is initialized globally below
    # Further parameters of PCMCI as described in the documentation can be
    # supplied here:
    # print("Parameter review")
    # print("j = {}".format(j))
    # print("dataframe in format {}, {}".format(dataframe.T, dataframe.N))
    # print("cond_ind_test: {}".format(cond_ind_test))
    # print("selected_links: {}".format(selected_links))
    # print("tau_min, tau_max = {}, {}".format(tau_min, tau_max))
    # print("pc_alpha = {}".format(pc_alpha))

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
        max_combinations=5
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

# Data frame construction
print("* Initiate data preprocessing.")
start = time.time()
dataset = 'hh101'
partion_config = (1, 1)
event_preprocessor = evt_proc.Hprocessor(dataset)
attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partion_config)
verbosity = 2
end = time.time()
print("* Data preprocessing finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))

# Prepare to initiate the task
for dataframe in dataframes:
    start = time.time()
    print("* Initiate stable PC.")
    T = dataframe.T; N = dataframe.N
    print("Number of records: {}".format(T))
    pc_alpha = 0.1; alpha_level = 0.05
    selected_variables = list(range(N))
    tau_max = 2; tau_min = 1
    max_conds_dim = None; max_conds_px = None
    selected_links = {n: {m: [(i, -t) for i in range(N) for \
                          t in range(tau_min, tau_max)] if m == n else [] for m in range(N)} for n in range(N)}
    print("selected_links: {}".format(selected_links))
    verbosity = 0
    cond_ind_test =CMIsymb()

    # Start the script
    if COMM.rank == 0:
        # Only the master node (rank=0) runs this
        if verbosity > -1:
            print("\n##\n## Running Parallelized Tigramite PC algorithm\n##"
                  "\n\nParameters:")
            print("\nindependence test = %s" % cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\nmax_conds_dim = %s" % max_conds_dim)
            print("\n")

        # Split selected_variables into however many cores are available.
        splitted_jobs = split(selected_variables, COMM.size)
        if verbosity > -1:
            print("Splitted selected_variables = ", splitted_jobs)
    else:
        splitted_jobs = None

    scattered_jobs = COMM.scatter(splitted_jobs, root=0)
    results = []
    for j in scattered_jobs:
        # Estimate conditions
        (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha, verbosity=verbosity)
        print(parents_of_j)
        results.append((j, pcmci_of_j, parents_of_j))

    # Gather results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)
    end = time.time()
    print("* Stable pc finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))
    print(results)
    break

#if COMM.rank == 0:
#    # Collect all results in dictionaries and send results to workers
#    all_parents = {}
#    pcmci_objects = {}
#    for res in results:
#        for (j, pcmci_of_j, parents_of_j) in res:
#            all_parents[j] = parents_of_j[j]
#            pcmci_objects[j] = pcmci_of_j
#
#    if verbosity > -1:
#        print("\n\n## Resulting condition sets:")
#        for j in [var for var in all_parents.keys()]:
#            pcmci_objects[j]._print_parents_single(j, all_parents[j],
#                                                   pcmci_objects[j].val_min[j],
#                                                   None)
#
#    if verbosity > -1:
#        print("\n##\n## Running Parallelized Tigramite MCI algorithm\n##"
#              "\n\nParameters:")
#
#        print("\nindependence test = %s" % cond_ind_test.measure
#              + "\ntau_min = %d" % tau_min
#              + "\ntau_max = %d" % tau_max
#              + "\nmax_conds_px = %s" % max_conds_px)
#        
#        print("Master node: Sending all_parents and pcmci_objects to workers.")
#    
#    for i in range(1, COMM.size):
#        COMM.send((all_parents, pcmci_objects), dest=i)
#
#else:
#    if verbosity > -1:
#        print("Slave node %d: Receiving all_parents and pcmci_objects..."
#              "" % COMM.rank)
#    (all_parents, pcmci_objects) = COMM.recv(source=0)
#
#print(all_parents)
#
#
###
###   MCI step
###
## Scatter jobs again across cores.
#scattered_jobs = COMM.scatter(splitted_jobs, root=0)
#
## Now each rank just does its jobs and collects everything in a results list.
#results = []
#for j in scattered_jobs:
#    (j, results_in_j) = run_mci_parallel(j, pcmci_objects[j], all_parents)
#    results.append((j, results_in_j))
#
## Gather results on rank 0.
#results = MPI.COMM_WORLD.gather(results, root=0)
#
#
#if COMM.rank == 0:
#    # Collect all results in dictionaries
#    # 
#    if verbosity > -1:
#        print("\nCollecting results...")
#    all_results = {}
#    for res in results:
#        for (j, results_in_j) in res:
#            for key in results_in_j.keys():
#                if results_in_j[key] is None:  
#                    all_results[key] = None
#                else:
#                    if key not in all_results.keys():
#                        if key == 'p_matrix':
#                            all_results[key] = numpy.ones(results_in_j[key].shape)
#                        else:
#                            all_results[key] = numpy.zeros(results_in_j[key].shape, dtype=results_in_j[key].dtype)
#                    all_results[key][:, j, :] = results_in_j[key][:, j, :]
#
#    p_matrix = all_results['p_matrix']
#    val_matrix = all_results['val_matrix']
#    conf_matrix = all_results['conf_matrix']
#    # if 'graph' in all_results.keys():
#    #     graph = all_results['graph']
#    #     if verbosity > -1:
#    #         print(all_results['graph'])
#
#    sig_links = (p_matrix <= alpha_level)
#
#    if verbosity > -1:
#        print("\n## Significant links at alpha = %s:" % alpha_level)
#        for j in selected_variables:
#
#            links = dict([((p[0], -p[1]), numpy.abs(val_matrix[p[0],
#                                                               j, abs(p[1])]))
#                          for p in zip(*numpy.where(sig_links[:, j, :]))])
#
#            # Sort by value
#            sorted_links = sorted(links, key=links.get, reverse=True)
#
#            n_links = len(links)
#
#            string = ""
#            string = ("\n    Variable %s has %d "
#                      "link(s):" % (var_names[j], n_links))
#            for p in sorted_links:
#                string += ("\n        (%s %d): pval = %.5f" %
#                           (var_names[p[0]], p[1],
#                            p_matrix[p[0], j, abs(p[1])]))
#
#                string += " | val = %.3f" % (
#                    val_matrix[p[0], j, abs(p[1])])
#
#                if conf_matrix is not None:
#                    string += " | conf = (%.3f, %.3f)" % (
#                        conf_matrix[p[0], j, abs(p[1])][0],
#                        conf_matrix[p[0], j, abs(p[1])][1])
#
#            print(string)
