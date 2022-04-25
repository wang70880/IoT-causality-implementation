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
        max_combinations=maximum_comb,
        max_conds_dim=max_conds_dim
    )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j


def run_mci_parallel(j, pcmci_of_j, all_parents,\
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

# Parameter setting
dataset = 'hh101'
partion_config = (1, 10)
cond_ind_test = CMIsymb()
tau_max = 1; tau_min = 1
verbosity = 2
## For stable-pc
pc_alpha = 0.2
max_conds_dim = 10
maximum_comb = 1
## For MCI
alpha_level = 0.005
max_conds_px = None; max_conds_py = None

# Data frame construction
#print("* Initiate data preprocessing.")
event_preprocessor = evt_proc.Hprocessor(dataset)
attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partion_config)
#print("* Data preprocessing finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))

# Prepare to initiate the task
frame_id = 0
for dataframe in dataframes:
    pc_stable_time = 0
    mci_time = 0
    pc_start = time.time()
    T = dataframe.T; N = dataframe.N
    selected_variables = list(range(N))
    selected_variables = [attr_names.index('D002')] # JC TODO: Remove ad-hoc codes here
    selected_links = {n: {m: [(i, -t) for i in range(N) for \
                          t in range(tau_min, tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}

    if COMM.rank == 0: # Print out parameter settings and prepare to call stable-pc
        if verbosity > -1:
            print("## Running Parallelized Tigramite PC algorithm\n##"
                  "\nParameters:")
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
    for j in scattered_jobs: # Call stable-pc
        # Estimate conditions
        start = time.time()
        (j, pcmci_of_j, parents_of_j) = run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test, selected_links=selected_links,\
                                                            tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                            max_conds_dim=max_conds_dim, verbosity=verbosity, maximum_comb=maximum_comb)
        results.append((j, pcmci_of_j, parents_of_j))
        end = time.time()
        if verbosity > 0:
            print("Rank {} finishes processing variable {}, consumed time: {} mins".format(COMM.rank, j, (end - pc_start) * 1.0 / 60))
            print("Parents of variable {}: {}".format(j, parents_of_j))

    # Gather stable-pc results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0: # Summarize pc-stable results
        all_parents = {}
        pcmci_objects = {}
        for res in results:
            for (j, pcmci_of_j, parents_of_j) in res:
                all_parents[j] = parents_of_j[j]
                pcmci_objects[j] = pcmci_of_j
        
        all_parents_with_name = {}
        for outcome_id, cause_list in all_parents.items():
            all_parents_with_name[attr_names[outcome_id]] = [(attr_names[cause_id],lag) for (cause_id, lag) in cause_list]
        
        end = time.time()
        pc_stable_time = (end - pc_start) * 1.0 / 60

        if verbosity > -1:
            print("\n\n## pc_results = {}".format(all_parents))
            print("## pc_results_with_name = {}".format(all_parents_with_name))
            print("Number of links: {}".format(sum([len(x) for x in all_parents_with_name.values()])))
            #for j in [var for var in all_parents.keys()]:
            #    pcmci_objects[j]._print_parents_single(j, all_parents[j],
            #                                           pcmci_objects[j].val_min[j],
            #                                           None)
            print("\n##PC stable elapsed time: {} mins".format(pc_stable_time))

            print("\n\n## Running Parallelized MCI algorithm\n##"
                  "\nParameters:")
            print(  "\nframe_id = %d" % frame_id
                  + "\nindependence test = %s" % cond_ind_test.measure
                  + "\nn_records = %d" % T 
                  + "\npartition_days = %d" % partion_config[1]
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\nalpha_level = %s" % alpha_level)

        for i in range(1, COMM.size):
            COMM.send((all_parents, pcmci_objects), dest=i)
    else:
        if verbosity > 0:
            print("Slave node %d: Receiving all_parents and pcmci_objects..."
                  "" % COMM.rank)
        (all_parents, pcmci_objects) = COMM.recv(source=0)

    mci_start = time.time()
    scattered_jobs = COMM.scatter(splitted_jobs, root=0)
    results = []
    for j in scattered_jobs: # Call mci
        (j, results_in_j) = run_mci_parallel(j, pcmci_objects[j], all_parents,\
                                        tau_min=tau_min, tau_max=tau_max, alpha_level=alpha_level,\
                                        max_conds_px = max_conds_px, max_conds_py=max_conds_py)
        results.append((j, results_in_j))
    results = MPI.COMM_WORLD.gather(results, root=0)
    end = time.time()
    mci_time = (end - mci_start) * 1.0 / 60
    if COMM.rank == 0: # Summarize mci results
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
    
        if verbosity > -1: # Print mci results
            print("\n## Significant links at alpha = %s:" % alpha_level)
            for j in selected_variables:
    
                links = dict([((p[0], -p[1]), numpy.abs(val_matrix[p[0],
                                                                   j, abs(p[1])]))
                              for p in zip(*numpy.where(sig_links[:, j, :]))])
    
                # Sort by value
                sorted_links = sorted(links, key=links.get, reverse=True)
                sorted_links_with_name = {}
                sorted_links_with_name[attr_names[j]] = []
                for p in sorted_links:
                    sorted_links_with_name[attr_names[j]].append((attr_names[p[0]], p[1], p_matrix[p[0], j, abs(p[1])]))
                print("\n\n## mci_results = {}".format(sorted_links))
                print("## mci_results_with_name = {}".format(sorted_links_with_name))                
                print("Number of links: {}".format(sum([len(x) for x in all_parents_with_name.values()])))
    
                #n_links = len(links)
                #string = ""
                #string = ("\n    Variable %s has %d "
                #          "link(s):" % (attr_names[j], n_links))
                #for p in sorted_links:
                #    string += ("\n        (%s %d): pval = %.5f" %
                #               (attr_names[p[0]], p[1],
                #                p_matrix[p[0], j, abs(p[1])]))
    
                #    string += " | val = %.3f" % (
                #        val_matrix[p[0], j, abs(p[1])])
    
                #    if conf_matrix is not None:
                #        string += " | conf = (%.3f, %.3f)" % (
                #            conf_matrix[p[0], j, abs(p[1])][0],
                #            conf_matrix[p[0], j, abs(p[1])][1])
            print("MCI elapsed time = {} minutes".format(mci_time))
    frame_id += 1
    if frame_id >= 1:
        break