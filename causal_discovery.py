import os, sys, pickle
#os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import statistics
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from src.background_generator import BackgroundGenerator
import src.security_guard as security_guard
from src.event_processing import Hprocessor
from src.bayesian_fitter import BayesianFitter
from src.causal_evaluation import Evaluator
from src.genetic_type import DataFrame, AttrEvent, DevAttribute

# Default communicator
COMM = MPI.COMM_WORLD
NORMAL = 0
ABNORMAL = 1

# Accept parameters for data partitioning and loading

apply_bk = 1
#apply_bk = int(sys.argv[4])

# Settings of test parameters
autocorrelation_flag = True 
skip_skeleton_estimation_flag = False
skip_bayesian_fitting_flag = 0
num_anomalies = 0
max_prop_length = 1

cond_ind_test = CMIsymb()
tau_max = 3; tau_min = 1
verbosity = -1 # -1: No debugging information; 0: Debugging information in this module; 2: Debugging info in PCMCI class; 3: Debugging info in CIT implementations
## For stable-pc
max_n_edges = 50
pc_alpha = 0.001
max_conds_dim = 5
maximum_comb = 10

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

# 1. Load data and create data frame
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3])
event_preprocessor:'Hprocessor' = Hprocessor(dataset=dataset,verbosity=0, partition_days=partition_days, training_ratio=training_ratio)
event_preprocessor.data_loading()
frame:'DataFrame' = event_preprocessor.frame_dict[0] # By default, we use the first data frame

# 2. Identify the background knowledge
tau_max = int(sys.argv[4]); tau_min = 1
background_generator = BackgroundGenerator(event_preprocessor, tau_max)

exit()

for frame_id in frame_dict.keys():

    """Parallel Interaction Miner"""
    # Result variables
    all_parents = {}; all_parents_with_name = {}
    pcmci_objects = {}
    pc_result_dict = {}; mci_result_dict = {}

    # Auxillary variables
    frame: 'DataFrame' = event_preprocessor.frame_dict[frame_id]
    index_device_dict:'dict[DevAttribute]' = event_preprocessor.index_device_dict
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
            print("Prepare to parallel discover links. For COMM.size = {}, splitted_jobs =\n{}".format(COMM.size, splitted_jobs))
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
        if COMM.rank == 0: # The root node gathers the result and generate the interaction graph: all_parents, all_parents_with_name, pcmci_objects
            all_parents = {}
            for res in results:
                for (j, pcmci_of_j, parents_of_j) in res:
                    all_parents[j] = parents_of_j[j]
                    pcmci_objects[j] = pcmci_of_j
            for outcome_id, cause_list in all_parents.items():
                all_parents_with_name[index_device_dict[outcome_id].name] = [(index_device_dict[cause_id].name,lag) for (cause_id, lag) in cause_list]
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
    end = time()

    if COMM.rank == 0 and not skip_skeleton_estimation_flag : # Plot the graph if the PC procedure is not skipped.
        print("Parallel Interaction Mining finished. Consumed time: {} minutes".format((end - start)*1.0/60))
        print("parents dict:\n{}".format(all_parents_with_name))
        answer_shape = (dataframe.N, dataframe.N, tau_max + 1)
        graph = np.zeros(answer_shape, dtype='<U3'); val_matrix = np.zeros(answer_shape); p_matrix = np.ones(answer_shape)
        for j in pcmci_objects.keys():
            pcmci_object:'PCMCI' = pcmci_objects[j]
            local_val_matrix = pcmci_object.results['val_matrix']; local_p_matrix = pcmci_object.results['p_matrix']; local_graph_matrix = pcmci_object.results['graph']
            assert(all([x == 0 for x in val_matrix[local_val_matrix > 0]])); val_matrix += local_val_matrix
            assert(all([x == 1 for x in p_matrix[local_p_matrix < 1]])); p_matrix[local_p_matrix < 1] = local_p_matrix[local_p_matrix < 1]
            assert(all([x == '' for x in graph[local_graph_matrix != '']])); graph[local_graph_matrix != ''] = local_graph_matrix[local_graph_matrix != '']
            for lag in range(tau_max + 1):
                print("Job {}'s val-matrix for lag = {}:\n{}\n".format(j, lag, local_val_matrix[:,:,lag]))
                print("Job {}'s graph for lag = {}:\n{}\n".format(j, lag, local_graph_matrix[:,:,lag]))
                print("Job {}'s p-matrix for lag = {}:\n{}\n".format(j, lag, local_p_matrix[:,:,lag]))
                print("Current summarized p-matrix for lag = {}:\n{}\n".format(lag, p_matrix[:,:,lag]))

        print("Final graph:\n{}\n".format(graph))
        tp.plot_time_series_graph(
            figsize=(6, 4),
            val_matrix=val_matrix,
            graph=graph,
            var_names= dataframe.var_names,
            link_colorbar_label='MCI'
        )
        plt.savefig("temp/image/{}cmi_test_tau{}.pdf".format(dataset, tau_max))
        plt.close('all')

    frame_id += 1

    if single_frame_test_flag == 1:
        break
    if frame_id == event_preprocessor.frame_count - 1:
        break