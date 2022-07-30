import os, sys, pickle
#os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import statistics
import pprint
from turtle import back
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from collections import defaultdict
from mpi4py import MPI

from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb
from src.tigramite.tigramite import plotting as tp

from src.background_generator import BackgroundGenerator
from src.event_processing import Hprocessor
from src.genetic_type import DataFrame, AttrEvent, DevAttribute

# Default communicator
COMM = MPI.COMM_WORLD

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
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3]); frame_id = 0 # JC NOTE: By default, we use the first data frame
event_preprocessor:'Hprocessor' = Hprocessor(dataset=dataset,verbosity=0, partition_days=partition_days, training_ratio=training_ratio)
event_preprocessor.data_loading()
frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
dataframe:pp.DataFrame = frame.training_dataframe; attr_names = frame.var_names
print("[Data preprocessing] Complete. dataset={}, partition_days={}, training_ratio={}, # of training records={}\nattr_names={}"\
        .format(dataset, partition_days, training_ratio, frame.n_events, frame.name_device_dict))

# 2. Identify the background knowledge
tau_max = int(sys.argv[4]); tau_min = 1; filter_threshold=float(sys.argv[5])
background_generator = BackgroundGenerator(event_preprocessor, tau_max, filter_threshold)
print("[Background Construction] Complete. tau-min={}, tau-max={}, temporal knowledge=\n{}"\
            .format(tau_min, tau_max, background_generator.temporal_pair_dict[frame_id]))

# 3. Use the background knowledge to filter edges
bk_level=int(sys.argv[6])
T = dataframe.T; N = dataframe.N
selected_variables = list(range(N))
selected_links = background_generator.generate_candidate_interactions(bk_level, frame_id, N) # Get candidate interactions

# 4. Split the job, and prepare to initiate causal discovery
splitted_jobs = None
results = []
if COMM.rank == 0: # Assign selected_variables into whatever cores are available.
    splitted_jobs = _split(selected_variables, COMM.size)
scattered_jobs = COMM.scatter(splitted_jobs, root=0)

# 5. Initiate parallel causal discovery
cond_ind_test = CMIsymb() # JC TODO: Replace the conditional independence test method here.
pc_alpha = float(sys.argv[7]); max_conds_dim = 5; maximum_comb = 10; max_n_edges = 20
pc_start = time()
for j in scattered_jobs: # 5.1 Each process calls stable-pc algorithm to infer the edges
    (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test,\
                                                        selected_links=selected_links, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                        max_conds_dim=max_conds_dim, verbosity=-1, maximum_comb=maximum_comb)
    filtered_parents_of_j = parents_of_j.copy()
    n_edges = min(len(filtered_parents_of_j[j]), max_n_edges) # Only select top max_n_edges causal edges with maximum MCI
    if n_edges > 0:
        filtered_parents_of_j[j] = filtered_parents_of_j[j][0: n_edges]
    results.append((j, pcmci_of_j, filtered_parents_of_j))
results = MPI.COMM_WORLD.gather(results, root=0)
pc_end = time()

if COMM.rank == 0: # 5.2 The root node gathers the result
    consumed_time = (pc_end - pc_start) * 1.0 / 60
    index_device_dict:'dict[DevAttribute]' = event_preprocessor.index_device_dict
    all_parents = {}; pcmci_objects = {}; all_parents_with_name = {}
    for res in results:
        for (j, pcmci_of_j, parents_of_j) in res:
            all_parents[j] = parents_of_j[j]
            pcmci_objects[j] = pcmci_of_j
    for outcome_id, cause_list in all_parents.items():
        all_parents_with_name[index_device_dict[outcome_id].name] = [(index_device_dict[cause_id].name,lag) for (cause_id, lag) in cause_list]
    print("Parallel Interaction Mining finished. Consumed time: {} minutes".format(consumed_time))
    print("parents dict:\n{}".format(all_parents_with_name))