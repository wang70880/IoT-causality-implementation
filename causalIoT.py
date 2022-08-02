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
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare
from src.tigramite.tigramite import plotting as tp

from src.background_generator import BackgroundGenerator
from src.event_processing import Hprocessor
from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.causal_evaluation import Evaluator
from src.genetic_type import DataFrame, AttrEvent, DevAttribute

# Default communicator
COMM = MPI.COMM_WORLD

## Resulting dict
pc_result_dict = {}; mci_result_dict = {}
# For evaluations
record_count_list =[]
pc_time_list = []; mci_time_list = []

def _elapsed_minutes(start):
    return (time()-start) * 1.0 / 60

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
dl_start =time()
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3]); frame_id = 0 # JC NOTE: By default, we use the first data frame
event_preprocessor:'Hprocessor' = Hprocessor(dataset=dataset,verbosity=0, partition_days=partition_days, training_ratio=training_ratio)
event_preprocessor.data_loading()
frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
dataframe:pp.DataFrame = frame.training_dataframe; attr_names = frame.var_names
if COMM.rank == 0:
    print("[Data preprocessing] Complete ({} minutes). dataset={}, partition_days={}, training_ratio={}, # of training records={}\nattr_names={}\n"\
        .format(_elapsed_minutes(dl_start), dataset, partition_days, training_ratio, frame.n_events, attr_names))

# 2. Identify the background knowledge
bg_start=time()
tau_max = int(sys.argv[4]); tau_min = 1; filter_threshold=float(sys.argv[5]) # JC NOTE: Here we set the filter threshold empirically
background_generator = BackgroundGenerator(event_preprocessor, tau_max, filter_threshold)
if COMM.rank == 0:
    print("[Background Construction] Complete ({} minutes). tau-min={}, tau-max={}\n"\
        .format(_elapsed_minutes(bg_start), tau_min, tau_max))

# 3. Use the background knowledge to filter edges
ef_start = time()
bk_level=int(sys.argv[6])
T = dataframe.T; N = dataframe.N
selected_variables = list(range(N))
selected_links = background_generator.generate_candidate_interactions(bk_level, frame_id, N) # Get candidate interactions
n_candidate_edges = 0
for worker_index, link_dict in selected_links.items():
    n_candidate_edges += sum([len(cause_list) for cause_list in link_dict.values()])
if COMM.rank == 0:
    print("[Edge Filtering] Complete ({} minutes). bk-level={}, T={}, N={}, # of qualified edges={}\n"\
        .format(_elapsed_minutes(ef_start), bk_level, T, N, n_candidate_edges))

# 4. Split the job, and prepare to initiate causal discovery
splitted_jobs = None
results = []
if COMM.rank == 0: # Assign selected_variables into whatever cores are available.
    splitted_jobs = _split(selected_variables, COMM.size)
scattered_jobs = COMM.scatter(splitted_jobs, root=0)
print("[Job Scattering Rank {}] Complete. scattered_jobs = {}\n".format(COMM.rank, scattered_jobs))

# 5. Initiate parallel causal discovery
pc_start = time()
cond_ind_test = ChiSquare() # JC NOTE: Here we can replace the conditional independence test method (e.g., using CMI test).
pc_alpha = float(sys.argv[7]); max_conds_dim = int(sys.argv[8]); maximum_comb = int(sys.argv[9]); max_n_edges = 20
for j in scattered_jobs:
    (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test,\
                                                        selected_links=selected_links, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                        max_conds_dim=max_conds_dim, maximum_comb=maximum_comb, verbosity=-1)
    filtered_parents_of_j = parents_of_j.copy()
    n_edges = min(len(filtered_parents_of_j[j]), max_n_edges) # Only select top max_n_edges causal edges with maximum MCI
    if n_edges > 0:
        filtered_parents_of_j[j] = filtered_parents_of_j[j][0: n_edges]
    results.append((j, pcmci_of_j, filtered_parents_of_j))
    print("     [PC Discovery Rank {}] Complete ({} minutes).".format(COMM.rank, (time() - pc_start) * 1.0 / 60))
results = MPI.COMM_WORLD.gather(results, root=0)

# 6. Gather the distributed result
if COMM.rank == 0:
    index_device_dict:'dict[DevAttribute]' = event_preprocessor.index_device_dict
    all_parents = {}; pcmci_objects = {}; all_parents_with_name = {}
    for res in results:
        for (j, pcmci_of_j, parents_of_j) in res:
            all_parents[j] = parents_of_j[j]
            pcmci_objects[j] = pcmci_of_j
    for outcome_id, cause_list in all_parents.items():
        all_parents_with_name[index_device_dict[outcome_id].name] = [(index_device_dict[cause_id].name,lag) for (cause_id, lag) in cause_list]
    print("[PC Discovery] Complete ({} minutes).".format(_elapsed_minutes(pc_start)))
    print("Results:\n{}\n".format(all_parents_with_name))

# After parallel causal discovery is finished, only RANK-0 process is kept.
if COMM.rank != 0:
    exit()

# 7. Bayesian Fitting and generate the parameterized interaction graph.
bf_start = time()
bayesian_fitter = BayesianFitter(frame, tau_max, all_parents_with_name)
bayesian_fitter.construct_bayesian_model()
print("[Bayesian Fitting] Complete ({} minutes).".format(_elapsed_minutes(bf_start)))

# 8. Create the evaluator and create the golden standard.
gd_start = time()
evaluator = Evaluator(event_preprocessor, background_generator, bayesian_fitter)
print("[Golden Standard Construction] Complete ({} minutes).".format(_elapsed_minutes(gd_start)))
## 8.1 Evaluate the discovery accuracy.
evaluator.evaluate_discovery_accuracy()

# 9. Initiate the anomaly injection and anomaly detection
# 9.1 Generate anomalies
injection_start = time()
n_anomalies = 100; case = 1; max_length = 1
testing_event_states, anomaly_positions, testing_benign_dict = evaluator.simulate_malicious_control(\
                        int_frame_id=frame_id, n_anomaly=n_anomalies, maximum_length=max_length, anomaly_case=case)
print("[Anomaly Injection] Complete ({} minutes). # of anomalies = {}".format(_elapsed_minutes(injection_start), len(anomaly_positions)))
# 9.2 Initiate anomaly detection
sig_levels = [0.9]
for sig_level in sig_levels:
    detection_start = time()
    security_guard = SecurityGuard(frame=frame, bayesian_fitter=bayesian_fitter, sig_level=sig_level)
    for event_id, tup in enumerate(testing_event_states):
        event, states = tup
        if event_id < tau_max:
            security_guard.initialize(event_id, event, states)
        else:
            anomaly_flag = security_guard.score_anomaly_detection(event_id=event_id, event=event, debugging_id_list=anomaly_positions)
        # JC NOTE: By default, we simulate a user involvement which calibrates the detection system and handles fps and fns automatically.
        security_guard.calibrate(event_id, testing_benign_dict)
        event_id += 1
    print("[Anomaly Detection] Complete ({} minutes). # of records = {}, sig_level = {}".format(_elapsed_minutes(detection_start), len(testing_event_states), sig_level))
    print("Total number of tps, fps, fns: {}, {}, {}".format(\
        sum([len(x) for x in security_guard.tp_debugging_dict.values()]),\
        sum([len(x) for x in security_guard.fp_debugging_dict.values()]),\
        sum([len(x) for x in security_guard.fn_debugging_dict.values()])))
    pprint(security_guard.tp_debugging_dict)
    pprint(security_guard.fp_debugging_dict)
    pprint(security_guard.fn_debugging_dict)