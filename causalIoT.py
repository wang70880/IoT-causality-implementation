from email.policy import default
import os, sys, pickle
#os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import statistics
import pprint
import numpy as np
import pandas as pd

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
from src.event_processing import Hprocessor, Cprocessor
from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.causal_evaluation import Evaluator
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.arm import ARMer

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
    maximum_comb = None, max_conds_dim=None, debugging_pair=None):
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
        save_iterations=True,
        pc_alpha=pc_alpha,
        max_combinations=maximum_comb,
        max_conds_dim=max_conds_dim,
        debugging_pair=debugging_pair
    )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j

# 0. Parameter settings
    # 0.1 Data-loading parameters
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3]); frame_id = 0 # JC NOTE: By default, we use the first data frame
    # 0.2 Background knowledge parameters
tau_max = int(sys.argv[4]); tau_min = 1; bk_level=int(sys.argv[5])
    # 0.3 PC discovery parameters
pc_alpha = float(sys.argv[6]); max_conds_dim = int(sys.argv[7]); maximum_comb = int(sys.argv[8]); max_n_edges = 15
    # 0.4 Anomaly generation parameters
n_anomalies = 100; case = 1; max_length = 1; sig_level = 0.9

if COMM.rank == 0:
    print("\n\n********************** Parameter Settings **********************\n"\
     + "bk = {}, pc-alpha = {}, partition_days = {}".format(bk_level, pc_alpha, partition_days))

# 1. Load data and create data frame
dl_start =time()
if dataset.startswith('hh'):
    event_preprocessor:'Hprocessor' = Hprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio)
elif dataset.startswith('contextact'):
    event_preprocessor:'Cprocessor' = Cprocessor(dataset, partition_days, training_ratio)
event_preprocessor.data_loading()
frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
dataframe:pp.DataFrame = frame.training_dataframe; var_names = frame.var_names
n_records = dataframe.T; n_vars = dataframe.N
preprocessing_consumed_time = _elapsed_minutes(dl_start)
if COMM.rank == 0:
    print("     [Dataset Overview] # training records = {}, devices = {} ({})".format(n_records, frame.var_names, n_vars))

# 2. Identify the background knowledge, and use the background knowledge to prune edges
bg_start=time()
background_generator = BackgroundGenerator(event_preprocessor, frame_id, tau_max)
selected_links, candidate_matrix = background_generator.generate_candidate_interactions(bk_level) # Get candidate interactions
n_candidate_edges = 0
for worker_index, link_dict in selected_links.items():
    n_candidate_edges += sum([len(cause_list) for cause_list in link_dict.values()])
bk_consumed_time = _elapsed_minutes(bg_start)
if COMM.rank == 0:
    print("     [Background Integration] # candidate edges = {}".format(n_candidate_edges))

# 3. Construct the golden standard.
evaluator = Evaluator(event_preprocessor, background_generator, None, bk_level, pc_alpha)

# 4. Initiate parallel causal discovery
## 4.1 Scatter the jobs
pc_start = time()
selected_variables = list(range(n_vars))
splitted_jobs = None
results = []
if COMM.rank == 0:
    splitted_jobs = _split(selected_variables, COMM.size)
scattered_jobs = COMM.scatter(splitted_jobs, root=0)
## 4.2 Initiate parallel causal discovery, and generate the interaction dict (all_parents_with_name)
debugging_pair = (frame.name_device_dict['D002'].index, frame.name_device_dict['M006'].index)
cond_ind_test = ChiSquare()
for j in scattered_jobs:
    (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test,\
                                                        selected_links=selected_links, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                        max_conds_dim=max_conds_dim, maximum_comb=maximum_comb, verbosity=-1, debugging_pair=debugging_pair)
    filtered_parents_of_j = parents_of_j.copy()
    #n_edges = min(len(filtered_parents_of_j[j]), max_n_edges) # Only select top max_n_edges causal edges with maximum statistic values
    #if n_edges > 0:
    #    filtered_parents_of_j[j] = filtered_parents_of_j[j][0: n_edges]
    results.append((j, pcmci_of_j, filtered_parents_of_j))
results = MPI.COMM_WORLD.gather(results, root=0)
pc_consumed_time = _elapsed_minutes(pc_start)

## 4.3 Gather the distributed pc-discovery result, and collect information of filtered edges
if COMM.rank == 0:
    interaction_array:'np.ndarray' = np.zeros((n_vars, n_vars, tau_max+1), dtype=np.int8)
    index_device_dict:'dict[DevAttribute]' = event_preprocessor.index_device_dict
    all_parents = {}; pcmci_objects = {}; all_parents_with_name = {}
    filtered_edges = {}; filtered_edge_infos = defaultdict(list)
    for res in results:
        for (j, pcmci_of_j, parents_of_j) in res:
            all_parents[j] = parents_of_j[j]
            pcmci_objects[j] = pcmci_of_j
            filtered_edges[j] = pcmci_of_j.filtered_edges[j]
    for outcome, filtered_edges_dict in filtered_edges.items(): # collect information of filtered edges
        for edge, edge_infos in filtered_edges_dict.items():
                filtered_edge_infos[outcome].append((edge, edge_infos['conds'], edge_infos['val'], edge_infos['pval']))
        #print("# filtered edges for outcome {}: {}, {}".format(outcome, len(tupled_filtered_edges[outcome]), tupled_filtered_edges[outcome]))
    for outcome_id, cause_list in all_parents.items():
        for (cause_id, lag) in cause_list:
            all_parents_with_name[index_device_dict[outcome_id].name] = (index_device_dict[cause_id].name, lag)
            interaction_array[cause_id, outcome_id, abs(lag)] = 1
    n_discovered_edges = np.sum(interaction_array)
        #print("# discovered edges for outcome {}: {}".format(outcome_id, len(cause_list)))

# 5. Evaluate the discovery accuracy
if COMM.rank == 0:
    ## 5.1 Evaluate the discovery precision and recall
    n_golden_edges, precision, recall, f1 = evaluator.evaluate_discovery_accuracy(interaction_array, verbosity=1)
    print("     [IM Accuracy] # golden edges = {}, precision = {}, recall = {}, f1 = {}"\
                        .format(n_golden_edges, precision, recall, f1))
    ## 5.2 Chain analysis
    #interactions, interaction_types, n_paths = evaluator.interpret_discovery_results(interaction_array, golden_frame_id=frame_id, golden_type='user')
    ## 5.3 Time efficiency analysis
    print("     [IM Efficiency] Consumed time for preprocessing, background, causal discovery = {}, {}, {}"\
                .format(preprocessing_consumed_time, bk_consumed_time, pc_consumed_time))
    ## 5.4 Compare with ARM and analyze the result
    #arm_start = time()
    #armer = ARMer(frame=frame, min_support=frame.n_days, min_confidence=1.0-pc_alpha)
    #association_array:'np.ndarray' = armer.association_rule_mining()
    #arm_consumed_time = _elapsed_minutes(arm_start)
    #print("     [Causal Discovery Evaluation] Consumed time for Association Rule Mining is {}".format(arm_consumed_time))
    #evaluator.compare_with_arm(discovery_results=interaction_array, arm_results=association_array,\
    #                                golden_frame_id=frame_id, golden_type='user')
exit()

# After parallel causal discovery is finished, only RANK-0 process is kept.
if COMM.rank != 0:
    exit()

# 4. Bayesian Fitting and generate the parameterized interaction graph.
bf_start = time()
bayesian_fitter = BayesianFitter(frame, tau_max, all_parents_with_name)
bayesian_fitter.construct_bayesian_model()
bf_consumed_time = _elapsed_minutes(bf_start)

# 6. Initiate the anomaly injection and anomaly detection
# 6.1 Generate anomalies
injection_start = time()
testing_event_states, anomaly_positions, testing_benign_dict = evaluator.simulate_malicious_control(\
                        int_frame_id=frame_id, n_anomaly=n_anomalies, maximum_length=max_length, anomaly_case=case)
print("[Anomaly Injection] Complete ({} minutes). # of anomalies = {}".format(_elapsed_minutes(injection_start), len(anomaly_positions)))
# 6.2 Initiate anomaly detection
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
print(security_guard.tp_debugging_dict)
print(security_guard.fp_debugging_dict)
print(security_guard.fn_debugging_dict)