from email.policy import default
import os, sys, pickle
#os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import statistics
from pprint import pprint
from turtle import back
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

from time import time
from collections import defaultdict
from mpi4py import MPI

from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare

from src.background_generator import BackgroundGenerator
from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
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

def _normalize_temporal_array(target_array:'np.ndaray', tau_max):
    new_array = target_array.copy()
    if len(new_array.shape) == 3 and new_array.shape[-1] == tau_max+1:
        new_array = sum([new_array[:,:,tau] for tau in range(1, tau_max+1)])
        new_array[new_array>0] = 1
    return new_array

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

def construct_candidate_links(event_preprocessor, frame_id, tau_max, bk_level, pc_alpha):
    bg_start=time()
    background_generator = BackgroundGenerator(event_preprocessor, frame_id, tau_max)
    selected_links, n_candidate_edges = background_generator.generate_candidate_interactions(bk_level) # Get candidate interactions
    bk_consumed_time = _elapsed_minutes(bg_start)
    evaluator = Evaluator(event_preprocessor, background_generator, None, bk_level, pc_alpha)
    return background_generator, selected_links, n_candidate_edges, evaluator, bk_consumed_time

def gather_pc_results(max_n_edges=10):
    pcmci_objects = {}; identified_edges = {}; vals = {}; pvals = {}; filtered_edges = {} # Collect identified and filtered edges.
    for res in results:
        for (j, pcmci_of_j, _) in res:
            pcmci_objects[j] = pcmci_of_j
            identified_edges[j]:'list' = pcmci_of_j.all_parents[j]
            vals[j]:'dict' = pcmci_of_j.val_min[j]
            pvals[j]:'dict' = pcmci_of_j.pval_max[j]
            filtered_edges[j] = pcmci_of_j.filtered_edges[j]

    temporal_pc_array:'np.ndarray' = np.zeros((n_vars, n_vars, tau_max+1), dtype=np.int8)
    identified_edge_infos = defaultdict(list) # For each identified edge, collect its information (edge, val, pval)
    for outcome, cause_list in identified_edges.items():
        for parent in cause_list:
            edge = (outcome, parent)
            identified_edge_infos[outcome].append((edge, vals[outcome][parent], pvals[outcome][parent]))
            temporal_pc_array[parent[0], outcome, abs(parent[1])] = 1

    filtered_edge_infos = defaultdict(list) # For each filtered edge, collect its information (edge, condition, val, pval)
    for outcome, filtered_edges_dict in filtered_edges.items():
        for edge, edge_infos in filtered_edges_dict.items():
                filtered_edge_infos[outcome].append((edge, edge_infos['conds'], edge_infos['val'], edge_infos['pval']))
    pprint(identified_edges)
    return identified_edges, temporal_pc_array, identified_edge_infos, filtered_edge_infos

def evaluate_mining_results(temporal_pc_array, identified_edge_infos, filtered_edge_infos):
    """
    This function evaluates the discovery accuracy of CausalIoT, HAWatcher, and ARM.
    Moreover, it initiates the efficiency analysis for the three methods.
    """
    print("\n********************** Interaction Mining Evaluation")
    ## 5.0 Save the discovery result
    pc_array:'np.ndarray' = _normalize_temporal_array(temporal_pc_array, tau_max)
    df = pd.DataFrame(pc_array,columns=var_names,dtype=int)
    df.to_csv('results/contextact-ground-truth-{}'.format(max_conds_dim), sep='\t', index=False, encoding='utf-8')

    ## 5.1 Evaluate the CausalIoT discovery accuracy
    tp, fp, fn, precision, recall, f1 = evaluator.evaluate_discovery_accuracy(pc_array, filtered_edge_infos, identified_edge_infos, verbosity=1)
    print("     [CausalIoT] tp, fp, fn = {}, {}, {}. # golden edges = {}, precision = {}, recall = {}, f1 = {}"\
                        .format(tp, fp, fn, tp+fn, precision, recall, f1))

    ## 5.2 Evaluate the HAWatcher
    tp, fp, fn, precision, recall, f1 = evaluator.evaluate_discovery_accuracy(evaluator.golden_standard_dict['hawatcher'], verbosity=0)
    print("     [HAWatcher] tp, fp, fn = {}, {}, {}. # golden edges = {}, precision = {}, recall = {}, f1 = {}"\
                        .format(tp, fp, fn, tp+fn, precision, recall, f1))

    ## 5.3 Evaluate ARM discovery accuracy
    arm_start = time()
    ### Parameter settings
    min_support = frame.n_days; min_confidence=1.0-pc_alpha
    ### Initiate discovery and evaluation
    armer = ARMer(frame=frame, min_support=min_support, min_confidence=min_confidence)
    association_array:'np.ndarray' = armer.association_rule_mining()
    arm_consumed_time = _elapsed_minutes(arm_start)
    tp, fp, fn, precision, recall, f1 = evaluator.evaluate_discovery_accuracy(association_array, verbosity=0)
    print("     [ARM] tp, fp, fn = {}, {}, {}. # golden edges = {}, precision = {}, recall = {}, f1 = {}"\
                        .format(tp, fp, fn, tp+fn, precision, recall, f1))
    print("\n********************** Efficiency analysis\n   [CausalIoT, HAWatcher, ARM] Consumed time  = {}, {}, {}"\
        .format(pc_consumed_time, bk_consumed_time, arm_consumed_time))

# 0. Parameter settings
    # 0.1 Data-loading parameters
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3]); frame_id = 0 # JC NOTE: By default, we use the first data frame
    # 0.2 Background knowledge parameters
tau_max = int(sys.argv[4]); tau_min = 1; bk_level=int(sys.argv[5])
    # 0.3 PC discovery parameters
pc_alpha = float(sys.argv[6]); max_conds_dim = int(sys.argv[7]); maximum_comb = int(sys.argv[8])
    # 0.4 Bayesian fitting parameters
n_max_edges = 10
    # 0.5 Anomaly generation parameters
n_anomalies = 100; case = 1; max_length = 1; sig_level = 0.99

# 1. Load data and create data frame
dl_start =time()
event_preprocessor:'GeneralProcessor' = Hprocessor(dataset, partition_days, training_ratio) if dataset.startswith('hh') else Cprocessor(dataset, partition_days, training_ratio)
event_preprocessor.data_loading()
frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
dataframe:pp.DataFrame = frame.training_dataframe; var_names = frame.var_names
n_records = dataframe.T; n_vars = dataframe.N
preprocessing_consumed_time = _elapsed_minutes(dl_start)
if COMM.rank == 0:
    print("\n********************** Parameter Settings **********************\n"\
     + "    bk={}, pc-alpha={}, partition_days={}, max(dim)={}, max(comb)={}"\
            .format(bk_level, pc_alpha, partition_days, max_conds_dim, maximum_comb))
    print("\n*** Dataset Overview ***")
    print("     Number of training records, devices, and attributes = {}, {}, {}".format(n_records, frame.n_vars, frame.n_attrs))
    print("     device list: {}".format(frame.var_names))
    print("     attribute list: {}".format(frame.attr_names))

# 2. Identify the background knowledge and construct the result of HAWatcher
background_generator, selected_links, n_candidate_edges, evaluator, bk_consumed_time =\
        construct_candidate_links(event_preprocessor, frame_id, tau_max, bk_level, pc_alpha)

# 3. Initiate parallel causal discovery
## 3.1 Scatter the jobs
pc_start = time()
selected_variables = list(range(n_vars))
splitted_jobs = None
results = []
if COMM.rank == 0:
    splitted_jobs = _split(selected_variables, COMM.size)
scattered_jobs = COMM.scatter(splitted_jobs, root=0)
## 3.2 Initiate parallel causal discovery, and generate the interaction dict (identified_edges_with_name)
cond_ind_test = ChiSquare()
for j in scattered_jobs:
    (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(j=j, dataframe=dataframe, cond_ind_test=cond_ind_test,\
                                                        selected_links=selected_links, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,\
                                                        max_conds_dim=max_conds_dim, maximum_comb=maximum_comb, verbosity=-1, debugging_pair=None)
    filtered_parents_of_j = parents_of_j.copy()
    #n_edges = min(len(filtered_parents_of_j[j]), max_n_edges) # Only select top max_n_edges causal edges with maximum statistic values
    #if n_edges > 0:
    #    filtered_parents_of_j[j] = filtered_parents_of_j[j][0: n_edges]
    results.append((j, pcmci_of_j, filtered_parents_of_j))
results = MPI.COMM_WORLD.gather(results, root=0)
pc_consumed_time = _elapsed_minutes(pc_start)

# After the discovery process finishes, exit all processes and only keep the process with RANK 0
if COMM.rank != 0:
    exit()

## 4. Gather the distributed pc-discovery result, and initiate the mining evaluations
identified_edges, temporal_pc_array, identified_edge_infos, filtered_edge_infos = gather_pc_results()
with open('{}{}'.format(event_preprocessor.result_path, 'identified-edges'), 'w+') as convert_file:
    convert_file.write(json.dumps(identified_edges))
evaluate_mining_results(temporal_pc_array, identified_edge_infos, filtered_edge_infos)

# 5. Bayesian Fitting and generate the parameterized interaction graph.
identified_edges_with_name = defaultdict(list)
index_device_dict:'dict[DevAttribute]' = event_preprocessor.index_device_dict
for index, x in np.ndenumerate(temporal_pc_array):
    if x == 1:
        outcome_name = index_device_dict[index[1]].name; cause_name = index_device_dict[index[0]].name
        lag = index[2]
        identified_edges_with_name[outcome_name].append((cause_name, lag))
bf_start = time()
bayesian_fitter = BayesianFitter(frame, tau_max, identified_edges, n_max_edges=n_max_edges)
bf_consumed_time = _elapsed_minutes(bf_start)

# 6. Security monitoring
security_guard = SecurityGuard(frame=frame, bayesian_fitter=bayesian_fitter, sig_level=sig_level)
## 6.1 Initialize the phantom state machine using the last training events
latest_event_states = [frame.training_events_states[-tau_max+i] for i in range(0, tau_max)]
machine_initial_states = [event_state[1] for event_state in latest_event_states]
security_guard.initialize_phantom_machine(machine_initial_states)
## JC TODO: 6.2 Inject anomalies to testing datasets
testing_event_states = frame.testing_events_states
## 6.3 Initiate the contextual anomaly detection
anomaly_scores = []
for evt_id, tup in enumerate(testing_event_states):
    event, states = tup
    anomaly_score, anomaly_flag = security_guard.contextual_anomaly_detection(evt_id, event)
    anomaly_scores.append(anomaly_score)
    security_guard.phantom_state_machine.set_latest_states(states)
sns.displot(anomaly_scores, kde=False, color='red', bins=1000)
plt.axvline(x=security_guard.score_threshold)
plt.title('Testing score distribution')
plt.xlabel('Scores')
plt.ylabel('Occurrences')
plt.savefig("temp/image/testing-score-distribution-{}.pdf".format(int(sig_level*100)))
plt.close('all')
security_guard.analyze_detection_results()
exit()

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