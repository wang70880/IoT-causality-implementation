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
from src.benchmark.markov_miner import MarkovMiner
from src.benchmark.iotwatcher import HAWatcher
from src.benchmark.bayesian_network import BayesianMiner

from src.tigramite.tigramite import data_processing as pp
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare

from src.background_generator import BackgroundGenerator
from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.causal_evaluation import Evaluator
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.benchmark.association_rule_miner import ARMMiner
from src.benchmark.ocsvm import OCSVMer

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
    tau_min=1, tau_max=1, pc_alpha = 0.1,verbosity=-1,\
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

def gather_pc_results():
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
    normalized_temporal_pc_array:'np.ndarray' = _normalize_temporal_array(temporal_pc_array, tau_max)

    filtered_edge_infos = defaultdict(list) # For each filtered edge, collect its information (edge, condition, val, pval)
    for outcome, filtered_edges_dict in filtered_edges.items():
        for edge, edge_infos in filtered_edges_dict.items():
                filtered_edge_infos[outcome].append((edge, edge_infos['conds'], edge_infos['val'], edge_infos['pval']))
    #pprint(identified_edges)
    return identified_edges, temporal_pc_array, normalized_temporal_pc_array, identified_edge_infos, filtered_edge_infos

# 0. Parameter settings
    # 0.1 Data-loading parameters
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3]); frame_id = 0 # JC NOTE: By default, we use the first data frame
    # 0.2 Background knowledge parameters
tau_max = int(sys.argv[4]); tau_min = 1
    # 0.3 PC discovery parameters
pc_alpha = float(sys.argv[5]); max_conds_dim = int(sys.argv[6]); maximum_comb = int(sys.argv[7]); n_max_edges = 15
    # 0.5 Anomaly generation parameters
sig_level = 0.99

# 1. Load data and create data frame
dl_start = time()
event_preprocessor:'GeneralProcessor' = Hprocessor(dataset, partition_days, training_ratio) if dataset.startswith('hh') else Cprocessor(dataset, partition_days, training_ratio)
event_preprocessor.data_loading()
frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
dataframe:pp.DataFrame = frame.training_dataframe; var_names = frame.var_names
n_records = dataframe.T; n_vars = dataframe.N
preprocessing_consumed_time = _elapsed_minutes(dl_start)
if COMM.rank == 0:
    print("\n********************** Parameter Settings **********************\n"\
     + "    pc-alpha={}, partition_days={}, n_training_events={}, n_testing_events={}, max(dim)={}, max(comb)={}"\
            .format(pc_alpha, partition_days, len(frame.training_events_states), len(frame.testing_events_states), max_conds_dim, maximum_comb))
    print("\n*** Dataset Overview ***")
    print("     Number of training records, devices, and attributes = {}, {}, {}".format(n_records, frame.n_vars, frame.n_attrs))
    print("     device list={}".format(frame.var_names))
    print("     attribute list={}".format(frame.attr_names))

"""Initiate the interaction mining process"""

# 1. Initiate parallel causal discovery
## 1.1 Scatter the jobs
pc_start = time()
splitted_jobs = None
if COMM.rank == 0:
    splitted_jobs = _split(list(range(n_vars)), COMM.size)
scattered_jobs = COMM.scatter(splitted_jobs, root=0)
## 1.2 Initiate parallel causal discovery, and generate the interaction dict
results = []
for j in scattered_jobs:
    selected_links = {n: {m: [(i, -t) for i in range(n_vars) for \
                t in range(1, tau_max+1)] if m == n else [] for m in range(n_vars)} for n in range(n_vars)}
    (j, pcmci_of_j, parents_of_j) = _run_pc_stable_parallel(\
        j=j, dataframe=dataframe, cond_ind_test=ChiSquare(),\
        selected_links=selected_links, tau_max=tau_max, pc_alpha=pc_alpha,\
        max_conds_dim=max_conds_dim, maximum_comb=maximum_comb)
    results.append((j, pcmci_of_j, parents_of_j))
results = MPI.COMM_WORLD.gather(results, root=0)
pc_time = _elapsed_minutes(pc_start)

if COMM.rank != 0:
    exit()

## 1.3 Gather pc discovery result and write to the file
causal_edges, causal_array, nor_causal_array, causal_edge_infos, causal_filtered_edge_infos = gather_pc_results()
with open('{}{}'.format(event_preprocessor.result_path, 'identified-edges'), 'w+') as convert_file:
    convert_file.write(json.dumps(causal_edges))

degrees = [(k, len(v)) for k, v in causal_edges.items()]
avg_degree = statistics.mean([tup[1] for tup in degrees])
max_degree = (0, 0); min_degree = (0, 100000)
for degree in degrees:
    if degree[1] > max_degree[1]:
        max_degree = degree
    if degree[1] < min_degree[1]:
        min_degree = degree
print("Discovery process finished.")
print("     Average degrees: {}".format(avg_degree))
print("     Variables with maximum degrees: {} {}".format(var_names[max_degree[0]], max_degree[1]))
print("     Variables with minimum degrees: {} {}".format(var_names[min_degree[0]], min_degree[1]))

"""Initiate the interaction mining evaluation"""
evaluator = Evaluator(event_preprocessor=event_preprocessor, frame=frame, tau_max=tau_max, pc_alpha=pc_alpha, causal_edges=causal_edges)
hawatcher = HAWatcher(event_preprocessor, frame, tau_max, 0.95)
causal_tp, causal_fp, causal_fn, causal_precision, causal_recall, causal_f1 = evaluator.evaluate_discovery_accuracy(nor_causal_array, evaluator.nor_golden_array, causal_filtered_edge_infos, causal_edge_infos, 'causal', hawatcher.background_generator, 1)

"""Initiate the parameter estimation process"""
causal_bayesian_fitter = BayesianFitter(frame, tau_max, causal_edges, n_max_edges=n_max_edges, model_name='causal')
causal_bayesian_fitter.bayesian_parameter_estimation()
causal_security_guard = SecurityGuard(frame, causal_bayesian_fitter, sig_level)

ground_truth_fitter = BayesianFitter(frame, tau_max, evaluator.golden_edges, n_max_edges, model_name='Golden')
ground_truth_fitter.bayesian_parameter_estimation()

"""Initiate anomaly injection, detection, and evaluation"""
# Anomaly detection benchmarks:
    # Stochastic-based: Markov chain
    # Rule-based: HAWatcher
    # ML-based: OCSVM
markov_start = time()
markov_miner = MarkovMiner(frame, tau_max, sig_level)
markov_time = _elapsed_minutes(markov_start)
ocsvm_start = time()
ocsvmer = OCSVMer(frame, tau_max)
ocsvm_time = _elapsed_minutes(ocsvm_start)
hawatcher_start = time()
hawatcher = HAWatcher(event_preprocessor, frame, tau_max, 0.95)
haw_time = _elapsed_minutes(hawatcher_start)
#print("[Training Efficiency] {} v.s. {} v.s. {} v.s. {}".format(pc_time, markov_time, ocsvm_time, haw_time))

# JC TEST: Check the FP/FN for the contextual anomaly detection
#n_anomalies_dict = {1:4000}
#for case_id in range(1, 2):
#    testing_event_states, anomaly_positions, testing_benign_dict = evaluator.inject_contextual_anomalies(ground_truth_fitter, sig_level, n_anomalies_dict[case_id], case_id)
#    evaluator.analyze_false_contextual_results(causal_bayesian_fitter, ground_truth_fitter, sig_level, case_id,
#                                                testing_event_states, anomaly_positions, testing_benign_dict,
#                                                markov_miner, ocsvmer, hawatcher)

# Anomaly injection, detection, and evaluation
n_anomalies_dict = {0:4000, 1:4000, 2:4000, 3:3000}
n_loops = 10
for case_id in range(1, 2):
    causal_results = []; markov_results = []; ocsvm_results = []; haw_results = []
    for i in range(n_loops):
        testing_event_states, anomaly_positions, testing_benign_dict = evaluator.inject_contextual_anomalies(ground_truth_fitter, sig_level, n_anomalies_dict[case_id], case_id)
        causal_alarm_position_events = causal_security_guard.kmax_anomaly_detection(testing_event_states, testing_benign_dict, kmax=1)
        causal_alarm_position_events = [item for chain in causal_alarm_position_events for item in chain]
        markov_alarm_position_events = markov_miner.anomaly_detection(testing_event_states, testing_benign_dict, kmax=1)
        ocsvm_alarm_position_events = ocsvmer.anomaly_detection(testing_event_states)
        hawatcher_alarm_position_events = hawatcher.anomaly_detection(testing_event_states, testing_benign_dict)
        #evaluator.analyze_false_contextual_results(causal_bayesian_fitter, ground_truth_fitter, sig_level, case_id, testing_event_states, anomaly_positions, testing_benign_dict)

        causal_precision, causal_recall, causal_f1 = evaluator.evaluate_contextual_detection_accuracy(causal_alarm_position_events, anomaly_positions, case_id, 'causal')
        markov_precision, markov_recall, markov_f1 = evaluator.evaluate_contextual_detection_accuracy(markov_alarm_position_events, anomaly_positions, case_id, 'markov')
        ocsvm_precision, ocsvm_recall, ocsvm_f1 = evaluator.evaluate_contextual_detection_accuracy(ocsvm_alarm_position_events, anomaly_positions, case_id, 'ocsvm')
        haw_precision, haw_recall, haw_f1 = evaluator.evaluate_contextual_detection_accuracy(hawatcher_alarm_position_events, anomaly_positions, case_id, 'hawatcher')
        causal_results.append((causal_precision, causal_recall, causal_f1))
        markov_results.append((markov_precision, markov_recall, markov_f1))
        ocsvm_results.append((ocsvm_precision, ocsvm_recall, ocsvm_f1))
        haw_results.append((haw_precision, haw_recall, haw_f1))
    print("[Case {}] Average detection accuracy for Causal, Markov, OCSVM, HAW".format(case_id))
    print("{} {} {}".format(statistics.mean([x[0] for x in causal_results]), statistics.mean([x[1] for x in causal_results]), statistics.mean([x[2] for x in causal_results])))
    print("{} {} {}".format(statistics.mean([x[0] for x in markov_results]), statistics.mean([x[1] for x in markov_results]), statistics.mean([x[2] for x in markov_results])))
    print("{} {} {}".format(statistics.mean([x[0] for x in ocsvm_results]), statistics.mean([x[1] for x in ocsvm_results]), statistics.mean([x[2] for x in ocsvm_results])))
    print("{} {} {}".format(statistics.mean([x[0] for x in haw_results]), statistics.mean([x[1] for x in haw_results]), statistics.mean([x[2] for x in haw_results])))
#
#exit()

# JC TEST: test the collective anomaly chain generation
n_anomalies_dict = {0:1000, 1:1000, 2:1000, 3:1000}
for kmax in range(2, 5):
    for case_id in range(0, 1):
        print("[Case {}, KMAX {} Step {}] Chain reconstruction accuracy for Causal, Markov, OCSVM, HAW".format(case_id, kmax, 2*tau_max))
        testing_event_states, position_len_dict, testing_benign_dict = evaluator.inject_collective_anomalies(ground_truth_fitter, sig_level, n_anomalies_dict[case_id], case_id, kmax, step=2*tau_max)
        causal_alarm_position_chains = causal_security_guard.kmax_anomaly_detection(testing_event_states, testing_benign_dict, kmax=kmax, debugging_dict=position_len_dict)
        causal_alarm_position_len_dict = {chain_list[0][0]: len(chain_list) for chain_list in causal_alarm_position_chains}
        causal_len_dict, missing_alarm_dict = evaluator.evaluate_collective_detection_accuracy(causal_alarm_position_len_dict , position_len_dict, kmax, case_id, 'causal')
        print("Missing alarm analysis")
        missing_alarm_infos = defaultdict(int)
        for pos, l in missing_alarm_dict.items():
            cur_chain = ''
            for i in range(0, l):
                cur_event = testing_event_states[pos+i][0]
                cur_chain += '{}:{},'.format(cur_event.dev, cur_event.value)
            missing_alarm_infos[cur_chain] += 1
        pprint(missing_alarm_infos)
        print("{}".format(causal_len_dict))
