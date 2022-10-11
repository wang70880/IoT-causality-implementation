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
from src.benchmark.iotwatcher import IoTWatcher
from src.benchmark.association_miner import AssociationMiner

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
tau_max = int(sys.argv[4]); tau_min = 1
    # 0.3 PC discovery parameters
pc_alpha = float(sys.argv[5]); max_conds_dim = int(sys.argv[6]); maximum_comb = int(sys.argv[7])
    # 0.5 Anomaly generation parameters
n_anomalies = 8000; sig_level = 0.99

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

# 3. Initiate parallel causal discovery
## 3.1 Scatter the jobs
pc_start = time()
splitted_jobs = None
if COMM.rank == 0:
    splitted_jobs = _split(list(range(n_vars)), COMM.size)
scattered_jobs = COMM.scatter(splitted_jobs, root=0)
## 3.2 Initiate parallel causal discovery, and generate the interaction dict
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
pc_consumed_time = _elapsed_minutes(pc_start)

if COMM.rank != 0:
    exit()

## 3.3 Gather pc discovery result and write to the file
causal_edges, causal_array, nor_causal_array, causal_edge_infos, causal_filtered_edge_infos = gather_pc_results()
with open('{}{}'.format(event_preprocessor.result_path, 'identified-edges'), 'w+') as convert_file:
    convert_file.write(json.dumps(causal_edges))

# ADDITION  Initiate other benchmark methods for interaction mining
asso_miner = AssociationMiner(event_preprocessor, frame, tau_max, pc_alpha)
hawatcher_miner = IoTWatcher(event_preprocessor, frame, tau_max)
evaluator = Evaluator(event_preprocessor, frame, tau_max, pc_alpha)

"""Initiate the discovery evaluation"""

causal_tp, causal_fp, causal_fn, causal_precision, causal_recall, causal_f1 = evaluator.evaluate_discovery_accuracy(nor_causal_array, evaluator.nor_golden_array, causal_filtered_edge_infos, causal_edge_infos, 0)
asso_tp, asso_fp, asso_fn, asso_precision, asso_recall, asso_f1 = evaluator.evaluate_discovery_accuracy(asso_miner.nor_mining_array, evaluator.nor_golden_array, 0)
haw_tp, haw_fp, haw_fn, haw_precision, haw_recall, haw_f1 = evaluator.evaluate_discovery_accuracy(hawatcher_miner.nor_mining_array, evaluator.nor_golden_array, 0)
print("*** Interaction mining evaluation for {} ground truth interactions ***".format(causal_tp+causal_fn))
print("[Precision] {} v.s. {} v.s. {}".format(causal_precision, asso_precision, haw_precision))
print("[Recall] {} v.s. {} v.s. {}".format(causal_recall, asso_recall, haw_recall))
print("[F1] {} v.s. {} v.s. {}".format(causal_f1, asso_f1, haw_f1))

# 4. Bayesian fitting and anomaly detection

for n_max_edges in range(10, 11):
    print("*** Contextual anomaly detection evaluation for n-max-edges={}***".format(n_max_edges))
    ground_truth_fitter = BayesianFitter(frame, tau_max, evaluator.golden_edges, n_max_edges, model_name='Golden')

    # Anomaly injection
    testing_event_states, anomaly_positions, anomaly_infos, testing_benign_dict = evaluator.inject_contextual_anomalies(ground_truth_fitter, sig_level, n_anomalies, case_id=0)

    causal_bayesian_fitter = BayesianFitter(frame, tau_max, causal_edges, n_max_edges=n_max_edges, model_name='causal')
    asso_bayesian_fitter = BayesianFitter(frame, tau_max, asso_miner.mining_edges, n_max_edges=n_max_edges, model_name='association')
    haw_bayesian_fitter = BayesianFitter(frame, tau_max, hawatcher_miner.mining_edges, n_max_edges=n_max_edges, model_name='HAWatcher')

    causal_security_guard = SecurityGuard(frame, causal_bayesian_fitter, sig_level)
    asso_security_guard = SecurityGuard(frame, asso_bayesian_fitter, sig_level)
    haw_security_guard = SecurityGuard(frame, haw_bayesian_fitter, sig_level)

    causal_alarms, causal_alarm_infos = causal_security_guard.contextual_anomaly_detection(testing_event_states, testing_benign_dict)
    asso_alarms, asso_alarm_infos = asso_security_guard.contextual_anomaly_detection(testing_event_states, testing_benign_dict)
    haw_alarms, haw_alarm_infos = haw_security_guard.contextual_anomaly_detection(testing_event_states, testing_benign_dict)

    evaluator.evaluate_contextual_detection_accuracy(anomaly_infos, causal_alarm_infos, 'causal')
    evaluator.evaluate_contextual_detection_accuracy(anomaly_infos, asso_alarm_infos, 'association')
    evaluator.evaluate_contextual_detection_accuracy(anomaly_infos, haw_alarm_infos, 'Hawatcher')
exit()