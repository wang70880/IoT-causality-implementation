import os

os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
from src.event_processing import Hprocessor, Cprocessor
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare
from src.tigramite.tigramite import plotting as tp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from collections import defaultdict
from pprint import pprint
from time import time

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.tigramite.tigramite import data_processing as pp
from src.drawer import Drawer
from src.causal_evaluation import Evaluator
from src.background_generator import BackgroundGenerator

class DataDebugger():

    def __init__(self, dataset, partition_days, training_ratio) -> None:
        self.dataset = dataset; self.partition_days = partition_days; self.training_ratio = training_ratio
        if self.dataset.startswith('hh'):
            self.preprocessor = Hprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio)
        elif self.dataset.startswith('contextact'):
            self.preprocessor = Cprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio, verbosity=0)
        self.preprocessor.initiate_data_preprocessing()
        self.preprocessor.data_loading()
    
    def validate_discretization(self):
        for dev, tup in self.preprocessor.discretization_dict.items():
            val_list = tup[0]; seg_point = tup[1]
            sns.displot(val_list, kde=False, color='red', bins=1000)
            plt.axvline(seg_point, 0, 1)
            plt.title('State distributions of device {}'.format(dev))
            plt.xlabel('State')
            plt.ylabel('Frequency')
            plt.savefig("temp/image/{}_{}_states.pdf".format(self.dataset, dev))
    
class BackgroundDebugger():

    def __init__(self, event_preprocessor, frame_id, tau_max) -> None:
        self.event_preprocessor = event_preprocessor
        self.dataset = dataset
        self.frame_id = frame_id; self.frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
        self.tau_max = tau_max
        self.background_generator:'BackgroundGenerator' = BackgroundGenerator(event_preprocessor, frame_id, tau_max)

class MinerDebugger():
    
    def __init__(self, frame, background_generator:'BackgroundGenerator', alpha) -> None:
        self.frame:'DataFrame' = frame; self.alpha = alpha
        self.background_generator:'BackgroundGenerator' = background_generator
        self.tau_max = background_generator.tau_max
    
    def check_false_positive(self, edge:'tuple'):
        """
        This function debugs a false-positive edge,
        which is supposed to be removed due to mediating variables (e.g., D002->M003->M005)
        1. First check the edge's temporal attributes (Do they sequentially occur frequently?)
        2. Initiate the conditional independence testing: Are they CI given the underlying mediating variables?
        """
        # Auxiliary variables
        name_device_dict:'dict[DevAttribute]' = self.frame.name_device_dict; index_device_dict:'dict[DevAttribute]' = self.frame.index_device_dict
        adjacency_array:'np.ndarray' = self.background_generator.spatial_array
        var_names = self.frame.var_names; n_vars = len(var_names)
        former = name_device_dict[edge[0]].index; latter = name_device_dict[edge[1]].index

        # 1. For each false-positive edge, its activation frequency is low (< .5 * n_days)
        frequency_list = [self.background_generator.frequency_array[former, latter, lag] for lag in range(1, self.tau_max+1)]
        activation_frequency_list = [self.background_generator.activation_frequency_array[former, latter, lag] for lag in range(1, self.tau_max+1)]
        print("Frequencies, activation frequencies of pair ({}, {}): {}, {}".format(edge[0], edge[1], frequency_list, activation_frequency_list))

        # 2. Check the conditional independence testing result
        pcmci = PCMCI(self.frame.training_dataframe, ChiSquare(), verbosity=1)
        neighbors = [x for x in range(n_vars) if np.count_nonzero(adjacency_array[x, latter, :])>0 and x not in [former, latter]]

        vals = []; pvals = []
        for lag in range(1, self.tau_max+1):
            val, pval = pcmci.cond_ind_test.run_test(X=[(former, (-lag))], Y=[(latter, 0)], Z=[], tau_max=self.tau_max)
            vals.append(val); pvals.append(pval)
        print("vals, pvals: {}, {}".format(vals, pvals))

class BayesianDebugger():

    def __init__(self, data_debugger = None, verbosity=1, analyze_golden_standard=True) -> None:
        assert(data_debugger is not None)
        self.data_debugger:'DataDebugger' = data_debugger
        self.verbosity = verbosity
        self.analyze_golden_standard = analyze_golden_standard
    
    def analyze_fitting_result(self, int_frame_id=0):
        # Auxillary variables
        tested_frame:'DataFrame' = self.data_debugger.preprocessor.frame_dict[int_frame_id]
        var_names = tested_frame.var_names; tau_max = self.data_debugger.tau_max
        index_device_dict:'dict[DevAttribute]' = self.data_debugger.preprocessor.index_device_dict
        # 1. Get the interaction matrix and transform to link dict
        if self.analyze_golden_standard:
            interaction_matrix:'np.ndarray' = self.data_debugger.derive_golden_standard(int_frame_id, 'user')
        else:
            miner_debugger:'MinerDebugger' = MinerDebugger(alpha=0.001, data_debugger=self.data_debugger)
            interaction_matrix:'np.ndarray' = miner_debugger.initiate_discovery_algorithm(int_frame_id)
        link_dict = defaultdict(list)
        for (i, j, lag), x in np.ndenumerate(interaction_matrix):
            if x == 1:
                link_dict[var_names[j]].append((var_names[i],-lag))
        #print("[Bayesian Debugger] Interaction dict to be analyzed:")
        #pprint(dict(link_dict))

        # 2. Initiate BayesianFitter class and estimate models
        bayesian_fitter = BayesianFitter(tested_frame, tau_max, link_dict)
        bayesian_fitter.construct_bayesian_model()

        # 3. For each edge, calculate its average activation-activation frequency
        activation_ratios = []; deactivation_ratios = []
        average_cpds = np.zeros((2, 2)); count = 0
        for outcome, cause_list in link_dict.items(): 
            for (cause, lag) in cause_list: # For each edge, estimate its cpt
                cause_name = bayesian_fitter._lag_name(cause, lag)
                assert(bayesian_fitter.expanded_causal_graph[bayesian_fitter.extended_name_device_dict[cause_name].index,\
                                                             bayesian_fitter.extended_name_device_dict[outcome].index] == 1)
                model = BayesianNetwork([(cause_name, outcome)])
                model.fit(pd.DataFrame(data=bayesian_fitter.expanded_data_array, columns=bayesian_fitter.expanded_var_names),\
                            estimator=MaximumLikelihoodEstimator)
                cpd = model.get_cpds(outcome)
                activation_ratio = cpd.values[(-1,) * len(cpd.variables)]; deactivation_ratio = cpd.values[(0,) * len(cpd.variables)]
                activation_ratios.append(activation_ratio); deactivation_ratios.append(deactivation_ratio)
                average_cpds += cpd.values
                count += 1
        average_cpds /= count
        if self.verbosity > 0:
            print("[Bayesian Debugger] Average cpds for golden standard edges:\n{}".format(average_cpds))
        drawer = Drawer()
        drawer.draw_1d_distribution(activation_ratios, fname='activation-cpt')
        drawer.draw_1d_distribution(deactivation_ratios, fname='deactivation-cpt')

        return bayesian_fitter

class GuardDebugger():

    def __init__(self, data_debugger=None, bayesian_debugger=None) -> None:
        self.data_debugger:'DataDebugger' = data_debugger
        self.bayesian_debugger:'BayesianDebugger' = bayesian_debugger
        self.drawer:'Drawer' = Drawer()

    def analyze_anomaly_threshold(self, int_frame_id=0, sig_level=1):
        frame = self.data_debugger.preprocessor.frame_dict[int_frame_id]
        bayesian_fitter = self.bayesian_debugger.analyze_fitting_result(int_frame_id)
        # 1. Initialize the security guard, and calculate anomaly scores
        security_guard = SecurityGuard(frame=frame, bayesian_fitter=bayesian_fitter, sig_level=sig_level)
        training_anomaly_scores = security_guard.training_anomaly_scores
        #print("[Guard Debugger] large-pscore-dict:"); pprint(dict(security_guard.large_pscore_dict))
        filterd_large_pscore_dict = {k:v for k, v in dict(security_guard.large_pscore_dict).items() if v > 1000}
        print("[Guard Debugger] filtered large-pscore-dict:"); pprint(filterd_large_pscore_dict)
        #print("[Guard Debugger] small-pscore-dict:"); pprint(dict(security_guard.small_pscore_dict))
        print("Average anomaly score = {}".format(sum(training_anomaly_scores)*1.0/len(training_anomaly_scores)))
        
        self.drawer.draw_1d_distribution(training_anomaly_scores, xlabel='Score', ylabel='Occurrence',\
            title='Training event detection using golden standard model', fname='prediction-probability-distribution')
        return security_guard
    
    def score_anomaly_detection(self, int_frame_id, sig_level=None, n_anomalies=None, maximum_length=None, anomaly_case=None):
        frame = self.data_debugger.preprocessor.frame_dict[int_frame_id]
        event_preprocessor = self.data_debugger.preprocessor; tau_max = event_preprocessor.tau_max
        background_generator = self.data_debugger.background_generator
        bayesian_fitter = self.bayesian_debugger.analyze_fitting_result(int_frame_id)

        # 1. Initialize the security guard, and derive the anomaly score threshold
        security_guard = SecurityGuard(frame=frame, bayesian_fitter=bayesian_fitter, sig_level=sig_level)
        print("The anomaly score threshold is {}".format(security_guard.score_threshold))
        # 2. Inject anomalies and generate testing event sequences
        evaluator = Evaluator(event_processor=event_preprocessor, background_generator=background_generator,\
                                             bayesian_fitter = bayesian_fitter, tau_max=tau_max)
        testing_event_states, anomaly_positions, testing_benign_dict = evaluator.simulate_malicious_control(\
                                    int_frame_id=int_frame_id, n_anomaly=n_anomalies, maximum_length=maximum_length, anomaly_case=anomaly_case)
        print("# of testing events: {}".format(len(testing_event_states)))
        # 3. initialize the testing
        anomaly_flag = False
        for event_id in range(len(testing_event_states)):
            event, states = testing_event_states[event_id]
            if event_id < tau_max:
                security_guard.initialize(event_id, event, states)
            else:
                anomaly_flag = security_guard.score_anomaly_detection(event_id=event_id, event=event, debugging_id_list=anomaly_positions)
            security_guard.calibrate(event_id, testing_benign_dict)

        tps, fps, fns = security_guard.analyze_detection_results()
        precision = 1.0 * tps / (fps + tps); recall = 1.0 * tps / (fns + tps)
        f1_score = 2.0 * precision * recall / (precision + recall)
        return precision, recall, f1_score

class EvaluationResultRepo():

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def discovery_accuracy_evaluation(self):
        """
        1. Calculate the precision and recall (comparisons with golden standards).
        2. Consumed time for discovery.
        3. Answer for false positives.
        4. Types of discovered interactions.
        5. Identified device interaction chains.
        """
        n_records = 173094; n_devices = 8; n_golden_edges = 42
        bk_nedges_dict = {0: 192, 1: 141, 2: 118}
        # The discovered result dict.
        bk_alpha_results_dict = {
            (0, 0.00001): {'precision':0.787, 'recall':0.881}, (0, 0.001): {'precision':0.78, 'recall':0.929}, (0, 0.1): {'precision':0.759, 'recall':0.976},
            (1, 0.00001):{'precision':0.804, 'recall':0.881}, (1, 0.001): {'precision':0.809, 'recall':0.905}, (1, 0.1):  {'precision':0.817, 'recall':0.952},
            (2, 0.00001): {'precision':1.0, 'recall':0.881}, (2, 0.001): {'precision':1.0, 'recall':0.905}, (2, 0.1): {'precision':1.0, 'recall':0.952}
        }
        precision_lists = [[bk_alpha_results_dict[(bk, alpha)]['precision']\
                            for bk in range(0, 3) ] for alpha in [0.00001, 0.001, 0.1]]
        recall_lists = [[bk_alpha_results_dict[(bk, alpha)]['recall']\
                            for bk in range(0, 3) ] for alpha in [0.00001, 0.001, 0.1]]
        return precision_lists, recall_lists
    
    def discovery_complexity_evaluation(self):
        n_devices = 8; n_golden_edges = 42
        bk_dsize_results_dict = {
            (0, 20): {'precision':0.783, 'recall':0.857, 'f1':0.818, 'time':0.707}, (0, 30): {'precision':0.787, 'recall':0.881, 'f1':0.831, 'time':0.806}, (0, 40): {'precision':0.787, 'recall':0.881, 'f1':0.831, 'time':0.994}, (0, 50): {'precision':0.771, 'recall':0.881, 'f1':0.822, 'time':1.065}, (0, 60): {'precision':0.765, 'recall':0.929, 'f1':0.839, 'time':1.220}, (0, 70): {'precision':0.760, 'recall':0.905, 'f1':0.826, 'time':1.289}, (0, 80): {'precision':0.769, 'recall':0.952, 'f1':0.851, 'time':1.439}, (0, 90): {'precision':0.760, 'recall':0.905, 'f1':0.826, 'time':1.553}, (0, 100): {'precision':0.759, 'recall':0.976, 'f1':0.854, 'time':1.837}, (0, 110): {'precision':0.719, 'recall':0.976, 'f1':0.828, 'time':1.993}, (0, 120): {'precision':0.695, 'recall':0.976, 'f1':0.812, 'time':2.282},
            (1, 20): {'precision':0.804, 'recall':0.881, 'f1':0.841, 'time':0.652}, (1, 30): {'precision':0.800, 'recall':0.857, 'f1':0.828, 'time':0.798}, (1, 40): {'precision':0.804, 'recall':0.881, 'f1':0.841, 'time':0.945}, (1, 50): {'precision':0.804, 'recall':0.881, 'f1':0.841, 'time':1.014}, (1, 60): {'precision':0.809, 'recall':0.905, 'f1':0.854, 'time':1.118}, (1, 70): {'precision':0.809, 'recall':0.905, 'f1':0.854, 'time':1.236}, (1, 80): {'precision':0.809, 'recall':0.905, 'f1':0.854, 'time':1.404}, (1, 90): {'precision':0.804, 'recall':0.881, 'f1':0.841, 'time':1.442}, (1, 100): {'precision':0.816, 'recall':0.952, 'f1':0.879, 'time':1.747}, (1, 110): {'precision':0.800, 'recall':0.952, 'f1':0.870, 'time':1.704}, (1, 120): {'precision':0.804, 'recall':0.976, 'f1':0.882, 'time':2.120},
            (2, 20): {'precision':1.000, 'recall':0.881, 'f1':0.937, 'time':0.591}, (2, 30): {'precision':1.000, 'recall':0.857, 'f1':0.923, 'time':0.609}, (2, 40): {'precision':1.000, 'recall':0.881, 'f1':0.937, 'time':0.662}, (2, 50): {'precision':1.000, 'recall':0.881, 'f1':0.937, 'time':0.778}, (2, 60): {'precision':1.000, 'recall':0.905, 'f1':0.950, 'time':0.845}, (2, 70): {'precision':1.000, 'recall':0.905, 'f1':0.950, 'time':0.925}, (2, 80): {'precision':1.000, 'recall':0.905, 'f1':0.950, 'time':1.063}, (2, 90): {'precision':1.000, 'recall':0.881, 'f1':0.937, 'time':1.108}, (2, 100): {'precision':1.000, 'recall':0.952, 'f1':0.976, 'time':1.469}, (2, 110): {'precision':1.000, 'recall':0.952, 'f1':0.976, 'time':1.561}, (2, 120): {'precision':1.000, 'recall':0.976, 'f1':0.988, 'time':1.901},
        }
        precision_lists = [[bk_dsize_results_dict[(bk, size)]['precision'] for size in range(20, 110, 10)] for bk in range(0, 3)]
        recall_lists = [[bk_dsize_results_dict[(bk, size)]['recall'] for size in range(20, 110, 10)] for bk in range(0, 3)]
        f1_lists = [[bk_dsize_results_dict[(bk, size)]['f1'] for size in range(20, 110, 10)] for bk in range(0, 3)]
        time_lists = [[bk_dsize_results_dict[(bk, size)]['time'] for size in range(20, 110, 10)] for bk in range(0, 3)]
        return precision_lists, recall_lists, f1_lists, time_lists

if __name__ == '__main__':

    dataset = 'contextact'; partition_days = 8; training_ratio = 0.8; tau_max = 3
    alpha = 0.01; int_frame_id = 0; analyze_golden_standard=False

    data_debugger = DataDebugger(dataset, partition_days, training_ratio)
    #data_debugger.validate_discretization()
    frame:'DataFrame' = data_debugger.preprocessor.frame_dict[int_frame_id]

    background_generator:'BackgroundGenerator' = BackgroundGenerator(data_debugger.preprocessor, int_frame_id, tau_max)
    background_generator.print_background_knowledge()
    exit()

    evaluator = Evaluator(data_debugger.preprocessor, background_generator, None, 0, alpha)
    evaluator.print_golden_standard('aggregation')

    miner_debugger = MinerDebugger(frame, background_generator, alpha)
    fp_edges = [('D002', 'M006')]
    for fp_edge in fp_edges:
        miner_debugger.check_false_positive(fp_edge)

    # 2. Evaluate causal discovery
    evaluation_repo = EvaluationResultRepo(dataset)
    drawer = Drawer(dataset)
    ## 2.1 Causal discovery accuracy result
    precision_lists, recall_lists = evaluation_repo.discovery_accuracy_evaluation()
    legends = ['alpha=1e-5', 'alpha=1e-3', 'alpha=1e-1']; groups = ['Pure-PC', 'BK-Temporal', 'BK-Spatial']
    drawer.draw_histogram(precision_lists, legends, groups, 'Background', 'Precision')
    drawer.draw_histogram(recall_lists, legends, groups, 'Background', 'Recall')
    ## 2.2 Causal discovery learning complexity result
    precision_lists, recall_lists, f1_lists, time_lists = evaluation_repo.discovery_complexity_evaluation()
    x_list = [i for i in range(20, 110, 10)]
    legends = ['Pure-PC', 'BK-Temporal', 'BK-Spatial']
    drawer.draw_line_chart(x_list, precision_lists, legends, 'Data-size', 'Precision')
    drawer.draw_line_chart(x_list, recall_lists, legends, 'Data-size', 'Recall')
    drawer.draw_line_chart(x_list, f1_lists, legends, 'Data-size', 'F1')
    drawer.draw_line_chart(x_list, time_lists, legends, 'Data-size', 'Time')

    #n_anomalies = 50; maximum_length = 1; anomaly_case = 1
    #sig_levels = list(np.arange(0.1, 1., 0.1)); sig_levels = [0.95]
    #precisions = []; recalls = []; f1_scores = []
    #for sig_level in sig_levels:
    #    guard_debugger = GuardDebugger(data_debugger, bayesian_debugger)
    #    precision, recall, f1_score = guard_debugger.score_anomaly_detection(int_frame_id, sig_level, n_anomalies, maximum_length, anomaly_case)
    #    precisions.append(precision); recalls.append(recall); f1_scores.append(f1_score)
    #drawer = Drawer()
    #drawer.draw_2d_distribution(sig_levels, [precisions, recalls, f1_scores], ['precision', 'recall', 'f1-score'], 'sig-level', 'value', 'No', 'sig-level-analysis')