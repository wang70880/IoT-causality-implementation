from src.event_processing import Hprocessor, Cprocessor
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare
from src.tigramite.tigramite import plotting as tp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from collections import defaultdict
from pprint import pprint
from time import time

import seaborn as sns
import numpy as np
import ruptures as rpt
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
            self.preprocessor = Cprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio, verbosity=1)
        #self.preprocessor.initiate_data_preprocessing()
        self.preprocessor.data_loading()
    
    def validate_discretization(self):
        for dev, val_list in self.preprocessor.continuous_dev_dict.items():
            plt.plot([x for x in range(len(val_list))], val_list)
            plt.title("{} state".format(dev))
            plt.savefig("temp/image/{}-state.pdf".format(dev))
            plt.close('all')

            algo = rpt.Pelt(model="rbf").fit(np.array(val_list))
            result = algo.predict(pen=int(len(val_list)/500)+1) # JC NOTE: Ad-hoc parameter settings here.
            rpt.display(np.array(val_list), [], result)
            #plt.plot([x for x in range(len(val_list))], val_list)
            #sns.displot(val_list, kde=False, color='red', bins=1000)
            #plt.axvline(seg_point, 0, 1)
            #plt.title('State distributions of device {}'.format(dev))
            #plt.xlabel('State')
            #plt.ylabel('Frequency')
            plt.title("{} state changepoint".format(dev))
            plt.savefig("temp/image/{}-changepoint.pdf".format(dev))
            plt.close('all')
    

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

    def __init__(self, frame, link_dict, tau_max, verbosity=1) -> None:
        self.frame:'DataFrame' = frame
        self.link_dict:'dict' = link_dict
        self.tau_max = tau_max
        self.verbosity = verbosity

        self.bayesian_fitter = BayesianFitter(frame, tau_max, link_dict)

    def test_backdoor_adjustment(self):
        var_names = self.frame.var_names; n_vars = len(var_names)
        n_lagged_vars = n_vars * (2 * self.tau_max + 1)
        edges = []; extended_edges = []
        for lag in range(0, self.tau_max+1):
            for outcome, cause_list in self.link_dict.items():
                if lag == 0:
                    edges += [(tuple(cause), (outcome, 0)) for cause in cause_list]
                extended_edges = extended_edges + [((cause[0], cause[1]-lag), (outcome, 0-lag)) for cause in cause_list]
        str_edges = [('{}({})'.format(cause[0], cause[1]), '{}({})'.format(outcome[0], outcome[1]))\
                            for (cause, outcome) in edges]
        extended_str_edges = [('{}({})'.format(cause[0], cause[1]), '{}({})'.format(outcome[0], outcome[1]))\
                            for (cause, outcome) in extended_edges]
        assert(all([str_edge in extended_str_edges for str_edge in str_edges]))
        
        model = BayesianNetwork(extended_str_edges)
        inference = CausalInference(model)
        for str_edge in str_edges:
            print("For edge {}:".format(str_edge))
            minimal_adjustment_set = inference.get_minimal_adjustment_set(str_edge[0], str_edge[1])
            print(" * Is there any minimal backdoor path? Length {} for {}".format(len(minimal_adjustment_set), minimal_adjustment_set))
    
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

if __name__ == '__main__':

    dataset = 'contextact'; partition_days = 8; training_ratio = 0.8; tau_max = 3
    alpha = 0.001; int_frame_id = 0; analyze_golden_standard=False

    data_debugger = DataDebugger(dataset, partition_days, training_ratio)
    frame:'DataFrame' = data_debugger.preprocessor.frame_dict[int_frame_id]
    link_dict = {"0": [[0, -1], [0, -2], [0, -3], [17, -1], [17, -3], [20, -1], [1, -1], [26, -1], [26, -2]], "1": [[1, -1], [1, -2], [1, -3], [24, -1], [17, -1], [24, -2], [17, -3], [24, -3]], "2": [[2, -2], [17, -1], [26, -1], [17, -3], [26, -3], [21, -1], [26, -2]], "3": [[3, -2], [18, -2], [18, -1]], "4": [[4, -1], [19, -1], [19, -3], [7, -1]], "5": [[5, -2], [16, -2], [16, -1], [26, -1]], "6": [[8, -1], [10, -1], [6, -1], [6, -2], [10, -3]], "7": [[7, -1], [7, -2], [7, -3], [19, -2], [19, -1], [19, -3], [22, -2], [26, -3], [25, -2], [26, -1], [25, -1], [26, -2], [22, -1], [18, -2], [25, -3], [18, -3], [18, -1]], "8": [[6, -1], [8, -1], [6, -3]], "9": [[10, -1], [9, -1], [8, -3], [10, -3], [20, -1]], "10": [[9, -1], [6, -1], [10, -1], [8, -3]], "11": [[11, -1], [11, -2], [11, -3], [26, -3], [26, -2], [26, -1], [18, -2], [23, -2], [17, -1], [20, -3], [17, -3], [16, -3], [16, -1], [20, -1], [16, -2], [20, -2], [18, -3], [23, -1], [23, -3], [18, -1], [21, -3], [17, -2], [14, -1], [21, -2], [22, -1], [21, -1], [22, -2], [22, -3], [19, -1], [19, -2], [19, -3], [0, -3], [0, -1], [14, -2], [14, -3], [0, -2], [6, -3]], "12": [[12, -1], [12, -2], [12, -3], [16, -2], [26, -2], [22, -2], [26, -3], [26, -1], [15, -1], [15, -3], [16, -1], [14, -1], [20, -2], [0, -1]], "13": [[13, -1], [13, -2], [13, -3], [26, -2], [26, -1], [26, -3], [21, -3], [25, -3], [20, -3], [25, -1], [8, -3], [17, -2], [25, -2], [20, -2], [17, -3], [21, -2], [21, -1], [17, -1], [23, -2], [23, -3], [23, -1], [11, -1], [22, -1], [22, -2], [22, -3], [11, -2], [11, -3]], "14": [[14, -1], [14, -2], [14, -3], [20, -2], [17, -1], [26, -2], [26, -3], [25, -1], [25, -2], [25, -3], [26, -1], [17, -3], [20, -1], [17, -2], [20, -3], [18, -2], [18, -3], [18, -1], [15, -1], [11, -3], [21, -2], [21, -1], [7, -2], [7, -3], [7, -1], [21, -3], [12, -3], [12, -2], [12, -1], [0, -3], [11, -1], [0, -2], [0, -1], [23, -1], [23, -2], [11, -2], [23, -3]], "15": [[15, -1], [15, -2], [15, -3], [26, -2], [26, -3], [26, -1], [20, -1], [20, -3], [14, -1], [18, -2], [21, -2], [18, -3], [25, -2], [18, -1], [10, -3]], "16": [[16, -2], [16, -1], [16, -3], [17, -1], [21, -1], [21, -2], [18, -1], [19, -1], [5, -2], [18, -2], [4, -3], [20, -3], [18, -3], [5, -1], [20, -2], [26, -1], [26, -3], [20, -1], [23, -2], [23, -1], [24, -1], [17, -3], [11, -1], [11, -3], [11, -2], [26, -2], [5, -3], [21, -3], [17, -2], [19, -2], [23, -3], [7, -2]], "17": [[17, -1], [17, -2], [17, -3], [21, -1], [26, -1], [16, -2], [26, -3], [18, -1], [26, -2], [18, -2], [16, -1], [21, -3], [16, -3], [2, -2], [2, -1], [2, -3], [19, -1], [1, -2], [19, -2], [1, -1], [0, -1], [0, -3], [0, -2], [21, -2], [1, -3], [19, -3], [9, -3], [10, -3], [9, -2], [9, -1], [10, -1], [10, -2], [25, -1], [18, -3], [25, -3], [24, -1], [25, -2], [11, -1], [11, -2], [11, -3], [24, -2], [7, -1], [7, -3], [7, -2], [23, -2], [20, -3]], "18": [[18, -2], [18, -1], [18, -3], [26, -2], [16, -1], [21, -2], [21, -3], [16, -2], [26, -1], [21, -1], [20, -3], [17, -3], [17, -1], [16, -3], [20, -2], [3, -1], [19, -3], [26, -3], [3, -2], [19, -2], [19, -1], [23, -2], [3, -3], [23, -1], [23, -3], [17, -2]], "19": [[19, -1], [19, -2], [19, -3], [7, -1], [4, -2], [16, -1], [4, -3], [18, -3], [16, -3], [18, -2], [4, -1], [18, -1], [16, -2], [21, -3]], "20": [[20, -1], [20, -2], [20, -3], [21, -1], [21, -3], [23, -2], [26, -3], [26, -2], [21, -2], [26, -1], [23, -3], [23, -1], [16, -2], [18, -2], [16, -3], [16, -1], [9, -1], [18, -1], [18, -3], [0, -1], [10, -3], [17, -3]], "21": [[21, -1], [21, -2], [21, -3], [17, -1], [20, -1], [17, -3], [20, -3], [18, -2], [16, -2], [17, -2], [16, -1], [20, -2], [16, -3], [19, -1], [18, -1], [26, -2], [6, -3], [8, -2], [26, -1], [6, -1], [18, -3], [26, -3], [25, -3], [19, -2], [24, -1], [22, -1], [14, -2], [14, -3], [14, -1], [2, -1]], "22": [[22, -2], [22, -3], [26, -2], [19, -3], [21, -1], [19, -1]], "23": [[23, -2], [23, -3], [26, -2], [20, -3], [26, -3], [26, -1], [20, -1], [20, -2], [18, -2], [16, -2], [23, -1]], "24": [[24, -2], [1, -1], [26, -1], [16, -1], [24, -1], [20, -3], [17, -1], [26, -3], [1, -2], [26, -2], [21, -3]], "25": [[25, -2], [25, -3], [26, -1], [21, -2], [21, -3], [24, -1], [17, -1]], "26": [[26, -2], [26, -1], [17, -3], [17, -2], [18, -2], [24, -1], [20, -3], [20, -2], [24, -3], [20, -1], [25, -1], [18, -3], [23, -2], [17, -1], [23, -3], [2, -1], [18, -1], [2, -3], [16, -2], [25, -3], [23, -1], [21, -3], [21, -1], [7, -3], [7, -1], [7, -2], [19, -2], [4, -1], [24, -2], [19, -3], [9, -1], [16, -1], [21, -2], [19, -1], [16, -3], [22, -2], [22, -1], [22, -3], [15, -3], [10, -1], [9, -2], [25, -2], [4, -3], [3, -3], [2, -2], [3, -2], [11, -1], [3, -1]], "27": [[27, -2], [27, -3]]}
    new_link_dict = {}
    for k, v in link_dict.items():
        new_link_dict[int(k)] = v
    link_dict = new_link_dict
    bayesian_debugger = BayesianDebugger(frame, link_dict, tau_max)

    security_guard = SecurityGuard(frame, bayesian_debugger.bayesian_fitter)
    print(security_guard.score_threshold)
    exit()

    background_generator:'BackgroundGenerator' = BackgroundGenerator(data_debugger.preprocessor, int_frame_id, tau_max)
    evaluator = Evaluator(data_debugger.preprocessor, background_generator, None, 0, alpha)

    miner_debugger = MinerDebugger(frame, background_generator, alpha)

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