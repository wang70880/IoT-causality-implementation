from src.event_processing import Hprocessor
from src.causal_evaluation import Evaluator
from src.background_generator import BackgroundGenerator
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb
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
import os

from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.tigramite.tigramite import data_processing as pp

class Drawer():

    def __init__(self) -> None:
        pass

    def draw_1d_distribution(self, val_list, xlabel='', ylabel='', title='', fname='default'):
        sns.displot(val_list, kde=False, color='red', bins=1000)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("temp/image/{}.pdf".format(fname))
        plt.close('all')
    
    def draw_2d_distribution(self, x_list, y_lists, label_list, x_label, y_label, title, fname):
        assert(len(label_list) == len(y_lists))
        for i, (label, y_list) in enumerate(list(zip(label_list, y_lists))):
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(x_list, y_list, label=label, c=col)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig('temp/image/{}.pdf'.format(fname))
        plt.close('all')

class DataDebugger():

    def __init__(self, dataset, partition_config=30, filter_threshold=30, training_ratio=1.0, tau_max = 4, alpha=0.001) -> None:
        self.dataset = dataset; self.partition_config = partition_config; self.filter_threshold = filter_threshold; self.training_ratio = training_ratio; self.tau_max = tau_max; self.alpha = alpha
        self.preprocessor = Hprocessor(dataset=dataset,tau_max=tau_max)
        self.preprocessor.initiate_data_preprocessing()
        self.preprocessor.data_loading(partition_config, training_ratio)
        self.background_generator:'BackgroundGenerator' = BackgroundGenerator(dataset, self.preprocessor, partition_config, tau_max)
        
        # Assigned after draw_golden_standard() function is called.
        self.golden_p_matrix = None; self.golden_val_matrix = None; self.golden_graph = None
    
    def validate_discretization(self):
        for dev, tup in self.preprocessor.discretization_dict.items():
            val_list = tup[0]; seg_point = tup[1]
            sns.displot(val_list, kde=False, color='red', bins=1000)
            plt.axvline(seg_point, 0, 1)
            plt.title('State distributions of device {}'.format(dev))
            plt.xlabel('State')
            plt.ylabel('Frequency')
            plt.savefig("temp/image/{}_{}_states.pdf".format(self.dataset, dev))

    def validate_ci_testing(self, tau_max=3, int_tau=3):

        # Load data and related attributes
        index_device_dict:'dict[DevAttribute]' = self.preprocessor.index_device_dict
        tested_frame:'DataFrame' = self.preprocessor.frame_dict[0]
        int_device_indices = [k for k in index_device_dict.keys() if index_device_dict[k].name.startswith('M')]
        n_vars = len(int_device_indices)
        # Initialize the conditional independence tester
        selected_links = {n: [(i, int(-int_tau)) for i in int_device_indices] for n in int_device_indices}
        pcmci = PCMCI(
            dataframe=tested_frame.training_dataframe,
            cond_ind_test=CMIsymb(),
            verbosity=-1)
        # Test the dependency relationship among motion sensors
        all_parents = defaultdict(dict); all_vals = defaultdict(dict); all_pvals = defaultdict(dict)

        for i in int_device_indices: # Calculate the statistic values and the p value
            parents = []
            val_dict = defaultdict(float)
            pval_dict = defaultdict(float)
            for j in int_device_indices:
                for tau in range(1, tau_max + 1):
                    lag = int(-tau)
                    Z = []
                    #Z = [(j, int(-inter_tau)) for inter_tau in range(1, tau-1)] # If conditioning on the previous variables
                    val, pval = pcmci.cond_ind_test.run_test(X=[(j, lag)],
                                                        Y=[(i, 0)],
                                                        Z=Z,
                                                        tau_max=tau_max)
                    parents.append((j, lag))
                    val_dict[(j, lag)] = val
                    pval_dict[(j, lag)] = pval
            all_parents[i] = parents
            all_vals[i] = val_dict
            all_pvals[i] = pval_dict
        
        def dict_to_matrix(all_vals, tau_max, int_attrs, default=1): # Inner helpful functions for output format transformation
            n_vars = len(int_attrs)
            matrix = np.ones((n_vars, n_vars, tau_max + 1))
            matrix *= default
            for j in all_vals.keys():
                for link in all_vals[j].keys():
                    k, tau = link
                    matrix[int_attrs.index(k), int_attrs.index(j), abs(tau)] = all_vals[j][link]
            return matrix
        
        val_matrix = dict_to_matrix(all_vals, tau_max, int_device_indices, default=0.)
        p_matrix = dict_to_matrix(all_pvals, tau_max, int_device_indices, default=1.)
        selected_links_matrix = np.empty((n_vars, n_vars, tau_max + 1)); selected_links_matrix.fill(1.1)
        for j in selected_links.keys():
            for link in selected_links[j]:
                k, tau = link
                selected_links_matrix[int_device_indices.index(k), int_device_indices.index(j), abs(tau)] = 0.
        final_graph = p_matrix + selected_links_matrix # Adjust graph according to the selected links: For non-selected links, penalize its p-value by adding 1.1
        final_graph = final_graph <= self.alpha # Adjust graph according to hypothesis testing
        graph = pcmci.convert_to_string_graph(final_graph)
        # Sorting all parents according to the CMI value
        for dev_index in int_device_indices:
            print("Sorted parents for device {}:".format(index_device_dict[dev_index].name))
            parents = [x for x in int_device_indices if final_graph[int_device_indices.index(x), int_device_indices.index(dev_index), int_tau]]
            parent_vals = [(x, val_matrix[int_device_indices.index(x) , int_device_indices.index(dev_index), int_tau])
                                 for x in parents]
            parent_vals.sort(key=lambda tup: tup[1], reverse=True)
            print(" -> ".join( [index_device_dict[x[0]].name for x in parent_vals] ))
        # Plotting
        var_names = [index_device_dict[k].name for k in int_device_indices]
        tp.plot_time_series_graph(
            figsize=(6, 4),
            val_matrix=val_matrix,
            graph=graph,
            var_names=var_names,
            link_colorbar_label='MCI'
        )
        plt.savefig("temp/image/{}cmi_test_tau{}.pdf".format(self.dataset, int_tau))
    
    def derive_golden_standard(self, int_frame_id, int_type):
        # Auxillary variables
        tested_frame:'DataFrame' = self.preprocessor.frame_dict[int_frame_id]
        n_vars = tested_frame.n_vars
        matrix_shape = (n_vars, n_vars, self.tau_max + 1)
        interaction_matrix:'np.ndarray' = np.zeros(matrix_shape)
        evaluator = Evaluator(self.preprocessor, self.background_generator, None, self.tau_max, self.filter_threshold)
        interaction_dict:'dict[np.ndarray]' = evaluator.golden_standard_dict[int_type]
        interaction_matrix = interaction_dict[int_frame_id]
        str = "[Picturing Golden Standard] Number of edges for each tau:\n"
        for lag in range(1, self.tau_max + 1):
            total = np.sum(interaction_matrix[:,:,lag])
            str += 'tau = {}, count = {}\n'.format(lag, total)
        #print(str)
        return interaction_matrix

    def analyze_golden_standard(self, int_frame_id=0, int_type='user'):

        # Auxillary variables
        tested_frame:'DataFrame' = self.preprocessor.frame_dict[int_frame_id]
        var_names = tested_frame.var_names; n_vars = tested_frame.n_vars
        matrix_shape = (n_vars, n_vars, self.tau_max + 1)
        index_device_dict:'dict[DevAttribute]' = self.preprocessor.index_device_dict

        # 1. Derive the golden standard matrix (with expert knowledge).
        interaction_matrix:'np.ndarray' = self.derive_golden_standard(int_frame_id, int_type)

        # 2. Calculate the p-value and CMI value for each edge.
        val_matrix:'np.ndarray' = np.zeros(matrix_shape); p_matrix:'np.ndarray' = np.ones(matrix_shape)
        p_matrix *= 10 # Initially we set a large number
        pcmci = PCMCI(dataframe=tested_frame.training_dataframe, cond_ind_test=CMIsymb(), verbosity=-1)
        for index, x in np.ndenumerate(interaction_matrix):
            if x == 1:
                i, j, lag = index
                val, pval = pcmci.cond_ind_test.run_test(X=[(i, int(-lag))],
                            Y=[(j, 0)],
                            Z=[],
                            tau_max=self.tau_max)
                val_matrix[i, j, lag] = val; p_matrix[i, j, lag] = pval
                #print("For index {}, val = {}, pval = {}".format(index, val ,pval))
        p_matrix[p_matrix >= 1] = 1; final_graph = p_matrix < 1
        graph = pcmci.convert_to_string_graph(final_graph)
        tp.plot_time_series_graph( # Plot the graph
            figsize=(6, 4),
            val_matrix=val_matrix,
            graph=graph,
            var_names=var_names,
            link_colorbar_label='MCI'
        )
        print("Total # of golden edges: {}\n".format(np.sum(interaction_matrix)))
        plt.savefig("temp/image/golden_standard_{}.pdf".format(int_type))

        # 3. Verify the causal assumptions
        ## 3.1 Verify causal minimality assumptions
        assert(interaction_matrix.shape == val_matrix.shape == p_matrix.shape)
        n_violation = 0
        for index, x in np.ndenumerate(interaction_matrix):
            i, j, lag = index
            if x == 1 and p_matrix[index] > self.alpha:
                n_violation += 1
                print("The edge {}(-{}) -> {} violates minimality assumption! p-value = {}, CMI = {}"\
                                    .format(index_device_dict[i].name, lag, index_device_dict[j].name, p_matrix[index], val_matrix[index]))
        print("Total # of violations: {}\n".format(n_violation))
        return interaction_matrix, p_matrix, val_matrix

class MinerDebugger():
    
    def __init__(self, alpha = 0.001, data_debugger = None) -> None:
        assert(data_debugger is not None)
        self.alpha = alpha 
        self.data_debugger:'DataDebugger' = data_debugger

    def initiate_discovery_algorithm(self, int_frame_id):
        # Auxillary variables
        name_device_dict = self.data_debugger.preprocessor.name_device_dict; tested_frame:'DataFrame' = self.data_debugger.preprocessor.frame_dict[int_frame_id]
        tau_max = self.data_debugger.tau_max; n_vars = tested_frame.n_vars
        matrix_shape = (n_vars, n_vars, tau_max + 1)
        # Return variables
        interaction_matrix:'np.ndarray' = np.zeros(matrix_shape); p_matrix:'np.ndarray' = np.ones(matrix_shape); val_matrix:'np.ndarray' = np.zeros(matrix_shape)

        # 1. Call pc algorithm to get the answer
        # JC TODO: Implement the discovery logic here.
        #all_parents_with_name = {'D002': [('D002', -1), ('D002', -2), ('M001', -1), ('M001', -2), ('D002', -3), ('M001', -3)],\
        #                        'M001': [('M001', -1), ('M001', -2), ('D002', -2), ('D002', -1), ('M005', -2), ('D002', -3), ('M004', -1), ('M001', -3), ('M003', -1), ('M005', -1), ('M002', -1), ('M006', -2), ('M006', -1), ('M005', -3), ('M006', -3)],\
        #                        'M002': [('M002', -2), ('M002', -1), ('M001', -1), ('M004', -1), ('M005', -2), ('M005', -1), ('M003', -1), ('M006', -1)],\
        #                        'M003': [('M003', -2), ('M003', -1), ('M005', -1), ('M004', -1), ('M004', -2), ('M001', -1), ('M002', -1), ('M006', -1), ('M002', -2)],\
        #                        'M004': [('M004', -2), ('M005', -1), ('M002', -1), ('M001', -1), ('M004', -1), ('M006', -1), ('M005', -3)],\
        #                        'M005': [('M005', -2), ('M005', -3), ('M005', -1), ('M004', -1), ('M003', -2), ('M001', -1), ('M003', -1), ('M002', -2), ('M002', -1), ('M006', -1), ('M001', -3), ('M002', -3), ('D002', -1), ('D002', -3), ('M006', -3), ('M004', -2), ('M001', -2), ('D002', -2)],\
        #                        'M006': [('M006', -2), ('M005', -1), ('M001', -1), ('M011', -2), ('M003', -1), ('M004', -1), ('M002', -3), ('M001', -2), ('M002', -1), ('M002', -2), ('M001', -3)],\
        #                        'M011': [('M011', -2), ('M006', -2), ('M011', -1)]} # With alpha = 0.001 and bk = 0
        all_parents_with_name = {'D002': [('D002', -1), ('D002', -2), ('M001', -1), ('M001', -2), ('D002', -3), ('M001', -3)],\
                                'M001': [('M001', -1), ('M001', -2), ('D002', -2), ('D002', -1), ('M005', -2), ('D002', -3), ('M004', -1), ('M001', -3), ('M003', -1), ('M005', -1), ('M002', -1), ('M006', -3), ('M005', -3)],\
                                'M002': [('M002', -2), ('M002', -1), ('M001', -1), ('M004', -1), ('M005', -2), ('M005', -1), ('M003', -1), ('M006', -1)],\
                                'M003': [('M003', -2), ('M003', -1), ('M005', -1), ('M004', -1), ('M004', -2), ('M001', -1), ('M002', -1), ('M006', -1), ('M002', -2)],\
                                'M004': [('M004', -2), ('M005', -1), ('M002', -1), ('M001', -1), ('M004', -1), ('M006', -1), ('M005', -3)],\
                                'M005': [('M005', -2), ('M005', -3), ('M005', -1), ('M004', -1), ('M003', -2), ('M001', -1), ('M003', -1), ('M002', -2), ('M002', -1), ('M006', -1), ('M001', -3), ('M002', -3), ('D002', -1), ('D002', -3), ('M006', -3), ('M004', -2), ('M001', -2), ('D002', -2)],\
                                'M006': [('M006', -2), ('M005', -1), ('M011', -2), ('M003', -1), ('M004', -1), ('M001', -3), ('M002', -1)],\
                                'M011': [('M011', -2), ('M006', -2), ('M011', -1)]} # With alpha = 0.001 and bk = 1
        
        # 2. Construct the interaction matrix given the pc-returned dict
        for outcome in all_parents_with_name:
            for (cause, lag) in all_parents_with_name[outcome]:
                interaction_matrix[name_device_dict[cause].index, name_device_dict[outcome].index, abs(lag)] = 1

        return interaction_matrix
    
    def analyze_discovery_result(self, int_frame_id=0):

        # 1. Fetch the golden interaction matrix
        golden_interaction_matrix:'np.ndarray' = self.data_debugger.derive_golden_standard(int_frame_id, 'user')

        # 2. Initiate PC discovery algorithm and get the discovered graph
        discovered_interaction_matrix:'np.ndarray' = self.initiate_discovery_algorithm(int_frame_id)

        total_count = np.sum(golden_interaction_matrix)
        tp = np.sum(discovered_interaction_matrix[golden_interaction_matrix == 1])
        fn = total_count - tp
        fp = np.sum(discovered_interaction_matrix[golden_interaction_matrix == 0])

        precision = 0. if (tp + fp) == 0 else 1.0 * tp / (tp + fp)
        recall = 0. if (tp + fn) == 0 else 1.0 * tp / (tp + fn)

        print("For alpha = {}, the discovered result statistics:\n\
                * tp = {}, fp = {}, fn = {}\n\
                * precision = {}, recall = {}\n".format(self.alpha, tp, fp, fn, precision, recall))

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
        drawer.draw_1d_distribution(activation_ratios, fname='activation-cpt-threshold{}'.format(self.data_debugger.filter_threshold))
        drawer.draw_1d_distribution(deactivation_ratios, fname='deactivation-cpt-threshold{}'.format(self.data_debugger.filter_threshold))

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
        
        self.drawer.draw_1d_distribution(training_anomaly_scores, xlabel='Score', ylabel='Occurrence', title='Training event detection using golden standard model', fname='prediction-probability-distribution-threshold{}'.format(self.data_debugger.filter_threshold))
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
                                             bayesian_fitter = bayesian_fitter, tau_max=tau_max, filter_threshold=self.data_debugger.filter_threshold)
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

    dataset = 'hh130'; partition_config = 30; filter_threshold = .5 * partition_config; training_ratio = 0.8; tau_max = 3
    alpha = 0.001; int_frame_id = 4; analyze_golden_standard=False
    data_debugger = DataDebugger(dataset, partition_config, filter_threshold, training_ratio, tau_max, alpha)
    bayesian_debugger = BayesianDebugger(data_debugger, verbosity=0, analyze_golden_standard=analyze_golden_standard)

    n_anomalies = 50; maximum_length = 1; anomaly_case = 1
    sig_levels = list(np.arange(0.1, 1., 0.1)); sig_levels = [0.95]
    precisions = []; recalls = []; f1_scores = []
    for sig_level in sig_levels:
        guard_debugger = GuardDebugger(data_debugger, bayesian_debugger)
        precision, recall, f1_score = guard_debugger.score_anomaly_detection(int_frame_id, sig_level, n_anomalies, maximum_length, anomaly_case)
        precisions.append(precision); recalls.append(recall); f1_scores.append(f1_score)

    drawer = Drawer()
    drawer.draw_2d_distribution(sig_levels, [precisions, recalls, f1_scores], ['precision', 'recall', 'f1-score'], 'sig-level', 'value', 'No', 'sig-level-analysis')