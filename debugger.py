from src.event_processing import Hprocessor, Cprocessor
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import ChiSquare
from src.tigramite.tigramite import plotting as tp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame
from src.benchmark.association_miner import AssociationMiner
from src.benchmark.ocsvm import OCSVMer

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

from src.event_processing import GeneralProcessor
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
            self.preprocessor = Hprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio, verbosity=1)
        elif self.dataset.startswith('contextact'):
            self.preprocessor = Cprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio, verbosity=1)
        self.preprocessor.initiate_data_preprocessing()
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

    def __init__(self, bayesian_fitter=None, sig_level=None) -> None:
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        self.security_guard:'SecurityGuard' = SecurityGuard(
            self.bayesian_fitter.frame, self.bayesian_fitter, sig_level
            )
    
    def check_suitable_injection_cases(self):
        var_names = self.bayesian_fitter.var_names
        def all_bits(n):
            if n: yield from (bits + [bit] for bits in all_bits(n-1) for bit in (0, 1))
            else: yield []
        # 1. Get all parent situations under which a device event will be regarded as an anomaly.
        anomaly_act_cases = defaultdict(list)
        anomaly_deact_cases = defaultdict(list)
        for var_name in var_names:
            parents = self.bayesian_fitter.model.get_parents(var_name); n_parents = len(parents)
            for potential_states in all_bits(len(parents)):
                states_str = "".join([str(x) for x in potential_states])
                parent_states = list(zip(parents, potential_states))
                anomalous_act_event = AttrEvent('', '', var_name, '', 1); anomalous_deact_event = AttrEvent('', '', var_name, '', 0)
                if 1-self.bayesian_fitter.estimate_cond_probability(anomalous_act_event, parent_states) > self.security_guard.score_threshold:
                    anomaly_act_cases[var_name].append(states_str)
                if 1-self.bayesian_fitter.estimate_cond_probability(anomalous_deact_event, parent_states) > self.security_guard.score_threshold:
                    anomaly_deact_cases[var_name].append(states_str)
            #print("# anomaly act/deact cases for device {}: {}, {}"\
            #            .format(var_name, len(anomaly_act_cases[var_name]), len(anomaly_deact_cases[var_name])))
        return anomaly_act_cases, anomaly_deact_cases

    def check_suitable_injection_positions(self):
        # Auxillary variables
        frame = self.bayesian_fitter.frame
        var_names = self.bayesian_fitter.var_names
        tau_max = self.bayesian_fitter.tau_max
        anomaly_act_cases, anomaly_deact_cases = self.check_suitable_injection_cases()

        # Retuan variables
        anomalous_dev_counts = defaultdict(int)
        potential_act_anomaly_positions = defaultdict(list)
        potential_deact_anomaly_positions = defaultdict(list)

        # Initialize the state machine
        latest_event_states = [frame.training_events_states[-tau_max+i] for i in range(0, tau_max)]
        machine_initial_states = [event_state[1] for event_state in latest_event_states]
        self.security_guard.initialize_phantom_machine(machine_initial_states)

        testing_event_states = frame.testing_events_states
        for evt_id, tup in enumerate(testing_event_states):
            event, states = tup
            # Traverse each potential anomalous device (except for the current benign device)
            for var_name in [x for x in var_names if x != event.dev]:
                parents = self.bayesian_fitter.model.get_parents(event.dev)
                int_index = self.bayesian_fitter.expanded_var_names.index(var_name)
                parent_indices = [self.bayesian_fitter.expanded_var_names.index(parent)-self.bayesian_fitter.n_vars for parent in parents]
                parent_states:'list[tuple(str, int)]' = list(zip(
                    parents,
                    self.security_guard.phantom_state_machine.get_indices_states(parent_indices)
                ))
                states_str = "".join([str(x[1]) for x in parent_states])
                if states_str in anomaly_act_cases[var_name] and self.security_guard.phantom_state_machine.get_indices_states([int_index])[0]==0:
                    potential_act_anomaly_positions[evt_id].append(var_name)
                    anomalous_dev_counts[var_name] += 1
                if states_str in anomaly_deact_cases[var_name] and self.security_guard.phantom_state_machine.get_indices_states([int_index])[0]==1:
                    potential_deact_anomaly_positions[evt_id].append(var_name)
                    anomalous_dev_counts[var_name] += 1
            self.security_guard.phantom_state_machine.set_latest_states(states)
        
        n_positions = 0
        for evt_id in range(len(frame.testing_events_states)):
            if len(potential_act_anomaly_positions[evt_id]) + len(potential_deact_anomaly_positions[evt_id]) > 0:
                n_positions += 1
        
        print("Total #, % potential positions for anomaly injection: {}, {}".format(n_positions, n_positions*1./len(frame.testing_events_states)))
        pprint(anomalous_dev_counts)

if __name__ == '__main__':

    dataset = 'hh130'; partition_days = 30; training_ratio = 0.8; tau_max = 2
    alpha = 0.001; int_frame_id = 0
    n_max_edges = 10; sig_level = 0.99

    data_debugger = DataDebugger(dataset, partition_days, training_ratio)
    event_preprocessor:'GeneralProcessor' = data_debugger.preprocessor
    frame:'DataFrame' = data_debugger.preprocessor.frame_dict[int_frame_id]

    ocsvmer = OCSVMer(frame, tau_max)
    alarm_position_events = ocsvmer.contextual_anomaly_detection()
    print("# OCSVM alarms: {}".format(len(alarm_position_events)))
    exit()

    association_miner = AssociationMiner(event_preprocessor, frame, tau_max, alpha)

    causal_link_dict = {"0": [[0, -1], [0, -2], [9, -1], [9, -2], [14, -1], [1, -1], [12, -1]], "1": [[1, -1], [1, -2], [15, -1], [9, -1], [15, -2], [9, -2]], "2": [[2, -2], [9, -1], [17, -1], [12, -1], [17, -2], [9, -2]], "3": [[3, -2], [10, -1], [10, -2], [17, -1], [17, -2]], "4": [[11, -1], [4, -1], [17, -1]], "5": [[5, -1], [5, -2], [11, -1], [11, -2], [13, -2], [13, -1], [16, -2], [17, -2], [17, -1], [16, -1], [9, -1]], "6": [[6, -1], [6, -2], [8, -1], [8, -2], [10, -1], [10, -2], [17, -2], [17, -1]], "7": [[7, -1], [7, -2], [14, -1], [14, -2], [12, -2], [12, -1], [16, -1], [16, -2], [17, -1], [17, -2], [10, -2], [13, -2], [10, -1], [13, -1], [9, -2], [9, -1], [8, -1], [6, -1]], "8": [[8, -1], [8, -2], [17, -2], [17, -1], [6, -1], [14, -1], [14, -2], [16, -1], [16, -2], [9, -1], [9, -2], [7, -1]], "9": [[9, -1], [9, -2], [10, -1], [10, -2], [12, -1], [2, -1], [11, -1], [11, -2], [1, -1], [2, -2], [15, -2], [17, -1], [0, -1], [16, -2], [15, -1], [17, -2], [16, -1], [1, -2], [12, -2], [0, -2], [7, -2], [7, -1], [5, -2], [14, -2], [5, -1], [3, -2], [3, -1], [8, -1], [8, -2], [14, -1]], "10": [[10, -2], [10, -1], [12, -1], [17, -1], [17, -2], [12, -2], [9, -1], [14, -1], [14, -2], [9, -2], [3, -1], [11, -1], [11, -2], [3, -2], [13, -1], [13, -2]], "11": [[11, -1], [11, -2], [4, -2], [5, -1], [4, -1], [10, -1], [10, -2], [13, -1], [12, -2]], "12": [[12, -2], [12, -1], [10, -1], [10, -2], [9, -1], [16, -2], [17, -2], [16, -1], [17, -1], [15, -1], [2, -1], [14, -2], [15, -2], [14, -1], [9, -2], [5, -2], [13, -1], [3, -2]], "13": [[13, -2], [11, -1], [17, -1], [17, -2]], "14": [[14, -2], [17, -1], [17, -2], [10, -1], [10, -2], [15, -1], [14, -1], [0, -1], [11, -1], [11, -2]], "15": [[15, -2], [17, -1], [1, -1], [17, -2], [15, -1], [1, -2], [9, -2], [12, -1], [14, -1], [11, -1], [9, -1], [16, -1]], "16": [[16, -2], [17, -1], [12, -2], [12, -1], [9, -2], [15, -1], [9, -1]], "17": [[17, -2], [17, -1], [10, -1], [10, -2], [15, -1], [14, -1], [14, -2], [16, -1], [2, -1], [15, -2], [13, -1], [11, -1], [12, -1], [11, -2], [12, -2], [13, -2], [18, -1], [8, -1], [8, -2], [2, -2], [4, -1], [3, -2], [3, -1], [5, -2], [9, -2]], "18": [[18, -2], [17, -1]]}
    causal_link_dict = {int(k): v for k, v in causal_link_dict.items()}

    evaluator = Evaluator(event_preprocessor, frame, tau_max, alpha)
    for n_max_edges in range(5, 10):
        print("n_max_edges={}".format(n_max_edges))
        evaluator.check_selection_bias(association_miner.mining_edges, n_max_edges, sig_level)
        evaluator.check_selection_bias(causal_link_dict, n_max_edges, sig_level)
    exit()

    bayesian_fitter = BayesianFitter(frame, tau_max, link_dict, n_max_edges=n_max_edges)

    guard_debugger = GuardDebugger(bayesian_fitter, sig_level)
    #guard_debugger.check_suitable_injection_cases()
    guard_debugger.check_suitable_injection_positions()
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