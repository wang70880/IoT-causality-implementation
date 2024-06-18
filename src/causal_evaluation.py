import collections
import itertools
import json
import numpy as np
import random
import pandas as pd
import statistics
from src.tigramite.tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from numpy import ndarray
from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.benchmark.association_miner import AssociationMiner
from collections import defaultdict
from pprint import pprint
from copy import deepcopy

from src.tigramite.tigramite import pcmci
from src.tigramite.tigramite.independence_tests.chi2 import ChiSquare

class Evaluator():

    def __init__(self, event_preprocessor, frame, tau_max, pc_alpha, golden_edges=None, golden_array=None, nor_golden_array=None, causal_edges=None):
        self.event_preprocessor:'GeneralProcessor' = event_preprocessor
        self.frame:'DataFrame' = frame
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.causal_edges = causal_edges
        if golden_edges is None:
            self.golden_edges, self.golden_rules, self.rule_chains, self.golden_array, self.nor_golden_array = self._construct_golden_standard()
        else:
            self.golden_edges = golden_edges
            self.golden_array = golden_array
            self.nor_golden_array = nor_golden_array
    
    """Helper functions."""

    def normalize_time_series_array(self, arr:'np.ndarray', threshold=0):
        n_rows = arr.shape[0]; n_cols = arr.shape[1]
        ret_arr:'np.ndarray' = np.zeros((n_rows, n_cols), dtype=np.int8)
        for i in range(n_rows):
            for j in range(n_cols):
                ret_arr[i, j] = 1 if np.sum(arr[i,j,:])>threshold else 0
        return ret_arr
    
    def calculate_accuracy(self, discovery_results, ground_truths):
        tp = len([x for x in discovery_results if x in ground_truths])
        fp = len([x for x in discovery_results if x not in ground_truths])
        fn = len([x for x in ground_truths if x not in discovery_results])
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0
        f1 = 2.0*precision*recall/(precision+recall) if (precision+recall) != 0 else 0
        return tp, fp, fn, precision, recall, f1
    
    def calculate_matrix_accuracy(self, discovery_array, ground_array):
        tp = 0; fp = 0; tn = 0; fn = 0; precision = 0.; recall = 0.; f1 = 0.
        assert(discovery_array.shape == ground_array.shape)
        for index, x in np.ndenumerate(discovery_array):
            if discovery_array[index] == ground_array[index] == 1:
                tp += 1
            elif discovery_array[index] == ground_array[index] == 0:
                tn += 1
            elif discovery_array[index] == 1 and ground_array[index] == 0:
                fp += 1
            elif discovery_array[index] == 0 and ground_array[index] == 1:
                fn += 1
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0
        f1 = 2.0*precision*recall/(precision+recall) if (precision+recall) != 0 else 0
        return tp, fp, fn, precision, recall, f1

    """Function classes for ground truth construction."""

    def _construct_golden_standard(self):
        """
        For all known interactions between two devices (in the ground truth file), we further determine the golden time lag
        """
        # Read the ground user interactions and ground physical interactions from the file
        ground_truth_array = None
        try:
            golden_df = pd.read_csv(self.event_preprocessor.ground_truth, encoding="utf-8", delim_whitespace=True, header=0, dtype=int)
            ground_truth_array = golden_df.values
            var_names = list(golden_df.columns)
        except:
            raise FileNotFoundError("Cannot construct the ground truth! (Maybe the ground truth file is missing?)")
        
        # Read the ground automation interactions from the file
        ground_automation_dict = None
        chained_automation_rules = []
        try:
            f = open(self.event_preprocessor.automation_path, 'r')
            ground_automation_dict = json.loads(f.read())
            ground_automation_dict = {(k.split(',')[0], int(k.split(',')[1])):(v.split(',')[0], int(v.split(',')[1])) for k, v in ground_automation_dict.items()}
            f.close()
            for trigger, action in ground_automation_dict.items():
                chain = []
                cur_trigger = trigger; cur_action = action
                chain.append(cur_trigger); chain.append(cur_action)
                while True:
                    if cur_action in ground_automation_dict.keys():
                        cur_trigger = cur_action; cur_action = ground_automation_dict[cur_trigger]
                        chain.append(cur_action)
                        if len(chain)>=5:
                            break
                    else:
                        break
                chained_automation_rules.append(chain)
            #print("# automation rules: {}".format(len(ground_automation_dict)))
            #pprint(chained_automation_rules)
        except:
            raise FileNotFoundError("Cannot construct the ground automations! (Maybe the ground truth file is missing?)")
        
        # Generate the interaction set, given the ground truth array
        # Currently we don't know the time lag yet, so we add edges with all lags
        selected_links = defaultdict(list)
        for (cause, outcome), x in np.ndenumerate(ground_truth_array):
            if x == 0:
                continue
            if self.causal_edges is None: # We use the discovered causal edge lag to calibrate the time lag of the ground truth
                for lag in range(1, self.tau_max+1):
                    selected_links[outcome].append((cause, -lag))
            else:
                candidate_causes = [] if outcome not in self.causal_edges.keys() else [tup for tup in self.causal_edges[outcome] if tup[0]==cause]
                if len(candidate_causes) > 0:
                    selected_links[outcome] += candidate_causes
                else:
                    for lag in range(1, self.tau_max+1):
                        selected_links[outcome].append((cause, -lag))
        # Insert the automation logic to the ground truth
        for k, v in ground_automation_dict.items():
            trigger_dev = k[0]; action_dev = v[0]
            trigger_index = var_names.index(trigger_dev); action_index = var_names.index(action_dev)
            for lag in range(1, self.tau_max+1):
                if (trigger_index, -lag) not in selected_links[action_index]:
                    selected_links[action_index].append((trigger_index, -lag))
                else:
                    pass
        # For all candidate time lags, we filter them according to the statistical test, and get the ranked edge sets
        asso_miner = AssociationMiner(self.event_preprocessor, self.frame, self.tau_max, 0.6)
        golden_edges, golden_array, nor_golden_array = asso_miner.interaction_mining(selected_links)
        return golden_edges, ground_automation_dict, chained_automation_rules, golden_array, nor_golden_array

    def categorize_interaction(self, p_dev:'DevAttribute', c_dev:'DevAttribute'):
        p_attr = p_dev.attr; c_attr = c_dev.attr
        attr_list = [p_attr, c_attr]
        int_type = 'unknown'

        actuators = ['Dimmer', 'Switch', 'Water-Meter', 'Power-Sensor']
        movement_detectors = ['Infrared-Movement-Sensor', 'Contact-Sensor']
        brightness_channels = ['Brightness-Sensor']
        for k, v in self.golden_rules.items():
            if p_dev.name == k[0] and c_dev.name == v[0]:
                return 'automation'
        if p_dev.name == c_dev.name:
            int_type = 'autocorrelation'
        elif p_attr in actuators+brightness_channels and c_attr in actuators:
            int_type = 'uau'
        elif p_attr in  actuators+brightness_channels and c_attr in movement_detectors:
            int_type = 'mau'
        elif all([x in movement_detectors for x in attr_list]):
            int_type = 'mam'
        elif p_attr in movement_detectors and c_attr in actuators:
            int_type = 'uam'
        elif c_attr in brightness_channels:
            int_type = 'brightness'
        else:
            print("Unknown categorization for pair {}".format(attr_list))
            int_type = 'unknown'
            
        return int_type

    def evaluate_discovery_accuracy(self, evaluated_array:'np.ndarray', golden_array:'np.ndarray',\
                            filtered_edge_infos:'dict'=None, identified_edge_infos:'dict'=None, model='unknown', background_generator=None, verbosity=0):
        assert(len(evaluated_array.shape)==len(golden_array.shape)==2) 
        # Auxillary variables
        frame:'DataFrame' = self.frame
        index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        frequency_array = background_generator.frequency_array

        # 1. Calculate tp, fp, fn
        tp, fp, fn, precision, recall, f1_score = self.calculate_matrix_accuracy(evaluated_array, golden_array)
        assert(tp+fn == np.sum(golden_array))

        # 2. Categorize each discovered ground truth edges
        tp_info_dict = defaultdict(list)
        for index, x in np.ndenumerate(evaluated_array):
            if evaluated_array[index] == golden_array[index] == 1:
                p_dev = index_device_dict[index[0]]; c_dev = index_device_dict[index[1]]
                tp_info_dict[self.categorize_interaction(p_dev, c_dev)].append((p_dev.name, c_dev.name))

        # 3. Save debugging information for each edge (if applied)
        tp_dict = None; tn_dict = None; fp_dict = None; fn_dict = None
        if filtered_edge_infos and identified_edge_infos:
            tp_dict = defaultdict(dict); tn_dict = defaultdict(dict); fp_dict = defaultdict(dict); fn_dict = defaultdict(dict)
            for index, x in np.ndenumerate(evaluated_array):
                pair_info = {
                    'adev-pair': '{}->{}'.format(index_device_dict[index[0]].name, index_device_dict[index[1]].name),
                    'attr-pair': (index_device_dict[index[0]].attr, index_device_dict[index[1]].attr),
                    'spatial': (index_device_dict[index[0]].location, index_device_dict[index[1]].location),
                    'frequency': max(list(frequency_array[index[0], index[1], :])),
                    'int-type': self.categorize_interaction(index_device_dict[index[0]], index_device_dict[index[1]])
                }
                if evaluated_array[index] == 1 and golden_array[index] == 1:
                    edge_infos = [edge_info for edge_info in identified_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    max_ate = 0.0
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        if abs(ate) > max_ate:
                            pair_info['pval'] = round(edge_info[2], 3)
                            pair_info['cpt'] = cpt
                            pair_info['cate'] = round(ate, 3)
                    tp_dict[pair_info['adev-pair']] = pair_info
                elif evaluated_array[index] == 1 and golden_array[index] == 0:
                    edge_infos = [edge_info for edge_info in identified_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    max_ate = 0.0
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        if abs(ate) > max_ate:
                            pair_info['pval'] = round(edge_info[2], 3)
                            pair_info['cpt'] = cpt
                            pair_info['cate'] = round(ate, 3)
                    fp_dict[pair_info['adev-pair']] = pair_info
                elif evaluated_array[index] == 0 and golden_array[index] == 1:
                    edge_infos = [edge_info for edge_info in filtered_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    filter_infos = []
                    max_ate = 0.; selected_cpt = {}
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        if abs(ate) > max_ate:
                            max_ate = ate
                            selected_cpt = cpt
                        filter_infos.append((edge_info[0][1][1], [(index_device_dict[cond[0]].name, cond[1]) for cond in edge_info[1]], round(edge_info[3],3), round(ate,3))) # lag, conds, pval, ate
                    sorted(filter_infos, key=lambda x: x[-1], reverse=True)
                    pair_info['filter-info'] = filter_infos
                    pair_info['cpt'] = selected_cpt
                    pair_info['cate'] = round(max_ate, 3)
                    fn_dict[pair_info['adev-pair']] = pair_info
                else:
                    edge_infos = [edge_info for edge_info in filtered_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    filter_infos = []
                    max_ate = 0.; selected_cpt = {}
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        if abs(ate) > max_ate:
                            max_ate = ate
                            selected_cpt = cpt
                        filter_infos.append((edge_info[0][1][1], [(index_device_dict[cond[0]].name, cond[1]) for cond in edge_info[1]], round(edge_info[3],3), round(ate,3))) # lag, conds, pval, ate
                    sorted(filter_infos, key=lambda x: x[-1], reverse=True)
                    pair_info['filter-info'] = filter_infos
                    pair_info['cpt'] = selected_cpt
                    pair_info['cate'] = round(max_ate, 3)
                    tn_dict[pair_info['adev-pair']] = pair_info
            tp_dict = dict(sorted(tp_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))
            fp_dict = dict(sorted(fp_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))
            tn_dict = dict(sorted(tn_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))
            fn_dict = dict(sorted(fn_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))

        # 1. Plot the golden standard graph and the discovered graph. Note that the plot functionality requires PCMCI objects
        #drawer = Drawer(self.background_generator.dataset)
        #pcmci = PCMCI(dataframe=frame.training_dataframe, cond_ind_test=ChiSquare(), verbosity=-1)
        #drawer.plot_interaction_graph(pcmci, discovery_results==1, 'mined-interaction-bklevel{}-alpha{}'\
        #                    .format(self.bk_level, int(1.0/self.pc_alpha)), link_label_fontsize=10)
        #drawer.plot_interaction_graph(pcmci, golden_standard_array==1, 'golden-interaction')
        # 3. Calculate the tau-free-precision and tau-free-recall for discovered results
        print("Interaction Mining evalutation for model {}".format(model))
        print("     [FP, FN, Precision, Recall, F1] = {}, {}, {}, {}, {}".format(fp, fn, precision, recall, f1_score))
        if verbosity:
            for k, v in tp_info_dict.items():
                print("     {} discovered {} interactions: {}".format(len(v), k, v))
            if filtered_edge_infos and identified_edge_infos:
                print('TP Infos ({})'.format(len(tp_dict.keys())))
                for tp_info in tp_dict.values():
                    pprint(tp_info)
                print('FP Infos ({})'.format(len(fp_dict.keys())))
                for fp_info in fp_dict.values():
                    pprint(fp_info)
                print('TN Infos ({})'.format(len(tn_dict.keys())))
                for tn_info in tn_dict.values():
                    pprint(tn_info)
                print('FN Infos ({})'.format(len(fn_dict.keys())))
                for fn_info in fn_dict.values():
                    pprint(fn_info)
        return tp, fp, fn, precision, recall, f1_score

    def compare_with_arm(self, discovery_results:'np.ndarray', arm_results:'np.ndarray'):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        var_names = frame.var_names; n_vars = frame.n_vars; index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        golden_standard_array:'np.ndarray' = self.ground_truth_array
        assert(discovery_results.shape == (n_vars, n_vars, self.tau_max + 1))
        assert(arm_results.shape == (n_vars, n_vars))

        # 1. Reduce discovery_results from (n_vars, n_vars, tau_max) to (n_vars, n_vars)
        tau_free_discovery_array = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        assert(tau_free_discovery_array.shape == golden_standard_array.shape == (n_vars, n_vars))

        # 2. Calculate the discovery accuracy for ARM and our algorithm, respectively
        causal_tp, causal_fp, causal_fn, causal_precision, causal_recall, causal_f1 = self.precision_recall_calculation(golden_standard_array, tau_free_discovery_array)
        arm_tp, arm_fp, arm_fn, arm_precision, arm_recall, arm_f1 = self.precision_recall_calculation(golden_standard_array, arm_results)
        print("Causal discovery tp, fp, fn, precision, recall, f1 = {}, {}, {}, {}, {}, {}".format(causal_tp, causal_fp, causal_fn, causal_precision, causal_recall, causal_f1))
        print("ARM tp, fp, fn, precision, recall, f1 = {}, {}, {}, {}, {}, {}".format(arm_tp, arm_fp, arm_fn, arm_precision, arm_recall, arm_f1))

        # 3. Analyze the situations of PC-filtered edges in ARM results
        removed_common_parent_associations = []; removed_chained_associations = []
        ## 3.1 Identify the set of deducted associations with common links
        for cause in range(n_vars):
            outcomes = [outcome for outcome in range(n_vars) if tau_free_discovery_array[(cause, outcome)]==golden_standard_array[(cause, outcome)]==1 and outcome != cause]
            candidate_pairs = list(itertools.permutations(outcomes, 2))
            removed_common_parent_associations += [(pair[0], pair[1], cause) for pair in candidate_pairs\
                            if tau_free_discovery_array[pair]==golden_standard_array[pair]==0]
        ## 3.2 Identify the set of deducted associations with intermediate variables
        for cause in range(n_vars):
            outcomes = [outcome for outcome in range(n_vars) if tau_free_discovery_array[(cause, outcome)]==golden_standard_array[(cause, outcome)]==1 and outcome != cause]
            for outcome in outcomes:
                further_outcomes = [further_outcome for further_outcome in range(n_vars) if tau_free_discovery_array[(outcome, further_outcome)]==golden_standard_array[(outcome, further_outcome)]==1 and further_outcome != outcome]
                removed_chained_associations += [(cause, further_outcome, outcome) for further_outcome in further_outcomes if tau_free_discovery_array[(cause, further_outcome)]==golden_standard_array[(cause, further_outcome)]==0]
        ## 3.3 For each removed spurious associations, check its existence in the ARM array
        spurious_cp_associations = [spurious_link for spurious_link in removed_common_parent_associations if arm_results[(spurious_link[0],spurious_link[1])] == 1]
        spurious_chained_associations = [spurious_link for spurious_link in removed_chained_associations if arm_results[(spurious_link[0],spurious_link[1])] == 1]
        n_spurious_cp_associations = len(spurious_cp_associations); n_spurious_chained_associations = len(spurious_chained_associations)
        print("Compared with ARM, causalIoT removes {} spurious common-parent edges and {} spurious chained edges.".format(n_spurious_cp_associations, n_spurious_chained_associations))
        example_str = 'Example spurious cp associations:\n' + ','.join(['{}<-{}->{}'.format(index_device_dict[spurious_cp_association[0]].name, index_device_dict[spurious_cp_association[2]].name, index_device_dict[spurious_cp_association[1]].name)\
                                                for spurious_cp_association in spurious_cp_associations]) + '\n'
        example_str += 'Example spurious chained associations:\n' + ','.join(['{}->{}->{}'.format(index_device_dict[spurious_chained_association[0]].name, index_device_dict[spurious_chained_association[2]].name, index_device_dict[spurious_chained_association[1]].name)\
                                                for spurious_chained_association in spurious_chained_associations]) + '\n'
        print(example_str)

    def interpret_discovery_results(self, discovery_results:'np.ndarray'):
        # Return variables
        interactions:'list[tuple]' = []; interaction_types:'set' = set(); n_paths:'int' = 0
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        var_names = frame.var_names; n_vars = len(var_names); index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        golden_standard_array:'np.ndarray' = self.ground_truth_array

        # 1. Analyze the discovered device interactions (After aggregation of time lag)
        tau_free_discovery_array:'np.ndarray' = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        discovered_golden_array = np.zeros((n_vars, n_vars))
        assert(tau_free_discovery_array.shape == golden_standard_array.shape == (n_vars, n_vars))
        for (i, j), x in np.ndenumerate(golden_standard_array):
            if tau_free_discovery_array[(i, j)] == x == 1:
                interactions.append((index_device_dict[i].name, index_device_dict[j].name))
                interaction_types.add((index_device_dict[i].name[0], index_device_dict[j].name[0]))
                discovered_golden_array[(i, j)] = 1
        print("# of golden interactions, discovered interactions, interaction types, and type lists: {}({}), {}({}), {}, {}"\
                .format(np.sum(golden_standard_array), np.sum(golden_standard_array), np.sum(tau_free_discovery_array), np.sum(discovery_results), len(interaction_types), interaction_types))

        # 2. Analyze the formed device interaction chains.
        path_array = np.linalg.matrix_power(tau_free_discovery_array, 3); n_paths = np.sum(path_array)
        print("# of interaction chains: {}".format(n_paths))
        return interactions, interaction_types, n_paths

    """Function class for automatic selection of n_max_edges (based on prediction error analysis)"""

    def calculate_prediction_error(self, bayesian_fitter, sig_level, use_training=False):
        int_event_states = self.frame.training_events_states if use_training else self.frame.testing_events_states
        #training_event_states = self.frame.training_events_states
        security_guard = SecurityGuard(self.frame, bayesian_fitter, sig_level)
        security_guard.initialize_phantom_machine()
        prediction_errors = []
        fp_cases = defaultdict(int)
        n_fps = 0
        for evt_id, tup in enumerate(int_event_states):
            event, states = int_event_states[evt_id]
            score = security_guard._compute_event_anomaly_score(event, security_guard.phantom_state_machine)
            if score > security_guard.score_threshold:
                fp_cases["{}={}:{}".format(event.dev, event.value, ''.join([str(x) for x in states]))] += 1
                n_fps += 1
            prediction_errors.append(score)
            security_guard.phantom_state_machine.set_latest_states(states)
        return prediction_errors, fp_cases, n_fps
    
    def determine_best_max_edges(self, link_dict, sig_level):
        """
        For a given model and anomaly detection threshold, iterate over different n_max_edges settings, and use the clean testing data to select the best n_max_edges with the lowest # false alarms.

        Return values:
            n_max_edge: The identified best parameter setting for bayesian fitting which achieves the lowest prediction error
            prediction_errors: The calculated anomaly score (prediction error) for the clean testing data
            fp_cases: The reported false alarms and counts in the best n_max_edge setting.
            n_fps: The # false alarms reported in the best n_max_edge setting.
        """
        testing_result_dict = {}; min_n_fps = 1000000
        for n_max_edges in range(5, 11):
            bayesian_fitter = BayesianFitter(self.frame, self.tau_max, link_dict, n_max_edges=n_max_edges)
            prediction_errors, fp_cases, n_fps = self.calculate_prediction_error(bayesian_fitter, sig_level)
            min_n_fps = n_fps if n_fps < min_n_fps else min_n_fps
            testing_result_dict[n_max_edges] = (prediction_errors, fp_cases, n_fps)
        # We use the minimum false alarms as the metric to select the best_max_edge
        best_max_edge = [x for x in testing_result_dict.keys() if testing_result_dict[x][2]==min_n_fps][0]

        return best_max_edge, testing_result_dict[best_max_edge][0], testing_result_dict[best_max_edge][1], testing_result_dict[best_max_edge][2]

    """Function class for analyzing the advantage of causal model over association model"""

    def check_selection_bias(self, link_dict, n_max_edges, sig_level):
        """
        The selection bias is defined as: Err_prediction(testing-data, model) - Err_prediction(training-data, model)
            * The larger bias is, the worser the model's generality is.
        """
        #n_max_edges, _, _, _ = self.determine_best_max_edges(link_dict, sig_level)
        bayesian_fitter = BayesianFitter(self.frame, self.tau_max, link_dict, n_max_edges=n_max_edges, use_training=True)
        training_prediction_errors, _, _ = self.calculate_prediction_error(bayesian_fitter, sig_level, use_training=True)
        testing_prediction_errors, _, _ = self.calculate_prediction_error(bayesian_fitter, sig_level, use_training=False)
        avg_testing_error = statistics.mean(testing_prediction_errors); avg_training_error = statistics.mean(training_prediction_errors)
        testing_error_variance = statistics.variance(testing_prediction_errors); training_error_variance = statistics.variance(training_prediction_errors)
        print("Difference between average prediction error of training and testing datasets: {:.3f}-{:.3f}={:.3f}"\
                    .format(avg_testing_error, avg_training_error, avg_testing_error-avg_training_error))
        print("Variance of the prediction error in testing and training datasets: {:.3f} v.s. {:.3f}".format(testing_error_variance, training_error_variance))

    """Function class for anomaly injection"""

    def get_int_anomaly_dict(self):
        """
        The case_id represents the injected anomaly type.
            0: Sensor Fault (Anomalous low/high readings reported by the ambient sensor)
            1: Burglar Intrusion (Anomalous activation event of the motion sensor/contact sensor)
            2: Remote Control (Anomalous actuator activation/de-activation events)
            3: Malicious Automation Rule (How to simulate it?)
        """
        int_anomaly_dict = None
        int_anomaly_dict = {
            0: ['Brightness-Sensor'],
            1: ['Infrared-Movement-Sensor', 'Contact-Sensor'],
            2: ['Power-Sensor', 'Water-Meter', 'Dimmer', 'Switch'],
            3: ['Power-Sensor', 'Water-Meter', 'Contact-Sensor', 'Dimmer', 'Switch']
        }
        return int_anomaly_dict
    
    def get_int_collective_anomaly_dict(self, kmax):
        """
        The case_id represents the injected anomaly type.
            0: Burglar wandering at homes.
            1: Unauthorized access of a set of actuators
        """
        int_anomaly_dict = None
        int_anomaly_dict = {i:[] for i in range(2)}
        for k in range(kmax):
            int_anomaly_dict[0].append(['Infrared-Movement-Sensor', 'Contact-Sensor'])
            int_anomaly_dict[1].append(['Power-Sensor', 'Water-Meter', 'Dimmer', 'Switch'])
        return int_anomaly_dict
    
    def randomly_generate_automations(self, n_auto):
        attr_dev_dict = self.event_preprocessor.attr_dev_dict
        conditions = ['Brightness-Sensor', 'Infrared-Movement-Sensor', 'Contact-Sensor', 'Power-Sensor', 'Water-Meter', 'Contact-Sensor', 'Dimmer', 'Switch']
        outcomes = ['Power-Sensor', 'Water-Meter', 'Contact-Sensor', 'Dimmer', 'Switch']
        candidate_condition_devices = []; candidate_outcome_devices = []
        for condition_attr in conditions:
            candidate_condition_devices += attr_dev_dict[condition_attr]
        for outcome_attr in outcomes:
            candidate_outcome_devices += attr_dev_dict[outcome_attr]
        
        selected_condition_devices = random.choices(candidate_condition_devices, k=n_auto)
        automations = {}
        for cond_device in selected_condition_devices:
            action_device = random.choice(candidate_outcome_devices)
            condition_state = random.choice([0, 1]); action_state = random.choice([0, 1])
            automations[(cond_device, condition_state)] = (action_device, action_state)

        return automations
    
    def inject_contextual_anomalies(self, ground_truth_fitter:'BayesianFitter', sig_level, n_anomaly, case_id):
        """
        The case_id represents the injected anomaly type.
            0: Sensor Fault (Anomalous low/high readings reported by the ambient sensor)
            1: Burglar Intrusion (Anomalous activation event of the motion sensor/contact sensor)
            2: Remote Control (Anomalous actuator activation/de-activation events)
            3: Malicious Automation Rule (How to simulate it?)
        """
        # Auxillary variables
        device_description_dict = self.event_preprocessor.device_description_dict
        frame:'DataFrame' = self.frame
        var_names = frame.var_names
        benign_event_states:'list[tuple(AttrEvent,ndarray)]' = frame.testing_events_states

        # 0. Before injecting anomalies, get the lagged system states of the normal testing events
        security_guard = SecurityGuard(frame, ground_truth_fitter, sig_level)
        security_guard.initialize_phantom_machine()
        benign_lagged_states = defaultdict(list)
        benign_lagged_events = defaultdict(list)
        lagged_events = []
        for i in range(self.tau_max):
            lagged_events.append(frame.training_events_states[-self.tau_max+i][0])
        # For each benign event, we record the lagged states after it happens
        for evt_id, (event, states) in enumerate(benign_event_states):
            security_guard.phantom_state_machine.set_latest_states(states)
            lagged_events = [*lagged_events[-(self.tau_max-1):], event]
            benign_lagged_states[evt_id] = security_guard.phantom_state_machine.phantom_states.copy()
            benign_lagged_events[evt_id] = lagged_events

        # 1. Determine the candidate anomaly positions and store it to a dict with evt_id -> (anomalous-event, anomalous-states)
        candidate_position_anomalies = {}
        if case_id<=2:
            security_guard.initialize_phantom_machine()
            # 1. Determine the anomaly case based on the case id
            int_anomaly_dict = self.get_int_anomaly_dict()
            # 2. Determine the candidate injection positions given the ground truth and the anomaly type
            for evt_id, (event, states) in enumerate(benign_event_states):
                security_guard.phantom_state_machine.set_latest_states(states)
                candidate_anomalies = []
                for var_name in var_names: # For each position, traverse all potential anomalous events, and randomly select one
                    var_attr = device_description_dict[var_name]['attr']
                    last_state = security_guard.phantom_state_machine.phantom_states[var_names.index(var_name)]
                    candidate_event = AttrEvent('', '', var_name, var_attr, 1-last_state)
                    anomaly_score = security_guard._compute_event_anomaly_score(candidate_event, security_guard.phantom_state_machine)
                    if var_attr in int_anomaly_dict[case_id] and anomaly_score >= security_guard.score_threshold:
                            anomalous_states = states.copy(); anomalous_states[var_names.index(var_name)] = candidate_event.value
                            candidate_anomalies.append((candidate_event, anomalous_states, anomaly_score))
                if len(candidate_anomalies)>0:
                    candidate_anomalies = sorted(candidate_anomalies, key=lambda x:x[2], reverse=True)
                    candidate_anomalies = [(tup[0], tup[1]) for tup in candidate_anomalies]
                    candidate_position_anomalies[evt_id] = random.choice(candidate_anomalies[:3])
        elif case_id==3:
            # We hope to guarantee a minimum number of injected anomalies.
            # Each time the malicious automation rules are generated and the anomaly positions are determined, the number of anomalies should be larger than 1000
            malicious_automations = {}
            while len(list(candidate_position_anomalies.keys())) < n_anomaly:
                security_guard.initialize_phantom_machine()
                candidate_position_anomalies = {}
                n_auto = 15
                # 1. Randomly generate n_auto malicious automations: {(cond_dev, cond_state):(action_dev, action_state)}
                malicious_automations = self.randomly_generate_automations(n_auto)
                #print("Inserted malicious automation rules:")
                #pprint(malicious_automations)
                for evt_id, (event, states) in enumerate(benign_event_states):
                    security_guard.phantom_state_machine.set_latest_states(states)
                    # If the current event satisfies the condition of some malicious automation rule
                    if (event.dev, event.value) in malicious_automations.keys():
                        (action_dev, action_state) = malicious_automations[(event.dev, event.value)]
                        anomalous_event = AttrEvent('', '', action_dev, device_description_dict[action_dev]['attr'], action_state)
                        anomalous_states = states.copy(); anomalous_states[var_names.index(action_dev)] = action_state
                        # If the injection of an event here can be regarded as a normal device event (by the ground truth), this is not a suitable position.
                        if security_guard._compute_event_anomaly_score(anomalous_event, security_guard.phantom_state_machine) >= security_guard.score_threshold:
                            candidate_position_anomalies[evt_id] = (anomalous_event, anomalous_states)
            print("Inserted malicious automations:")
            pprint(malicious_automations)

        # 2. Insert the selected anomalies to the testing sequence
        n_candidate_anomalies = len(list(candidate_position_anomalies.keys()))
        selected_anomaly_positions = sorted(random.sample(list(candidate_position_anomalies.keys()),\
                    min(n_anomaly, n_candidate_anomalies)))
        new_testing_event_states:'list[tuple(AttrEvent,ndarray)]' = []; anomaly_positions = []; testing_benign_dict:'dict[int]' = {}
        new_event_id = 0
        security_guard.initialize_phantom_machine()
        for evt_id, (event, states) in enumerate(benign_event_states):
            # 2.1 First add benign testing events
            security_guard.phantom_state_machine.set_latest_states(states)
            new_testing_event_states.append((event, states))
            testing_benign_dict[new_event_id] = (evt_id, benign_lagged_states[evt_id], benign_lagged_events[evt_id])
            new_event_id += 1
            # 2.2 Add the anomalous event
            if evt_id in selected_anomaly_positions:
                anomaly_event_state = candidate_position_anomalies[evt_id]
                new_testing_event_states.append(anomaly_event_state)
                anomaly_positions.append(new_event_id)
                testing_benign_dict[new_event_id] = (evt_id, benign_lagged_states[evt_id], benign_lagged_events[evt_id]) # For rolling back
                new_event_id += 1
        #print("Total # injected anomalies: {}".format(len(anomaly_positions)))

        return new_testing_event_states, anomaly_positions, testing_benign_dict

    def inject_collective_anomalies(self, ground_truth_fitter:'BayesianFitter', sig_level, n_anomaly, case_id, kmax_threshold, step=0):
        assert(kmax_threshold>1)
        # Auxillary variables
        device_description_dict = self.event_preprocessor.device_description_dict
        nor_golden_array = self.nor_golden_array
        frame:'DataFrame' = self.frame
        var_names = frame.var_names
        benign_event_states:'list[tuple(AttrEvent,ndarray)]' = frame.testing_events_states
        security_guard = SecurityGuard(frame, ground_truth_fitter, sig_level)

        # 0. Before injecting anomalies, get the lagged system states of the normal testing events
        security_guard.initialize_phantom_machine()
        benign_lagged_states = defaultdict(list)
        benign_lagged_events = defaultdict(list)
        lagged_events = []
        for i in range(self.tau_max):
            lagged_events.append(frame.training_events_states[-self.tau_max+i][0])
        # For each benign event, we record the lagged states after it happens
        for evt_id, (event, states) in enumerate(benign_event_states):
            security_guard.phantom_state_machine.set_latest_states(states)
            lagged_events = [*lagged_events[-(self.tau_max-1):], event]
            benign_lagged_states[evt_id] = security_guard.phantom_state_machine.phantom_states.copy()
            benign_lagged_events[evt_id] = lagged_events
        
        def identify_candidate_collective_events(pre_event, state_machine, int_attrs):
            """
            We hope to identify a set of events which
                (1) have interactions with the one specified in pre_event,
                (2) have the attribute type of the int_attr, and
                (3) the anomaly score using the security guard satisfies the criteria (< or =>), and
            """
            candidate_events = []
            for var_name in var_names:
                var_attr = device_description_dict[var_name]['attr']
                pre_index = var_names.index(pre_event.dev); cur_index = var_names.index(var_name)
                if var_attr not in int_attrs or nor_golden_array[pre_index, cur_index]<=0:
                    continue
                last_state = state_machine.phantom_states[var_names.index(var_name)]
                candidate_event = AttrEvent('', '', var_name, var_attr, 1-last_state)
                anomaly_score = security_guard._compute_event_anomaly_score(candidate_event, state_machine)
                if  anomaly_score < security_guard.score_threshold:
                    candidate_events.append((candidate_event, anomaly_score))
            candidate_events = sorted(candidate_events, key=lambda x:x[1])
            candidate_events = [tup[0] for tup in candidate_events]
            return candidate_events

        # 1. Generate the anomaly chains
        candidate_position_chains = {}; debugging_position_chains = {}
        if case_id in [0, 1]:
            int_attrs_list = self.get_int_collective_anomaly_dict(kmax_threshold)[case_id]
            security_guard.initialize_phantom_machine()
            for evt_id, (event, states) in enumerate(benign_event_states):
                security_guard.phantom_state_machine.set_latest_states(states)
                candidate_anomaly_chains = []
                # 1. Generate candidate anomaly chains at each potential position
                kmax = random.choice(list(range(2,kmax_threshold+1,1)))
                for k in range(kmax):
                    int_attrs = int_attrs_list[k]
                    if k==0: # Determine the first candidate contextual anomaly
                        for var_name in var_names:
                            var_attr = device_description_dict[var_name]['attr']
                            if var_attr not in int_attrs:
                                continue
                            last_state = security_guard.phantom_state_machine.phantom_states[var_names.index(var_name)] # Generate the flipped state
                            candidate_event = AttrEvent('', '', var_name, var_attr, 1-last_state)
                            anomaly_score = security_guard._compute_event_anomaly_score(candidate_event, security_guard.phantom_state_machine)
                            if anomaly_score >= security_guard.score_threshold:
                                anomalous_states = states.copy(); anomalous_states[var_names.index(var_name)] = candidate_event.value
                                temp_state_machine = deepcopy(security_guard.phantom_state_machine)
                                temp_state_machine.set_latest_states(anomalous_states)
                                candidate_anomaly_chains.append([(candidate_event, anomalous_states, temp_state_machine, anomaly_score)])
                        # Sort all candidate contextual anomalies according to the anomalous level.
                        candidate_anomaly_chains = sorted(candidate_anomaly_chains, key=lambda x:x[0][3], reverse=True) 
                        candidate_anomaly_chains = [[(tup[0], tup[1], tup[2])] for chain in candidate_anomaly_chains for tup in chain]
                        #candidate_anomaly_chains = [(tup[0], tup[1], tup[2]) for chain in candidate_anomaly_chains for tup in chain]
                    else:
                        for cur_chain in candidate_anomaly_chains:
                            if len(cur_chain) != k: # If the current chain is a borken chain which cannot propagate anymore at some timestamp, do not consider it again
                                continue
                            last_event, last_states, state_machine = cur_chain[-1]
                            candidate_collective_events = identify_candidate_collective_events(last_event, state_machine, int_attrs)
                            if len(candidate_collective_events) == 0: # The current chain cannot be propgated again
                                continue
                            # JC NOTE: Ad-hoc codes here
                            collective_event = candidate_collective_events[0]
                            #collective_event = random.choice(candidate_collective_events)
                            anomalous_states = last_states.copy(); anomalous_states[var_names.index(collective_event.dev)] = collective_event.value
                            state_machine.set_latest_states(anomalous_states)
                            cur_chain.append((collective_event, anomalous_states, state_machine))
                # 2. Randomly select an anomaly chain at each position
                candidate_anomaly_chains = [chain for chain in candidate_anomaly_chains if len(chain)==kmax]
                if len(candidate_anomaly_chains) == 0:
                    continue
                #  JC NOTE: ad-hoc code here. We randomly select top-3 most anoamlous contextual anomaly chain as candidates
                selected_anomaly_chain = random.choice(candidate_anomaly_chains[:3])
                # selected_anomaly_chain = random.choice(candidate_anomaly_chains)
                candidate_position_chains[evt_id] = [(tup[0], tup[1]) for tup in selected_anomaly_chain] # We do not need the recorded state machine, we only need the anomalous event and the states
                debugging_position_chains[evt_id] = [(tup[0].dev, tup[0].value) for tup in selected_anomaly_chain]

        elif case_id == 2:
            security_guard.initialize_phantom_machine()
            # For each position, we first identify a set of candidate anomalies (candidate_anomaly_chains). Then we randomly select one as the selected anomaly chain.
            for evt_id, (event, states) in enumerate(benign_event_states):
                security_guard.phantom_state_machine.set_latest_states(states)
                candidate_anomaly_chains = []
                kmax = random.choice(list(range(2,kmax_threshold+1,1)))
                for k in range(kmax):
                    if k==0: # Determine the first candidate contextual anomaly, which should trigger some anomaly chains
                        for var_name in var_names:
                            last_state = security_guard.phantom_state_machine.phantom_states[var_names.index(var_name)] # Generate the flipped state
                            anomalous_state = 1 - last_state # We prepare to inejct a contextual anomaly (var_name, anomalous_state)
                            suitable_rule_chains = [chain for chain in self.rule_chains if chain[0]==(var_name,anomalous_state)]
                            if len(suitable_rule_chains)==0: # If there is no chains which can be triggered, we do not consider the anomaly
                                continue
                            chain_lens = [len(chain) for chain in suitable_rule_chains]; nor_chain_lens = [1.*l/sum(chain_lens) for l in chain_lens]
                            # Select an anomaly chain based on its lengths: The larger the length is, the larger the sampling probabilitiy
                            selected_anomaly_chain = random.choices(suitable_rule_chains, weights=chain_lens, k=1)[0]
                            var_attr = device_description_dict[var_name]['attr']
                            candidate_event = AttrEvent('', '', var_name, var_attr, anomalous_state)
                            anomaly_score = security_guard._compute_event_anomaly_score(candidate_event, security_guard.phantom_state_machine)
                            if anomaly_score >= security_guard.score_threshold:
                                anomalous_states = states.copy(); anomalous_states[var_names.index(var_name)] = candidate_event.value
                                temp_state_machine = deepcopy(security_guard.phantom_state_machine); temp_state_machine.set_latest_states(anomalous_states)
                                candidate_anomaly_chains.append([(candidate_event, anomalous_states, temp_state_machine, selected_anomaly_chain, True, anomaly_score)])
                        candidate_anomaly_chains = sorted(candidate_anomaly_chains, key=lambda x:x[0][5], reverse=True)
                        candidate_anomaly_chains = [[(tup[0], tup[1], tup[2], tup[3], tup[4])] for chain in candidate_anomaly_chains for tup in chain]
                    else:
                        for cur_chain in candidate_anomaly_chains:
                            rule_chain = cur_chain[-1][3]
                            if (cur_chain[-1][-1] is False) or (len(cur_chain)>=len(rule_chain)) or (len(cur_chain)==kmax): # If the current chain is a borken chain which cannot propagate anymore at some timestamp, do not consider it again
                                continue
                            last_event, last_states, state_machine = cur_chain[-1][0:3]
                            collective_dev, collective_state = rule_chain[k]
                            collective_event = AttrEvent(last_event.date, last_event.time, collective_dev, device_description_dict[collective_dev]['attr'], collective_state)
                            if (state_machine.phantom_states[var_names.index(collective_dev)] != collective_state) and security_guard._compute_event_anomaly_score(collective_event, state_machine) < security_guard.score_threshold:
                                anomalous_states = last_states.copy(); anomalous_states[var_names.index(collective_event.dev)] = collective_event.value
                                state_machine.set_latest_states(anomalous_states)
                                cur_chain.append((collective_event, anomalous_states, state_machine, rule_chain, True))
                            else:
                                # Stop tracking the chain
                                cur_chain[-1] = (cur_chain[-1][0], cur_chain[-1][1], cur_chain[-1][2], cur_chain[-1][3], False)
                # We want to identify chained automation executions with length at least 2
                candidate_anomaly_chains = [chain for chain in candidate_anomaly_chains if len(chain)==kmax]
                if len(candidate_anomaly_chains) == 0:
                    continue
                # JC NOTE: Ad-hoc codes here. We randomly select top-3 most anoamlous contextual anomaly chain as candidates
                selected_anomaly_chain = random.choice(candidate_anomaly_chains[:3])
                #selected_anomaly_chain = random.choice(candidate_anomaly_chains)
                candidate_position_chains[evt_id] = [(tup[0], tup[1]) for tup in selected_anomaly_chain] # We do not need the recorded state machine, we only need the anomalous event and the states
                debugging_position_chains[evt_id] = [(tup[0].dev, tup[0].value) for tup in selected_anomaly_chain]

        # 2. Insert the selected anomaly chain to the testing sequence
        new_testing_event_states:'list[tuple(AttrEvent,ndarray)]' = []; position_len_dict = {}; testing_benign_dict:'dict[int]' = {}
        # 2.1 Randomly select n_anomaly positions as the injectted anomaly
        candidate_positions = list(candidate_position_chains.keys())
        n_candidate_anomalies = len(candidate_positions)
        # JC NOTE: Ad-hoc position selection here
        selected_position_indices = list(range(1, n_candidate_anomalies, step))
        selected_anomaly_positions = [candidate_positions[index] for index in selected_position_indices]
        selected_anomaly_positions = selected_anomaly_positions[:n_anomaly]
        #selected_anomaly_positions = sorted(random.sample(list(candidate_position_chains.keys()),\
        #            min(n_anomaly, n_candidate_anomalies)))
        new_event_id = 0
        security_guard.initialize_phantom_machine()
        # 2.2 Traverse the benign event sequence. If it is normal: Add it to the testing seqeuence; Otherwise start injecting the anomaly chain
        #print("[KMAX={}] Collective anomaly injection.".format(kmax))
        instance_dict = {}
        for evt_id, (event, states) in enumerate(benign_event_states):
            security_guard.phantom_state_machine.set_latest_states(states)
            new_testing_event_states.append((event, states))
            testing_benign_dict[new_event_id] = (evt_id, benign_lagged_states[evt_id], benign_lagged_events[evt_id])
            new_event_id += 1
            if evt_id in selected_anomaly_positions:
                #print("     Position {} with phantom states: {}".format(new_event_id, security_guard.phantom_state_machine.get_lagged_states()))
                anomaly_chain = candidate_position_chains[evt_id]
                position_len_dict[new_event_id] = len(anomaly_chain)
                instance = ''
                for id, anomaly_event_state in enumerate(anomaly_chain):
                    if id == 0:
                        assert(security_guard._compute_event_anomaly_score(anomaly_event_state[0], security_guard.phantom_state_machine) >= security_guard.score_threshold)
                    else:
                        temp_state_machine = deepcopy(security_guard.phantom_state_machine)
                        for i in range(id):
                            temp_state_machine.set_latest_states(anomaly_chain[i][1])
                        assert(security_guard._compute_event_anomaly_score(anomaly_event_state[0], temp_state_machine) < security_guard.score_threshold)
                    #print("             {} with state {}".format(anomaly_event_state[0], anomaly_event_state[1]))
                    instance += '{}:{},'.format(anomaly_event_state[0].dev, anomaly_event_state[0].value)
                    new_testing_event_states.append(anomaly_event_state)
                    testing_benign_dict[new_event_id] = (evt_id, benign_lagged_states[evt_id], benign_lagged_events[evt_id]) # For rolling back
                    new_event_id += 1
                instance_dict[instance]=1 if instance not in instance_dict.keys() else instance_dict[instance]+1
        print("Injected collective anoamly sequences:")
        for evt_id in selected_anomaly_positions:
            printing_anomaly_chain = debugging_position_chains[evt_id]
            print("     {}".format(printing_anomaly_chain))
        print("# anomaly instances: {}".format(len(list(instance_dict.keys()))))
        #print("# anomaly cases and testing events: {} {}".format(len(anomaly_positions), len(new_testing_event_states)))
        return new_testing_event_states, position_len_dict, testing_benign_dict

    """Function classes for anomaly detection evaluation."""

    def evaluate_contextual_detection_accuracy(self, alarm_position_events, anomaly_positions, case_id, model_name='unknown'):
        anomaly_cases = self.get_int_anomaly_dict()[case_id]
        alarm_positions = [alarm[0] for alarm in alarm_position_events if alarm[1].attr in anomaly_cases]
        tp, fp, fn, precision, recall, f1 = self.calculate_accuracy(alarm_positions, anomaly_positions)
        #print("[{}-Model Contextual Anomaly Detection for Case-{}] Precision, recall, f1 = {:.3f}, {:.3f}, {:.3f}".format(model_name, case_id, precision, recall, f1))
        return precision, recall, f1
    
    def evaluate_collective_detection_accuracy(self, causal_alarm_position_len_dict , position_len_dict, kmax, case_id, model_name='unknown'):
        n_anomalies = len(list(position_len_dict.keys()))
        detected_len_dict = {0:0}
        n_detections = []
        n_fully_detected_chain = 0; n_detected_chain = 0; n_missing_chain = 0
        missing_alarm_dict = {}
        alarm_positions = sorted([k for k in causal_alarm_position_len_dict.keys()])
        for pos, l in position_len_dict.items():
            assert(l<=kmax)
            post_alarm_positions = [x for x in alarm_positions if x>=pos]
            if len(post_alarm_positions) == 0 or post_alarm_positions[0]-pos>=l-1:
                n_missing_chain += 1
                missing_alarm_dict[pos] = l
            else:
                nearest_alarm_position = post_alarm_positions[0]; alarm_l = causal_alarm_position_len_dict[nearest_alarm_position]
                detected_l = min(nearest_alarm_position+alarm_l-1, pos+l-1) - nearest_alarm_position + 1
                if detected_l > 1:
                    n_detected_chain += 1
                    n_detections.append(detected_l)
                else:
                    n_missing_chain += 1
                n_fully_detected_chain = n_fully_detected_chain+1 if detected_l==l else n_fully_detected_chain
                detected_len_dict[detected_l] = 1 if detected_l not in detected_len_dict.keys() else detected_len_dict[detected_l] + 1
        for k, v in detected_len_dict.items():
            detected_len_dict[k] = round(1.*v/n_anomalies, 4)
        avg_chain_len = round(statistics.mean(list(position_len_dict.values())), 3); avg_detection_len = round(statistics.mean(n_detections), 3)
        print("     # injected anomaly chains, Average length of chains, average detection length: {} {}, {}".format(n_anomalies, avg_chain_len, avg_detection_len))
        print("     ratio of detected chains (>2), fully reconstructed chains = {}, {}".format(1.*n_detected_chain/n_anomalies, 1.*n_fully_detected_chain/n_anomalies))
        return detected_len_dict, missing_alarm_dict

    def analyze_false_contextual_results(self, causal_bayesian_fitter:'BayesianFitter', ground_truth_fitter:'BayesianFitter', sig_level, case_id,\
                                            testing_event_states, anomaly_positions, testing_benign_dict,
                                            markov_miner, ocsvmer, hawatcher):

        anomaly_cases = self.get_int_anomaly_dict()[case_id]

        causal_guard = SecurityGuard(self.frame, causal_bayesian_fitter, sig_level); causal_guard.initialize_phantom_machine()
        ground_guard = SecurityGuard(self.frame, ground_truth_fitter, sig_level); ground_guard.initialize_phantom_machine()
        test_ground_fitter = BayesianFitter(causal_bayesian_fitter.frame, self.tau_max, self.golden_edges, causal_bayesian_fitter.n_max_edges, model_name='Test-golden', use_training=False)
        test_ground_fitter.bayesian_parameter_estimation()
        test_ground_guard = SecurityGuard(self.frame, test_ground_fitter, sig_level); test_ground_guard.initialize_phantom_machine()
        assert(causal_guard.phantom_state_machine.equal(ground_guard.phantom_state_machine.phantom_states))
        print("[CASE={}] Calculated score threshold Causal v.s. Ground = {} v.s. {}".format(case_id, causal_guard.score_threshold, ground_guard.score_threshold))

        markov_alarm_position_events = markov_miner.anomaly_detection(testing_event_states, testing_benign_dict, verbosity=1)
        ocsvm_alarm_position_events = ocsvmer.anomaly_detection(testing_event_states, verbosity=0)
        hawatcher_alarm_position_events = hawatcher.anomaly_detection(testing_event_states, testing_benign_dict, verbosity=1)

        tp=0; fp = 0; fn = 0; tn = 0
        tp_dev_dict = defaultdict(dict); fp_dev_dict = defaultdict(dict); fn_dev_dict = defaultdict(dict); tn_dev_dict = defaultdict(dict)
        for evt_id, (event, states)  in enumerate(testing_event_states):
            parents = ground_guard.bayesian_fitter.model.get_parents(event.dev)
            indices = [ground_guard.bayesian_fitter.expanded_var_names.index(parent)-ground_guard.bayesian_fitter.n_vars for parent in parents]
            parent_states = ground_guard.phantom_state_machine.get_indices_states(indices)
            parent_states:'list[tuple(str, int)]' = list(zip(parents, parent_states))

            causal_score = causal_guard._compute_event_anomaly_score(event, causal_guard.phantom_state_machine)
            ground_score = ground_guard._compute_event_anomaly_score(event, ground_guard.phantom_state_machine)
            test_ground_score = test_ground_guard._compute_event_anomaly_score(event, test_ground_guard.phantom_state_machine)

            causal_parents = set(causal_guard.bayesian_fitter.model.get_parents(event.dev))
            ground_parents = set(ground_guard.bayesian_fitter.model.get_parents(event.dev))
            common_parents = sorted(causal_parents.intersection(ground_parents))
            causal_exc_parents = sorted(causal_parents.difference(ground_parents))
            ground_exc_parents = sorted(ground_parents.difference(causal_parents))

            event_info = (event.dev, event.value, ground_score, causal_score)

            if evt_id in anomaly_positions:
                assert(ground_score >= ground_guard.score_threshold) 
                if (causal_score < causal_guard.score_threshold) and (event.attr in anomaly_cases):
                    # A false negative is detected.
                    fn_dev_dict[event_info]['count'] = 1 if 'count' not in fn_dev_dict[event_info].keys() else fn_dev_dict[event_info]['count']+1
                    fn+=1
                    info_str = 'Parent situation: {}\n\
                                [Model similarity] common parents = {}\n\
                                [Model difference] causal - ground = {}\n\
                                [Model difference] ground - causal = {}\n'.format(parent_states, common_parents, causal_exc_parents, ground_exc_parents)
                    fn_dev_dict[event_info]['info'] = info_str
                else:
                    tp_dev_dict[event_info]['count'] = 1 if 'count' not in tp_dev_dict[event_info].keys() else tp_dev_dict[event_info]['count']+1
                    tp+=1
                    info_str = 'Parent situation: {}\n\
                                [Model similarity] common parents = {}\n\
                                [Model difference] causal - ground = {}\n\
                                [Model difference] ground - causal = {}\n'.format(parent_states, common_parents, causal_exc_parents, ground_exc_parents)
                    tp_dev_dict[event_info]['info'] = info_str
                    if evt_id not in [alarm[0] for alarm in markov_alarm_position_events] and 'markov-fn-info' not in tp_dev_dict[event_info].keys():
                        # JC TODO: A markov false negative detected, what is the helpful info?
                        pass
            else:
                if (causal_score >= causal_guard.score_threshold) and (event.attr in anomaly_cases): # A false negative is detected
                    fp_dev_dict[event_info]['count'] = 1 if 'count' not in fp_dev_dict[event_info].keys() else fp_dev_dict[event_info]['count']+1
                    fp+=1
                    info_str = 'Parent situation: {}\n\
                                [Model similarity] common parents = {}\n\
                                [Model difference] causal - ground = {}\n\
                                [Model difference] ground - causal = {}\n'.format(parent_states, common_parents, causal_exc_parents, ground_exc_parents)
                    if ground_score >= ground_guard.score_threshold:
                        # Some rare behaviors in the training data happen in the testing data
                        info_str += "[Model shift] A model shift is detected. Training v.s. Testing probability: {} v.s. {}\n".format(1-ground_score, 1-test_ground_score)
                    fp_dev_dict[event_info]['info'] = info_str
                else:
                    tn_dev_dict[event_info]['count'] = 1 if 'count' not in tn_dev_dict[event_info].keys() else tn_dev_dict[event_info]['count']+1
                    tn+=1
                    info_str = 'Parent situation: {}\n\
                                [Model similarity] common parents = {}\n\
                                [Model difference] causal - ground = {}\n\
                                [Model difference] ground - causal = {}\n'.format(parent_states, common_parents, causal_exc_parents, ground_exc_parents)
                    tn_dev_dict[event_info]['info'] = info_str
                    if evt_id in [alarm[0] for alarm in markov_alarm_position_events] and 'markov-fn-info' not in tn_dev_dict[event_info].keys():
                        markov_false_alarm = [alarm for alarm in markov_alarm_position_events if alarm[0] == evt_id][0]
                        tn_dev_dict[event_info]['markov-fn-info'] = markov_false_alarm[2]
                    if evt_id in [alarm[0] for alarm in hawatcher_alarm_position_events] and 'haw-fn-info' not in tn_dev_dict[event_info].keys():
                        haw_false_alarm = [alarm for alarm in hawatcher_alarm_position_events if alarm[0] == evt_id][0]
                        tn_dev_dict[event_info]['haw-fn-info'] = haw_false_alarm[2]
            causal_guard.phantom_state_machine.flush(testing_benign_dict[evt_id][1])
            ground_guard.phantom_state_machine.flush(testing_benign_dict[evt_id][1])
            test_ground_guard.phantom_state_machine.flush(testing_benign_dict[evt_id][1])
        print("Anomaly case {}: FP, FN, Pre, Recall = {} {} {} {}".format(case_id, fp, fn, 1.*tp/(tp+fp), 1.*tp/(tp+fn)))
        for k, info_dict in tp_dev_dict.items():
            if info_dict['count'] >= 200:
                print("Significant true positive detected: {}".format(k))
                for index, info in info_dict.items():
                    print("{}: {}".format(index, info))
        for k, info_dict in fp_dev_dict.items():
            if info_dict['count'] >= 10:
                print("Significant false positive detected: {}".format(k))
                for index, info in info_dict.items():
                    print("{}: {}".format(index, info))
        for k, info_dict in fn_dev_dict.items():
            if info_dict['count'] >= 10:
                print("Significant false negative detected: {}".format(k))
                for index, info in info_dict.items():
                    print("{}: {}".format(index, info))
        for k, info_dict in tn_dev_dict.items():
            if info_dict['count'] >= 100 and len(list(info_dict.keys()))>2:
                print("Significant true negative detected: {}".format(k))
                for index, info in info_dict.items():
                    print("{}: {}".format(index, info))