import collections
import itertools
import numpy as np
import random
import pandas as pd
import statistics
from src.tigramite.tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from numpy import ndarray
from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
from src.drawer import Drawer
from src.background_generator import BackgroundGenerator
from src.bayesian_fitter import BayesianFitter
from src.security_guard import SecurityGuard
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.benchmark.association_miner import AssociationMiner
from src.tigramite.tigramite import plotting as ti_plotting
from collections import defaultdict
from pprint import pprint

from src.tigramite.tigramite import pcmci
from src.tigramite.tigramite.independence_tests.chi2 import ChiSquare

class Evaluator():

    def __init__(self, event_preprocessor, frame, tau_max, pc_alpha):
        self.event_preprocessor = event_preprocessor
        self.frame:'DataFrame' = frame
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.golden_edges, self.golden_array, self.nor_golden_array = self._construct_golden_standard()
    
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
    """Function classes for ground truth construction."""

    def _construct_golden_standard(self):
        """
        For all known interactions between two devices (in the ground truth file), we further determine the golden time lag
        """
        ground_truth_array = None
        try:
            golden_df = pd.read_csv(self.event_preprocessor.ground_truth, encoding="utf-8", delim_whitespace=True, header=0, dtype=int)
            ground_truth_array = golden_df.values
        except:
            raise FileNotFoundError("Cannot construct the ground truth! (Maybe the ground truth file is missing?)")
        
        # Generate the interaction set, given the ground truth array
        # Currently we don't know the time lag yet, so we add edges with all lags
        selected_links = defaultdict(list)
        for (cause, outcome), x in np.ndenumerate(ground_truth_array):
            if x == 0:
                continue
            for lag in range(1, self.tau_max+1):
                selected_links[outcome].append((cause, -lag))
        # For all candidate time lags, we filter them according to the statistical test, and get the ranked edge sets
        asso_miner = AssociationMiner(self.event_preprocessor, self.frame, self.tau_max, self.pc_alpha*100)
        golden_edges, golden_array, nor_golden_array = asso_miner.interaction_mining(selected_links)
        return golden_edges, golden_array, nor_golden_array

    """Function classes for causal discovery evaluation."""

    def precision_recall_calculation(self, golden_array:'np.ndarray', evaluated_array:'np.ndarray',\
                        filtered_edge_infos:'dict'=None, identified_edge_infos:'dict'=None):
        # Auxillary variables
        frame:'DataFrame' = self.frame
        index_device_dict:'dict[DevAttribute]' = frame.index_device_dict

        # Calculate tp, fp, fn
        tp = 0; tn = 0; fp = 0; fn = 0; precision = 0.0; recall = 0.0
        for index, x in np.ndenumerate(evaluated_array):
            if evaluated_array[index] == 1 and golden_array[index] == 1:
                tp += 1
            elif evaluated_array[index] == 1 and golden_array[index] == 0:
                fp += 1
            elif evaluated_array[index] == 0 and golden_array[index] == 1:
                fn += 1
            else:
                tn += 1

        # Save debugging information for each edge (if applied)
        tp_dict = None; tn_dict = None; fp_dict = None; fn_dict = None
        if filtered_edge_infos and identified_edge_infos:
            tp_dict = defaultdict(dict); tn_dict = defaultdict(dict); fp_dict = defaultdict(dict); fn_dict = defaultdict(dict)
            for index, x in np.ndenumerate(evaluated_array):
                pair_info = {
                    'adev-pair': '{}->{}'.format(index_device_dict[index[0]].name, index_device_dict[index[1]].name),
                    'attr-pair': (index_device_dict[index[0]].attr, index_device_dict[index[1]].attr),
                    'spatial': (index_device_dict[index[0]].location, index_device_dict[index[1]].location)
                }
                if evaluated_array[index] == 1 and golden_array[index] == 1:
                    edge_infos = [edge_info for edge_info in identified_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    max_ate = 0.0
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        if abs(ate) > max_ate:
                            pair_info['pval'] = round(edge_info[2], 3)
                            #pair_info['cpt'] = cpt
                            pair_info['cate'] = round(ate, 3)
                    tp_dict[pair_info['adev-pair']] = pair_info
                elif evaluated_array[index] == 1 and golden_array[index] == 0:
                    edge_infos = [edge_info for edge_info in identified_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    max_ate = 0.0
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        if abs(ate) > max_ate:
                            pair_info['pval'] = round(edge_info[2], 3)
                            #pair_info['cpt'] = edge_info[2]
                            pair_info['cate'] = round(ate, 3)
                    fp_dict[pair_info['adev-pair']] = pair_info
                elif evaluated_array[index] == 0 and golden_array[index] == 1:
                    edge_infos = [edge_info for edge_info in filtered_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    filter_infos = []; max_ate = 0.
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        max_ate = ate if abs(ate) > max_ate else max_ate
                        filter_infos.append((edge_info[0][1][1], [(index_device_dict[cond[0]].name, cond[1]) for cond in edge_info[1]], round(edge_info[3],3), round(ate,3))) # lag, conds, pval, ate
                    sorted(filter_infos, key=lambda x: x[-1], reverse=True)
                    pair_info['filter-info'] = filter_infos
                    pair_info['cate'] = round(max_ate, 3)
                    fn_dict[pair_info['adev-pair']] = pair_info
                else:
                    edge_infos = [edge_info for edge_info in filtered_edge_infos[index[1]] if edge_info[0][1][0]==index[0]]
                    filter_infos = []; max_ate = 0.
                    for edge_info in edge_infos:
                        cpt, ate = frame.get_cpt_and_ate(prior_vars=[edge_info[0][1]], latter_vars=[(edge_info[0][0], 0)], cond_vars=[], tau_max=self.tau_max)
                        max_ate = ate if abs(ate) > max_ate else max_ate
                        filter_infos.append((edge_info[0][1][1], [(index_device_dict[cond[0]].name, cond[1]) for cond in edge_info[1]], round(edge_info[3],3), round(ate,3))) # lag, conds, pval, ate
                    sorted(filter_infos, key=lambda x: x[-1], reverse=True)
                    pair_info['filter-info'] = filter_infos
                    pair_info['cate'] = round(max_ate, 3)
                    tn_dict[pair_info['adev-pair']] = pair_info
            tp_dict = dict(sorted(tp_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))
            fp_dict = dict(sorted(fp_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))
            tn_dict = dict(sorted(tn_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))
            fn_dict = dict(sorted(fn_dict.items(), key=lambda item: abs(item[1]['cate']), reverse=True))

        precision = tp * 1.0 / (tp + fp) if (tp+fp) != 0 else 0
        recall = tp * 1.0 / (tp + fn) if (tp+fn) != 0 else 0
        f1_score = 2.0*precision*recall / (precision+recall) if (precision+recall) != 0 else 0

        return tp, fp, fn, precision, recall, f1_score, tp_dict, tn_dict, fp_dict, fn_dict

    def evaluate_discovery_accuracy(self, discovery_results:'np.ndarray', ground_truth_array:'np.ndarray',\
                            filtered_edge_infos:'dict'=None, identified_edge_infos:'dict'=None, verbosity=0):
        assert(len(discovery_results.shape)==len(ground_truth_array.shape)==2) 
        # Auxillary variables
        frame:'DataFrame' = self.frame
        var_names = frame.var_names; n_vars = len(var_names)

        # 1. Plot the golden standard graph and the discovered graph. Note that the plot functionality requires PCMCI objects
        #drawer = Drawer(self.background_generator.dataset)
        #pcmci = PCMCI(dataframe=frame.training_dataframe, cond_ind_test=ChiSquare(), verbosity=-1)
        #drawer.plot_interaction_graph(pcmci, discovery_results==1, 'mined-interaction-bklevel{}-alpha{}'\
        #                    .format(self.bk_level, int(1.0/self.pc_alpha)), link_label_fontsize=10)
        #drawer.plot_interaction_graph(pcmci, golden_standard_array==1, 'golden-interaction')
        # 3. Calculate the tau-free-precision and tau-free-recall for discovered results
        tp, fp, fn, precision, recall, f1, tp_dict, tn_dict, fp_dict, fn_dict \
             = self.precision_recall_calculation(ground_truth_array, discovery_results, filtered_edge_infos, identified_edge_infos)
        assert(tp+fn == np.sum(ground_truth_array))
        if verbosity:
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
        return tp, fp, fn, precision, recall, f1

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

    def inject_contextual_anomalies(self, ground_truth_fitter:'BayesianFitter', sig_level, n_anomaly, case_id):
        """
        The case_id represents the injected anomaly type.
        """
        # Return variables
        new_testing_event_states:'list[tuple(AttrEvent,ndarray)]' = []; anomaly_positions = []; testing_benign_dict:'dict[int]' = {}
        # Auxillary variables
        device_description_dict = self.event_preprocessor.device_description_dict
        frame:'DataFrame' = self.frame
        var_names = frame.var_names
        benign_event_states:'list[tuple(AttrEvent,ndarray)]' = frame.testing_events_states
        anomaly_infos = defaultdict(list)

        # 0. Initialize the state machine
        security_guard = SecurityGuard(frame, ground_truth_fitter, sig_level)
        security_guard.initialize_phantom_machine()

        # 1. Randomly pick n_anomaly positions for anomaly injection
        candidate_positions = sorted(random.sample(range(0, len(benign_event_states)), n_anomaly))

        # 2. Generate the eventual testing sequences with anomaly injected
        new_event_id = 0
        for evt_id, (event, states) in enumerate(benign_event_states):
            # 2.1 First add benign testing events
            security_guard.phantom_state_machine.set_latest_states(states)
            new_testing_event_states.append((event, states))
            testing_benign_dict[new_event_id] = evt_id
            new_event_id += 1

            if evt_id in candidate_positions:
                # 2.2 Determine the set of potential anomaly events for current anomaly position
                candidate_anomalies = defaultdict(list)
                for var_name in var_names:
                    last_state = security_guard.phantom_state_machine.phantom_states[var_names.index(var_name)]
                    # Generate state-flipped anomalous event
                    candidate_event = AttrEvent('', '', var_name, device_description_dict[var_name]['attr'], 1-last_state)
                    # If the filpped event is deemed as an anomalous event by the ground truth fitter: Store it as a candidate anomalous event 
                    if security_guard._compute_event_anomaly_score(candidate_event, security_guard.phantom_state_machine) >= security_guard.score_threshold:
                        anomalous_event = candidate_event
                        anomalous_states = states.copy(); anomalous_states[var_names.index(var_name)] = anomalous_event.value
                        candidate_anomalies[evt_id].append((anomalous_event, anomalous_states))
                
                # 2.2 Randomly generate an anomaly event from the candidate set, and record it into the final testing event sequences
                if len(candidate_anomalies[evt_id]) > 0:
                    anomaly_event_state = random.choice(candidate_anomalies[evt_id])
                    new_testing_event_states.append(anomaly_event_state)
                    anomaly_positions.append(new_event_id)
                    anomaly_infos[anomaly_event_state[0].attr].append(new_event_id)
                    testing_benign_dict[new_event_id] = evt_id # For rolling back
                    new_event_id += 1
        print("Total # injected anomalies: {}".format(len(anomaly_positions)))

        return new_testing_event_states, anomaly_positions, anomaly_infos, testing_benign_dict

    """Function classes for anomaly detection evaluation."""

    def evaluate_false_alarm_rate(self, sig_level, link_dict):
        """
        Using testing data (without anomaly injection), this function compares the prediction error (i.e., false positives) of both causal and association model.
            1. Call determine_best_max_edge to select the best n_max_edges for both models.
        """
        association_miner = AssociationMiner(self.event_preprocessor, self.frame, self.tau_max, self.pc_alpha)
        causal_max_edge, causal_prediction_errors, causal_fp_cases, causal_n_fps = self.determine_best_max_edges(link_dict, sig_level)
        asso_max_edge, asso_prediction_errors, asso_fp_cases, asso_n_fps = self.determine_best_max_edges(association_miner.mining_edges, sig_level)
        n_testing_events = len(causal_prediction_errors)

        causual_avg_error = statistics.mean(causal_prediction_errors); association_avg_error = statistics.mean(asso_prediction_errors)
        print("[Anomaly-FP-Analysis (sig-level={})] Benchmark selection: Causal Model v.s. Association Model".format(sig_level))
        print("     (1) Maximum edges ({} v.s. {}), False alarm rate ({} v.s. {}), Average prediction error ({} v.s. {})"\
                    .format(causal_max_edge, asso_max_edge, causal_n_fps, asso_n_fps, causual_avg_error, association_avg_error))
        causal_adv_cases = {}
        for x in list(asso_fp_cases.keys()):
            if x not in list(causal_fp_cases.keys()):
                causal_adv_cases[x] = asso_fp_cases[x]
        #pprint(causal_adv_cases)

    def evaluate_contextual_detection_accuracy(self, anomaly_infos:'dict[list]', alarm_infos:'dict[list]', model_name='unknown'):
        anomaly_positions = [position for positions in anomaly_infos.values() for position in positions]
        alarm_positions = [position for positions in alarm_infos.values() for position in positions]
        tp, fp, fn, precision, recall, f1 = self.calculate_accuracy(alarm_positions, anomaly_positions)
        print("[{}-Model Contextual Anomaly Detection] Overall precision, recall, f1 = {:.3f}, {:.3f}, {:.3f}".format(model_name, precision, recall, f1))
        if self.event_preprocessor.dataset == 'contextact':
            anomalous_cases = {
                'Burglar Intrusion': anomaly_infos['Infrared Movement Sensor'] + anomaly_infos['Contact Sensor'],
                'Sensor Fault': anomaly_infos['Brightness Sensor'] + anomaly_infos['Power Sensor'] + anomaly_infos['Water Meter'],
                'Remote Control': anomaly_infos['Dimmer'] + anomaly_infos['Switch']
            }
            alarm_cases = {
                'Burglar Intrusion': alarm_infos['Infrared Movement Sensor'] + alarm_infos['Contact Sensor'],
                'Sensor Fault': alarm_infos['Brightness Sensor'] + alarm_infos['Power Sensor'] + alarm_infos['Water Meter'],
                'Remote Control': alarm_infos['Dimmer'] + alarm_infos['Switch']
            }

            for anomaly_case, anomaly_case_positions in anomalous_cases.items():
                alarm_case_positions = alarm_cases[anomaly_case] if anomaly_case in alarm_cases.keys() else []
                tp, fp, fn, precision, recall, f1 = self.calculate_accuracy(alarm_case_positions, anomaly_case_positions)
                print("     Case ({}) detection results: {}, {}, {}".format(anomaly_case, precision, recall, f1))