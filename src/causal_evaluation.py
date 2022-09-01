import collections
import itertools
import numpy as np
import random
import pandas as pd
from src.tigramite.tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from numpy import ndarray
from src.event_processing import Hprocessor
from src.drawer import Drawer
from src.background_generator import BackgroundGenerator
from src.bayesian_fitter import BayesianFitter
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.tigramite.tigramite import plotting as ti_plotting
from collections import defaultdict

from src.tigramite.tigramite import pcmci
from src.tigramite.tigramite.independence_tests.chi2 import ChiSquare

class Evaluator():

    def __init__(self, background_generator, bayesian_fitter,\
        bk_level, pc_alpha):
        self.background_generator:'BackgroundGenerator' = background_generator
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        self.bk_level = bk_level; self.pc_alpha = pc_alpha
        self.tau_max = self.background_generator.tau_max
        self.golden_standard_dict = self._construct_golden_standard()
    
    """Function classes for golden standard construction."""

    def _construct_golden_standard(self):
        # JC NOTE: Currently we only consider hh-series datasets
        golden_standard_dict = {}
        golden_standard_dict['user'] = self._identify_user_interactions()
        golden_standard_dict['physics'] = self._identify_physical_interactions()
        golden_standard_dict['automation'] = self._identify_automation_interactions()

        aggregation_array = sum([golden_array for golden_array in golden_standard_dict.values()])
        assert(all([x <= 1 for index, x in np.ndenumerate(aggregation_array)])) # We hope that for any two devices, there exists only one type of interactions
        golden_standard_dict['aggregation'] = aggregation_array

        return golden_standard_dict

    def _identify_user_interactions(self):
        """
        In HH-series dataset, the user interaction is in the form of M->M, M->D, or D->M
        In the current frame, for any two devices, they have interactions iff (1) they are spatially adjacent, and (2) they are usually sequentially activated.
            (1) The identification of spatial adjacency is done by the background generator.
            (2) The identification of sequential activation is done by checking its occurrence within time lag tau_max.
        """
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        name_device_dict = frame.name_device_dict; n_vars = frame.n_vars
        training_events:'list[AttrEvent]' = [tup[0] for tup in frame.training_events_states] 
        # Return variables
        golden_user_array:'np.ndarray' = np.zeros((n_vars, n_vars, self.tau_max+1), dtype=np.int8)

        # 1. Spatial requirement verification: Fetch spatial-adjacency pairs and get the spatial background information.
        spatial_array:'np.ndarray' = self.background_generator.knowledge_dict['spatial']

        # 2. Temporal requirement verification: For any two devices, they should be sequentially triggered at least once.
        frequency_matrix:'np.ndarray' = np.zeros((n_vars, n_vars, self.tau_max+1), dtype=np.int32)
        last_act_dev = None; interval = 0
        for event in training_events:
            if event.value == 1: # An activation event is detected.
                if last_act_dev and interval <= self.tau_max:
                    frequency_matrix[name_device_dict[last_act_dev].index, name_device_dict[event.dev].index, interval] += 1
                last_act_dev = event.dev
                interval = 1
            else:
                interval += 1

        for i in range(n_vars): # If the aggregated activation frequency is larger than n_days (e.g., 100), then the edge satisfies the rule.
            for j in range(n_vars):
                frequency_matrix[i,j,:] = 1 if np.sum(frequency_matrix[i,j,:]) >= .5 * frame.n_days else 0 # JC NOTE: Ad-hoc temporal rules for golden standard identification (< half of days)

        # 3. Spatial requirement verification: For those pairs which satisfy the temporal requirement, further integrate spatial knowledge.
        assert(frequency_matrix.shape == spatial_array.shape)
        golden_user_array = sum([frequency_matrix, spatial_array])
        golden_user_array[golden_user_array <= 1] = 0; golden_user_array[golden_user_array > 1] = 1

        return golden_user_array

    def _identify_physical_interactions(self):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        name_device_dict = frame.name_device_dict; n_vars = frame.n_vars
        # Return variables
        golden_physical_array:'np.ndarray' = np.zeros((n_vars, n_vars, self.tau_max+1), dtype=np.int8)

        # JC TODO: Implement the logic here.

        return golden_physical_array
    
    def _identify_automation_interactions(self):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        name_device_dict = frame.name_device_dict; n_vars = frame.n_vars
        # Return variables
        golden_automation_array:'np.ndarray' = np.zeros((n_vars, n_vars, self.tau_max+1), dtype=np.int8)

        # JC TODO: Implement the logic here.

        return golden_automation_array

    def print_golden_standard(self, golden_type='aggregation'):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        tau_max = self.background_generator.tau_max; var_names = frame.var_names

        golden_array = self.golden_standard_dict[golden_type]
        tau_free_golden_array = sum(
            [golden_array[:,:,tau] for tau in range(1, tau_max+1)]
        )
        print("Golden array with type {} (After lag aggregation):".format(golden_type))
        df = pd.DataFrame(tau_free_golden_array, columns=var_names, index=var_names)
        print(df)

    """Function classes for causal discovery evaluation."""

    def interpret_discovery_results(self, discovery_results:'np.ndarray'):
        # Return variables
        interactions:'list[tuple]' = []; interaction_types:'set' = set(); n_paths:'int' = 0
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        var_names = frame.var_names; n_vars = len(var_names); index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        golden_standard_array:'np.ndarray' = self.golden_standard_dict['aggregation']

        # 1. Analyze the discovered device interactions (After aggregation of time lag)
        tau_free_discovery_array:'np.ndarray' = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        tau_free_golden_array:'np.ndarray' = sum([golden_standard_array[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_golden_array[tau_free_golden_array > 0] = 1
        discovered_golden_array = np.zeros((n_vars, n_vars))
        assert(tau_free_discovery_array.shape == tau_free_golden_array.shape == (n_vars, n_vars))
        for (i, j), x in np.ndenumerate(tau_free_golden_array):
            if tau_free_discovery_array[(i, j)] == x == 1:
                interactions.append((index_device_dict[i].name, index_device_dict[j].name))
                interaction_types.add((index_device_dict[i].name[0], index_device_dict[j].name[0]))
                discovered_golden_array[(i, j)] = 1
        print("# of golden interactions, discovered interactions, interaction types, and type lists: {}({}), {}({}), {}, {}"\
                .format(np.sum(tau_free_golden_array), np.sum(golden_standard_array), np.sum(tau_free_discovery_array), np.sum(discovery_results), len(interaction_types), interaction_types))

        # 2. Analyze the formed device interaction chains.
        path_array = np.linalg.matrix_power(tau_free_discovery_array, 3); n_paths = np.sum(path_array)
        print("# of interaction chains: {}".format(n_paths))
        return interactions, interaction_types, n_paths

    def precision_recall_calculation(self, golden_array:'np.ndarray', evaluated_array:'np.ndarray', verbosity):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        tp = 0; fp = 0; fn = 0; precision = 0.0; recall = 0.0
        fps = []; fns = []
        for index, x in np.ndenumerate(evaluated_array):
            if evaluated_array[index] == 1 and golden_array[index] == 1:
                tp += 1
            elif evaluated_array[index] == 1 and golden_array[index] == 0:
                fp += 1
                fps.append('{}->{}'.format(index_device_dict[index[0]].name, index_device_dict[index[1]].name))
            elif evaluated_array[index] == 0 and golden_array[index] == 1:
                fn += 1
                fns.append('{}->{}'.format(index_device_dict[index[0]].name, index_device_dict[index[1]].name))
        precision = tp * 1.0 / (tp + fp) if (tp+fp) != 0 else 0
        recall = tp * 1.0 / (tp + fn) if (tp+fn) != 0 else 0
        f1_score = 2.0*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
        if verbosity:
            print("Discovery FPs: {}".format(','.join(fps)))
            print("Discovery FNs: {}".format(','.join(fns)))
        return tp, fp, fn, precision, recall, f1_score

    def evaluate_discovery_accuracy(self, discovery_results:'np.ndarray', verbosity=0):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        var_names = frame.var_names; n_vars = len(var_names)
        golden_standard_array:'np.ndarray' = self.golden_standard_dict['aggregation']

        # 1. Plot the golden standard graph and the discovered graph. Note that the plot functionality requires PCMCI objects
        drawer = Drawer(self.background_generator.dataset)
        pcmci = PCMCI(dataframe=frame.training_dataframe, cond_ind_test=ChiSquare(), verbosity=-1)
        drawer.plot_interaction_graph(pcmci, discovery_results==1, 'mined-interaction-bklevel{}-alpha{}-threshold{}'\
                            .format(self.bk_level, int(1.0/self.pc_alpha), self.background_generator.filter_threshold), link_label_fontsize=10)
        drawer.plot_interaction_graph(pcmci, golden_standard_array==1, 'golden-interaction-threshold{}'\
                                        .format(int(self.background_generator.filter_threshold)))
        assert(discovery_results.shape == golden_standard_array.shape == (n_vars, n_vars, self.tau_max+1))
        # 2. Calculate the precision and recall for discovered results.
        #tp, fp, fn, precision, recall, f1 = self.precision_recall_calculation(golden_standard_array, discovery_results, verbosity=verbosity)
        # 3. Calculate the tau-free-precision and tau-free-recall for discovered results
        tau_free_discovery_array = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        tau_free_golden_array = sum([golden_standard_array[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_golden_array[tau_free_golden_array > 0] = 1
        tau_free_tp, tau_free_fp, tau_free_fn, tau_free_precision, tau_free_recall, tau_free_f1 = self.precision_recall_calculation(tau_free_golden_array, tau_free_discovery_array, verbosity=verbosity)
        return tau_free_tp+tau_free_fn, tau_free_precision, tau_free_recall, tau_free_f1

    def compare_with_arm(self, discovery_results:'np.ndarray', arm_results:'np.ndarray'):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        var_names = frame.var_names; n_vars = frame.n_vars; index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        golden_standard_array:'np.ndarray' = self.golden_standard_dict['aggregation']
        assert(discovery_results.shape == (n_vars, n_vars, self.tau_max + 1))
        assert(arm_results.shape == (n_vars, n_vars))

        # 1. Reduce discovery_results from (n_vars, n_vars, tau_max) to (n_vars, n_vars)
        tau_free_discovery_array = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        tau_free_golden_array = sum([golden_standard_array[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_golden_array[tau_free_golden_array > 0] = 1
        assert(tau_free_discovery_array.shape == tau_free_golden_array.shape == (n_vars, n_vars))

        # 2. Calculate the discovery accuracy for ARM and our algorithm, respectively
        causal_tp, causal_fp, causal_fn, causal_precision, causal_recall, causal_f1 = self.precision_recall_calculation(tau_free_golden_array, tau_free_discovery_array)
        arm_tp, arm_fp, arm_fn, arm_precision, arm_recall, arm_f1 = self.precision_recall_calculation(tau_free_golden_array, arm_results)
        print("Causal discovery tp, fp, fn, precision, recall, f1 = {}, {}, {}, {}, {}, {}".format(causal_tp, causal_fp, causal_fn, causal_precision, causal_recall, causal_f1))
        print("ARM tp, fp, fn, precision, recall, f1 = {}, {}, {}, {}, {}, {}".format(arm_tp, arm_fp, arm_fn, arm_precision, arm_recall, arm_f1))

        # 3. Analyze the situations of PC-filtered edges in ARM results
        removed_common_parent_associations = []; removed_chained_associations = []
        ## 3.1 Identify the set of deducted associations with common links
        for cause in range(n_vars):
            outcomes = [outcome for outcome in range(n_vars) if tau_free_discovery_array[(cause, outcome)]==tau_free_golden_array[(cause, outcome)]==1 and outcome != cause]
            candidate_pairs = list(itertools.permutations(outcomes, 2))
            removed_common_parent_associations += [(pair[0], pair[1], cause) for pair in candidate_pairs\
                            if tau_free_discovery_array[pair]==tau_free_golden_array[pair]==0]
        ## 3.2 Identify the set of deducted associations with intermediate variables
        for cause in range(n_vars):
            outcomes = [outcome for outcome in range(n_vars) if tau_free_discovery_array[(cause, outcome)]==tau_free_golden_array[(cause, outcome)]==1 and outcome != cause]
            for outcome in outcomes:
                further_outcomes = [further_outcome for further_outcome in range(n_vars) if tau_free_discovery_array[(outcome, further_outcome)]==tau_free_golden_array[(outcome, further_outcome)]==1 and further_outcome != outcome]
                removed_chained_associations += [(cause, further_outcome, outcome) for further_outcome in further_outcomes if tau_free_discovery_array[(cause, further_outcome)]==tau_free_golden_array[(cause, further_outcome)]==0]
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

    """Function classes for anomaly detection evaluation."""

    def construct_golden_standard_bayesian_fitter(self):
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        var_names = frame.var_names

        golden_standard_interaction_matrix:'np.ndarray' = self.golden_standard_dict['aggregation']
        link_dict = defaultdict(list)
        for (i, j, lag), x in np.ndenumerate(golden_standard_interaction_matrix):
            if x == 1:
                link_dict[var_names[j]].append((var_names[i],-lag))
        golden_bayesian_fitter = BayesianFitter(frame, self.tau_max, link_dict)
        golden_bayesian_fitter.construct_bayesian_model()
        return golden_bayesian_fitter

    def simulate_malicious_control(self, n_anomaly, maximum_length, anomaly_case):
        """
        The function injects anomalous events to the benign testing data, and generates the testing event sequences (simulating Malicious Control Attack).
        1. Randomly select n_anomaly positions.
        2. Traverse each benign event, and maintain a phantom state machine to track the system state.
        3. When reaching to a pre-designated position:
            * Determine the anomalous device and its state (flips of benign states).
            * Generate an anomalous event and insert it to the testing sequence.
            * If maximum_length > 1, propagate the anomaly according to the golden standard interaction.
        4. Record the nearest stable benign states for each testing sequence.
        Parameters:
            frame: The dataframe storing benign testing sequences
            n_anomaly: The number of injected anomalies
            maximum_length: The maximum length of the anomaly chain. If 0, the function injects single point anomalies.
        Returns:
            testing_event_sequence: The list of testing events (with injected anomalies)
            anomaly_positions: The list of injection positions in testing sequences
            stable_states_dict: Dict of {Key: position in testing log; Value: stable roll-back (event, states) pair}
        """
        # Return variables
        testing_event_states:'list[tuple(AttrEvent,ndarray)]' = []; anomaly_positions = []; testing_benign_dict:'dict[int]' = {}
        # Auxillary variables
        frame:'DataFrame' = self.background_generator.frame
        name_device_dict:'dict[DevAttribute]' = frame.name_device_dict
        benign_event_states:'list[tuple(AttrEvent,ndarray)]' = frame.testing_events_states

        # 1. Determine the set of anomaly events according to the anomaly case
        candidate_positions = []; anomalous_event_states = []; real_candidate_positions = []
        if anomaly_case == 1: # Outlier Intrusion: Ghost motion sensor activation event
            golden_bayesian_fitter = self.construct_golden_standard_bayesian_fitter()
            motion_devices = [dev for dev in frame.var_names if dev.startswith('M')]
            candidate_positions = sorted(random.sample(\
                                    range(self.tau_max, len(benign_event_states) - 1, self.tau_max + maximum_length),\
                                    n_anomaly))
            recent_devices = []
            for i, (event, stable_states) in enumerate(benign_event_states):
                recent_devices.append(event.dev)
                if i in candidate_positions:  # Determine the anomaly event
                    recent_tau_devices = list(set(recent_devices[-(self.tau_max):].copy()))
                    children_devices = golden_bayesian_fitter.get_expanded_children(recent_tau_devices) # Avoid child devices in the golden standard
                    potentially_anomaly_devices = [x for x in motion_devices if x not in children_devices and stable_states[name_device_dict[x].index] == 0]
                    if len(potentially_anomaly_devices) == 0:
                        continue
                    real_candidate_positions.append(i)
                    anomalous_device = random.choice(potentially_anomaly_devices); anomalous_device_index = name_device_dict[anomalous_device].index; anomalous_device_state = 1
                    anomalous_event = AttrEvent(date=event.date, time=event.time, dev=anomalous_device, attr='Case1-Anomaly', value=anomalous_device_state)
                    anomaly_states = stable_states.copy(); anomaly_states[anomalous_device_index] = anomalous_device_state
                    anomalous_event_states.append((anomalous_event, anomaly_states))
        else:
            pass

        # 2. Insert these anomaly events into the dataset
        testing_count = 0; anomaly_count = 0
        for i, (event, stable_states) in enumerate(benign_event_states):
            testing_event_states.append((event, stable_states))
            testing_benign_dict[testing_count] = i; testing_count += 1
            if i in real_candidate_positions: # If reaching the anomaly position.
                anomalous_event, anomalous_attr_state = anomalous_event_states[anomaly_count]
                testing_event_states.append((anomalous_event, anomalous_attr_state)); anomaly_positions.append(testing_count)
                testing_benign_dict[testing_count] = i; testing_count += 1; anomaly_count += 1

        return testing_event_states, anomaly_positions, testing_benign_dict

    def evaluate_detection_accuracy(self, golden_standard:'list[int]', result:'list[int]'):
        print("Golden standard with number {}: {}".format(len(golden_standard), golden_standard))
        print("Your result with number {}: {}".format(len(result), result))
        tp = len([x for x in result if x in golden_standard])
        fp = len([x for x in result if x not in golden_standard])
        fn = len([x for x in golden_standard if x not in result])
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0
        print("Precision, recall = {:.2f}, {:.2f}".format(precision, recall))