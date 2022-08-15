import collections
import itertools
import numpy as np
import random
from src.tigramite.tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from numpy import ndarray, var
from src.event_processing import Hprocessor
from src.drawer import Drawer
from src.background_generator import BackgroundGenerator
from src.bayesian_fitter import BayesianFitter
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.tigramite.tigramite import plotting as ti_plotting
import statistics
from collections import defaultdict

from src.tigramite.tigramite import pcmci
from src.tigramite.tigramite.independence_tests.chi2 import ChiSquare

class Evaluator():

    def __init__(self, event_processor, background_generator, bayesian_fitter, bk_level=0, pc_alpha=0., filter_threshold=1) -> None:
        self.event_processor:'Hprocessor' = event_processor
        self.background_generator:'BackgroundGenerator' = background_generator
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        self.bk_level = bk_level; self.pc_alpha = pc_alpha; self.filter_threshold = filter_threshold
        self.tau_max = self.background_generator.tau_max
        self.golden_standard_dict = self._construct_golden_standard()
    
    """Function classes for golden standard construction."""

    def _construct_golden_standard(self):
        # JC NOTE: Currently we only consider hh-series datasets
        golden_standard_dict = {}
        golden_standard_dict['user'] = self._identify_user_interactions()
        golden_standard_dict['physics'] = self._identify_physical_interactions()
        golden_standard_dict['automation'] = self._identify_automation_interactions()
        return golden_standard_dict

    def _identify_user_interactions(self):
        """
        In HH-series dataset, the user interaction is in the form of M->M, M->D, or D->M
        For any two devices, they have interactions iff (1) they are spatially adjacent, and (2) they are usually sequentially activated.
            (1) The identification of spatial adjacency is done by the background generator.
            (2) The identification of sequential activation is done by counting with a filtering mechanism (as specified by the filter_threshold parameter).
        """
        # Return variables
        user_interaction_dict = defaultdict(dict)
        # Auxillary variables
        name_device_dict = self.event_processor.name_device_dict; n_vars = self.event_processor.n_vars
        frame_dict:'dict[DataFrame]' = self.event_processor.frame_dict
        # Fetch spatial-adjacency pairs
        spatial_array:'np.ndarray' = self.background_generator.correlation_dict['spatial'][0][1] # The index does not matter, since for all frames and tau, the adjacency matrix is the same.
        # Analyze activation intervals to determine tau for each spatial adjacency pair
        activation_adjacency_dict = defaultdict(dict)
        for frame_id in frame_dict.keys(): # Initialize the dict
            for tau in range(1, self.tau_max + 1):
                activation_adjacency_dict[frame_id][tau] = np.zeros((n_vars, n_vars), dtype=np.int32)
        for frame_id in frame_dict.keys():
            frame: 'DataFrame' = frame_dict[frame_id]
            training_events:'list[AttrEvent]' = [tup[0] for tup in frame.training_events_states] 
            last_act_dev = None; interval = 0 # JC NOTE: The interval identification requires that it is better to only keep devices of interest in the dataset. Otherwise, the interval can be enlarged by other devices (e.g., T or LS).
            for event in training_events:
                if event.dev.startswith(('M', 'D')) and event.value == 1: # An activation event is detected.
                    if last_act_dev and interval <= self.tau_max:
                        activation_adjacency_dict[frame_id][interval][name_device_dict[last_act_dev].index, name_device_dict[event.dev].index] += 1
                    last_act_dev = event.dev
                    interval = 1
                elif event.dev.startswith(('M', 'D')) and event.value == 0:
                    interval += 1
        for frame_id in frame_dict.keys(): # Normalize the frequency dict using the filter_threshold parameter
            for tau in range(1, self.tau_max + 1):
                activation_adjacency_dict[frame_id][tau][activation_adjacency_dict[frame_id][tau] < self.filter_threshold] = 0
                activation_adjacency_dict[frame_id][tau][activation_adjacency_dict[frame_id][tau] >= self.filter_threshold] = 1
        
        user_interaction_dict:'dict[np.ndarray]' = {}
        for frame_id in frame_dict.keys(): # Finally, generate the user_interaction_dict (which should be spatially adjacent and have legitimate activation orders)
            golden_standard_array = np.zeros((n_vars, n_vars, self.tau_max + 1), dtype=np.int8)
            for tau in range(1, self.tau_max + 1):
                activation_adjacency_dict[frame_id][tau][spatial_array == 0] = 0 # Combine spatial knowledge with sequential activation knowledge
                for i in range(n_vars):
                    for j in range(n_vars):
                        golden_standard_array[i, j, tau] = activation_adjacency_dict[frame_id][tau][i, j]
            user_interaction_dict[frame_id] = golden_standard_array
        return user_interaction_dict

    def _identify_physical_interactions(self):
        # In HH-series dataset, there is no physical interactions...
        return {}
    
    def _identify_automation_interactions(self):
        # In HH-series dataset, there is no automation interactions...
        return {}

    """Function classes for causal discovery evaluation."""

    def interpret_discovery_results(self, discovery_results:'np.ndarray', golden_frame_id:'int', golden_type:'str'):
        # Return variables
        interactions:'list[tuple]' = []; interaction_types:'set' = set(); n_paths:'int' = 0
        # Auxillary variables
        frame:'DataFrame' = self.event_processor.frame_dict[golden_frame_id]
        var_names = frame.var_names; n_vars = len(var_names); index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        golden_standard_array:'np.ndarray' = self.golden_standard_dict[golden_type][golden_frame_id]

        # 1. Analyze the discovered device interactions
        tau_free_discovery_array = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        tau_free_golden_array = sum([golden_standard_array[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_golden_array[tau_free_golden_array > 0] = 1
        discovered_golden_array = np.zeros((n_vars, n_vars))
        assert(tau_free_discovery_array.shape == tau_free_golden_array.shape == (n_vars, n_vars))
        for (i, j), x in np.ndenumerate(tau_free_golden_array):
            if tau_free_discovery_array[(i, j)] == x == 1:
                interactions.append((index_device_dict[i].name, index_device_dict[j].name))
                interaction_types.add((index_device_dict[i].name[0], index_device_dict[j].name[0]))
                discovered_golden_array[(i, j)] = 1
        print("# of discovered interactions: {}".format(len(interactions)))
        print("# of interaction types and type lists: {}, {}".format(len(interaction_types), interaction_types))

        # 2. Analyze the formed device interaction chains.
        path_array = np.linalg.matrix_power(discovered_golden_array, 5); n_paths = np.sum(path_array)
        print("# of interaction chains: {}".format(n_paths))
        return interactions, interaction_types, n_paths

    def precision_recall_calculation(self, golden_array:'np.ndarray', evaluated_array:'np.ndarray'):
        tp = 0; fp = 0; fn = 0; precision = 0.0; recall = 0.0
        for index, x in np.ndenumerate(evaluated_array):
            if evaluated_array[index] == 1 and golden_array[index] == 1:
                tp += 1
            elif evaluated_array[index] == 1 and golden_array[index] == 0:
                fp += 1
            elif evaluated_array[index] == 0 and golden_array[index] == 1:
                fn += 1
        precision = tp * 1.0 / (tp + fp) if (tp+fp) != 0 else 0
        recall = tp * 1.0 / (tp + fn) if (tp+fn) != 0 else 0
        return tp, fp, fn, precision, recall

    def evaluate_discovery_accuracy(self, discovery_results:'np.ndarray', golden_frame_id:'int', golden_type:'str'):
        # Auxillary variables
        frame:'DataFrame' = self.event_processor.frame_dict[golden_frame_id]
        var_names = frame.var_names; n_vars = len(var_names)
        golden_standard_array:'np.ndarray' = self.golden_standard_dict[golden_type][golden_frame_id]

        # 1. Plot the golden standard graph and the discovered graph. Note that the plot functionality requires PCMCI objects
        drawer = Drawer(self.event_processor.dataset)
        pcmci = PCMCI(dataframe=frame.training_dataframe, cond_ind_test=ChiSquare(), verbosity=-1)
        drawer.plot_interaction_graph(pcmci, discovery_results==1, 'mined-interaction-bklevel{}-alpha{}-threshold{}'\
                            .format(self.bk_level, int(1.0/self.pc_alpha), self.filter_threshold))
        drawer.plot_interaction_graph(pcmci, golden_standard_array==1, 'golden-interaction-threshold{}'.format(int(self.filter_threshold)))
        assert(discovery_results.shape == golden_standard_array.shape == (n_vars, n_vars, self.tau_max + 1))
        # 2. Calculate the precision and recall for discovered results.
        tp, fp, fn, precision, recall = self.precision_recall_calculation(golden_standard_array, discovery_results)
        # 3. Calculate the tau-free-precision and tau-free-recall for discovered results
        tau_free_discovery_array = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        tau_free_golden_array = sum([golden_standard_array[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_golden_array[tau_free_golden_array > 0] = 1
        tau_free_tp, tau_free_fp, tau_free_fn, tau_free_precision, tau_free_recall = self.precision_recall_calculation(tau_free_golden_array, tau_free_discovery_array)
        return tau_free_tp+tau_free_fn, tau_free_precision, tau_free_recall

    def compare_with_arm(self, discovery_results:'np.ndarray', arm_results:'np.ndarray', golden_frame_id:'int', golden_type:'str'):
        # Auxillary variables
        frame:'DataFrame' = self.event_processor.frame_dict[golden_frame_id]
        var_names = frame.var_names; n_vars = len(var_names); index_device_dict:'dict[DevAttribute]' = frame.index_device_dict
        golden_standard_array:'np.ndarray' = self.golden_standard_dict[golden_type][golden_frame_id]
        assert(discovery_results.shape == (n_vars, n_vars, self.tau_max + 1))
        assert(arm_results.shape == (n_vars, n_vars))

        # 1. Reduce discovery_results from (n_vars, n_vars, tau_max) to (n_vars, n_vars)
        tau_free_discovery_array = sum([discovery_results[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_discovery_array[tau_free_discovery_array > 0] = 1
        tau_free_golden_array = sum([golden_standard_array[:,:,tau] for tau in range(1, self.tau_max + 1)]); tau_free_golden_array[tau_free_golden_array > 0] = 1

        # 2. Calculate the discovery accuracy for ARM and our algorithm, respectively
        causal_tp, causal_fp, causal_fn, causal_precision, causal_recall = self.precision_recall_calculation(tau_free_golden_array, tau_free_discovery_array)
        arm_tp, arm_fp, arm_fn, arm_precision, arm_recall = self.precision_recall_calculation(tau_free_golden_array, arm_results)
        print("Causal discovery tp, fp, fn, precision, recall = {}, {}, {}, {}, {}".format(causal_tp, causal_fp, causal_fn, causal_precision, causal_recall))
        print("ARM tp, fp, fn, precision, recall = {}, {}, {}, {}, {}".format(arm_tp, arm_fp, arm_fn, arm_precision, arm_recall))


    """Function classes for anomaly detection evaluation."""

    def construct_golden_standard_bayesian_fitter(self, int_frame_id=0, int_type='user'):
        # Auxillary variables
        int_frame:'DataFrame' = self.event_processor.frame_dict[int_frame_id]
        var_names = self.bayesian_fitter.var_names

        golden_standard_interaction_matrix:'np.ndarray' = self.golden_standard_dict[int_type][int_frame_id]
        link_dict = defaultdict(list)
        for (i, j, lag), x in np.ndenumerate(golden_standard_interaction_matrix):
            if x == 1:
                link_dict[var_names[j]].append((var_names[i],-lag))
        golden_bayesian_fitter = BayesianFitter(int_frame, self.tau_max, link_dict)
        golden_bayesian_fitter.construct_bayesian_model()
        return golden_bayesian_fitter

    def simulate_malicious_control(self, int_frame_id, n_anomaly, maximum_length, anomaly_case):
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
        frame:'DataFrame' = self.event_processor.frame_dict[int_frame_id]
        name_device_dict:'dict[DevAttribute]' = frame.name_device_dict
        benign_event_states:'list[tuple(AttrEvent,ndarray)]' = frame.testing_events_states

        # 1. Determine the set of anomaly events according to the anomaly case
        candidate_positions = []; anomalous_event_states = []; real_candidate_positions = []
        if anomaly_case == 1: # Outlier Intrusion: Ghost motion sensor activation event
            golden_bayesian_fitter = self.construct_golden_standard_bayesian_fitter(int_frame_id, 'user')
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