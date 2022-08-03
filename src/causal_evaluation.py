import collections
import itertools
import numpy as np
import random
from numpy import ndarray
from src.event_processing import Hprocessor
from src.background_generator import BackgroundGenerator
from src.bayesian_fitter import BayesianFitter
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
import statistics
from collections import defaultdict

class Evaluator():

    def __init__(self, event_processor, background_generator, bayesian_fitter) -> None:
        self.event_processor:'Hprocessor' = event_processor
        self.background_generator:'BackgroundGenerator' = background_generator
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        self.tau_max = self.background_generator.tau_max; self.filter_threshold = self.background_generator.filter_threshold
        self.golden_standard_dict = self._construct_golden_standard()
    
    def _construct_golden_standard(self):
        # JC NOTE: Currently we only consider hh-series datasets
        golden_standard_dict = {}
        golden_standard_dict['user'] = self._identify_user_interactions(self.filter_threshold)
        golden_standard_dict['physics'] = self._identify_physical_interactions()
        golden_standard_dict['automation'] = self._identify_automation_interactions()
        return golden_standard_dict

    def _identify_user_interactions(self, filter_threshold=0):
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
                activation_adjacency_dict[frame_id][tau][activation_adjacency_dict[frame_id][tau] < filter_threshold] = 0
                activation_adjacency_dict[frame_id][tau][activation_adjacency_dict[frame_id][tau] >= filter_threshold] = 1
        
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

    def inject_anomalies(self, frame_id, n_anomalies, maximum_length):
        """
        The function which is used for injecting anomalies to the testing data.

        Parameters:
            frame_id: The id of currently testing dataframe
            n_anomalies: The number of point anomalies which are going to be injected
            maximum_length: The maximum length of the anomaly chain. If 0, the function injects single point anomalies.

        Returns:
            testing_event_sequence: The list of testing events (with injected anomalies)
            anomaly_positions: The list of injection positions
            benign_position_dict: The index-in-testing-sequence:index-in-original-sequence dict
        """
        testing_event_sequence = []; anomaly_positions = []; benign_position_dict = {}
        original_frame = self.event_processor.frame_dict[frame_id]
        benign_testing_event_sequence = list(zip(original_frame['testing-attr-sequence'], original_frame['testing-state-sequence'])); n_benign_events = len(benign_testing_event_sequence)
        anomalous_sequences = []; anomaly_lag = 1 # Injecting lag-1 anomalies
        if n_anomalies == 0:
            testing_event_sequence = benign_testing_event_sequence.copy()
            anomaly_positions = []
            benign_position_dict = {x: x for x in range(n_benign_events)}
        else:
            # First determine the injection positions in the original event sequence, and generate the propagated anomaly sequence
            split_positions = sorted(random.sample(range(self.tau_max+1, n_benign_events-1, self.tau_max + maximum_length), n_anomalies))
            for split_position in split_positions:
                anomalous_sequence = []
                preceding_attr = benign_testing_event_sequence[split_position][0]; preceding_attr_index = original_frame['var-name'].index(preceding_attr)
                candidate_anomalous_attrs = [original_frame['var-name'][i] for i in list(np.where(self.background_generator.candidate_pair_dict[frame_id][anomaly_lag][preceding_attr_index] == 0)[0])]
                anomalous_attr = random.choice(candidate_anomalous_attrs)
                while anomalous_attr in self.bayesian_fitter.nointeraction_attr_list: # JC NOTE: We assume that the nointeraction attr will not be anomalous.
                    anomalous_attr = random.choice(candidate_anomalous_attrs)
                anomalous_sequence.append(anomalous_attr)
                for i in range(maximum_length - 1): # Propagate the anomaly chain (given pre-selected anomalous attr)
                    preceding_attr_index = original_frame['var-name'].index(anomalous_attr)
                    candidate_anomalous_attrs = [original_frame['var-name'][i] for i in list(np.where(self.background_generator.candidate_pair_dict[frame_id][anomaly_lag][preceding_attr_index] == 1)[0])]
                    if len(candidate_anomalous_attrs) == 0:
                        break
                    else:
                        anomalous_attr = random.choice(candidate_anomalous_attrs)
                        anomalous_sequence.append(anomalous_attr)
                anomalous_sequences.append(anomalous_sequence)
            
            # Then generate the testing sequence by combining original sequence with generated anomaly sequences
            starting_index = 0
            for i in range(0, n_anomalies):
                benign_starting_index = len(testing_event_sequence)
                testing_event_sequence += benign_testing_event_sequence[starting_index: split_positions[i]+1].copy()
                anomaly_start_index = len(testing_event_sequence)
                for (x, y) in list(zip([n for n in range(benign_starting_index, anomaly_start_index)], [m for m in range(starting_index, split_positions[i]+1)])):
                    benign_position_dict[x] = y
                anomaly_positions.append(anomaly_start_index)
                testing_event_sequence += [(attr, 1) for attr in anomalous_sequences[i]]
                starting_index = split_positions[i]+1
            benign_starting_index = len(testing_event_sequence)
            testing_event_sequence += benign_testing_event_sequence[starting_index:].copy()
            for (x, y) in list(zip([n for n in range(benign_starting_index, len(testing_event_sequence))], [m for m in range(starting_index, n_benign_events)])):
                benign_position_dict[x] = y
        assert(all([x not in list(benign_position_dict.keys()) for x in anomaly_positions]))
        assert(len(testing_event_sequence) == len(benign_testing_event_sequence)\
                         + sum([len(anomaly_sequence) for anomaly_sequence in anomalous_sequences]))
        
        #print("Injected positions: {}, anomalies: {}".format(anomaly_positions, anomalous_sequences))
        #print("Benign positions: {}".format(benign_position_dict.keys()))

        return testing_event_sequence, anomaly_positions, benign_position_dict 

    def evaluate_discovery_accuracy(self, discovery_results:'np.ndarray', golden_frame_id:'int', golden_type:'str'):
        precision = 0.0; recall = 0.0
        golden_standard_array:'np.ndarray' = self.golden_standard_dict[golden_frame_id][golden_type]
        assert(discovery_results.shape == golden_standard_array.shape)
        tp = np.count_nonzero(np.sum(golden_standard_array, discovery_results) == 2)
        fn = np.count_nonzero(golden_standard_array == 1) - tp; fp = np.count_nonzero(discovery_results == 1) - tp
        precision = tp * 1.0 / (tp + fp) if (tp+fp) != 0 else 0
        recall = tp * 1.0 / (tp + fn) if (tp+fn) != 0 else 0
        return tp+fn, precision, recall # Return # golden edges, precision, recall

    def evaluate_detection_accuracy(self, golden_standard:'list[int]', result:'list[int]'):
        print("Golden standard with number {}: {}".format(len(golden_standard), golden_standard))
        print("Your result with number {}: {}".format(len(result), result))
        tp = len([x for x in result if x in golden_standard])
        fp = len([x for x in result if x not in golden_standard])
        fn = len([x for x in golden_standard if x not in result])
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0
        print("Precision, recall = {:.2f}, {:.2f}".format(precision, recall))