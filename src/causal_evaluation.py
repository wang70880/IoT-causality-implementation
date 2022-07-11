from turtle import back
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

    def __init__(self, dataset, event_processor, background_generator, bayesian_fitter, tau_max) -> None:
        self.dataset = dataset
        self.tau_max = tau_max
        self.event_processor:'Hprocessor' = event_processor
        self.background_generator:'BackgroundGenerator' = background_generator
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter

        self.golden_standard_dict = {}
    
    def evaluate_detection_accuracy(self, golden_standard:'list[int]', result:'list[int]'):
        print("Golden standard with number {}: {}".format(len(golden_standard), golden_standard))
        print("Your result with number {}: {}".format(len(result), result))
        tp = len([x for x in result if x in golden_standard])
        fp = len([x for x in result if x not in golden_standard])
        fn = len([x for x in golden_standard if x not in result])
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0
        print("Precision, recall = {:.2f}, {:.2f}".format(precision, recall))

    def candidate_interaction_matching(self, frame_id=0, tau=1, interactions_list=[]):
        match_count = 0
        candidate_interaction_array = self.background_generator.candidate_pair_dict[frame_id][tau]
        for interaction in interactions_list:
            if candidate_interaction_array[interaction[0], interaction[1]] == 1:
                match_count += 1
        return match_count

    def construct_golden_standard(self, filter_threshold=0):
        # JC NOTE: Currently we only consider hh-series datasets
        self.golden_standard_dict['user'] = self._identify_user_interactions(filter_threshold)
        self.golden_standard_dict['physics'] = self._identify_physical_interactions()
        self.golden_standard_dict['automation'] = self._identify_automation_interactions()

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

    def _print_pair_list(self, interested_array):
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        pair_list = []
        for index, x in np.ndenumerate(interested_array):
            if x == 1:
                pair_list.append((attr_names[index[0]], attr_names[index[1]]))
        print("Pair list with lens {}: {}".format(len(pair_list), pair_list))

    def print_benchmark_info(self,frame_id= 0, tau= 1, type = ''):
        """Print out the identified device correlations.

        Args:
            frame_id (int, optional): _description_. Defaults to 0.
            tau (int, optional): _description_. Defaults to 1.
            type (str, optional): 'activity' or 'physics' or 'automation'
        """
        print("The {} correlation dict for frame_id = {}, tau = {}: ".format(type, frame_id, tau))
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        self._print_pair_list(self.correlation_dict[type][frame_id][tau])

    def estimate_single_discovery_accuracy(self, frame_id, tau, result):
        """
        This function estimates the discovery accuracy for only user activity correlations.
        Moreover, it specifies a certain frame_id and a tau for the discovered result.
        """
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        pcmci_array = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
        for outcome_attr in result.keys(): # Transform the pcmci_results dict into array format (given the specific time lag \tau)
            for (cause_attr, lag) in result[outcome_attr]:
                pcmci_array[attr_names.index(cause_attr), attr_names.index(outcome_attr)] = 1 if lag == -1 * tau else 0
        #print("[frame_id={}, tau={}] Evaluating accuracy for user-activity correlations".format(frame_id, tau))
        discovery_array = pcmci_array * self.background_generator.functionality_pair_dict['activity']; truth_array = None
        n_discovery = np.sum(discovery_array); truth_count = np.sum(truth_array)
        tp = 0; fn = 0; fp = 0
        fn_list = []
        fp_list = []
        for idx, x in np.ndenumerate(truth_array):
            if truth_array[idx[0], idx[1]] == discovery_array[idx[0], idx[1]] == 1:
                tp += 1
            elif truth_array[idx[0], idx[1]] == 1:
                fn += 1
                fn_list.append("{} -> {}".format(attr_names[idx[0]], attr_names[idx[1]]))
            elif discovery_array[idx[0], idx[1]] == 1:
                fp_list.append("{} -> {}".format(attr_names[idx[0]], attr_names[idx[1]]))
                fp += 1
        precision = (tp * 1.0) / (tp + fp)
        recall = (tp * 1.0) / (tp + fn)
        #print("* FNs: {}".format(fn_list))
        #print("* FPs: {}".format(fp_list))
        #print("n_discovery = %d" % n_discovery
        #          + "\ntruth_count = %s" % truth_count 
        #          + "\ntp = %d" % tp
        #          + "\nfn = %d" % fn 
        #          + "\nfp = %d" % fp
        #          + "\nprecision = {}".format(precision)
        #          + "\nrecall = {}".format(recall))
        return truth_count, precision, recall
    
    def estimate_average_discovery_accuracy(self, tau, result_dict):
        truth_count_list = []; precision_list = []; recall_list = []
        for frame_id, result in result_dict.items():
            truth_count, precision, recall = self.estimate_single_discovery_accuracy(frame_id, tau, result)
            truth_count_list.append(truth_count); precision_list.append(precision); recall_list.append(recall)
        return statistics.mean(truth_count_list), statistics.mean(precision_list), statistics.mean(recall_list)

    def simulate_malicious_control(self, frame, n_anomaly, maximum_length):
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
        testing_event_sequence = []; anomaly_positions = []; anomaly_events = []; stable_states_dict = {}
        benign_event_sequence = list(zip(frame['testing-attr-sequence'], frame['testing-state-sequence']))
        candidate_positions = sorted(random.sample(\
                                    range(self.tau_max, len(benign_event_sequence) - 1, self.tau_max + maximum_length),\
                                    n_anomaly))

        benign_count = 0; testing_count = 0
        for benign_count in range(len(benign_event_sequence)):
            # First update the list according to the benign events and the corresponding stable_states vector
            event = benign_event_sequence[benign_count]; stable_states = frame['testing-data'].values[benign_count].copy() 
            testing_event_sequence.append(event); stable_states_dict[testing_count] = (event, stable_states); testing_count += 1 
            # Reach the anomaly position: inject the anomaly after the benign events.
            if benign_count in candidate_positions:
                # Determine the abnormal attribute
                candidate_anomalous_attrs = frame['var-name'].copy()
                anomalous_attr = random.choice(candidate_anomalous_attrs)
                while anomalous_attr in self.bayesian_fitter.nointeraction_attr_list: # JC NOTE: We assume that the nointeraction attr will not be anomalous.
                    anomalous_attr = random.choice(candidate_anomalous_attrs)
                anomalous_attr_state = int(1 - stable_states[frame['var-name'].index(event[0])]) # Flip the state
                anomaly_events.append((anomalous_attr, anomalous_attr_state))
                testing_event_sequence.append((anomalous_attr, anomalous_attr_state)); anomaly_positions.append(testing_count); stable_states_dict[testing_count] = (event, stable_states)
                testing_count += 1
                if maximum_length > 1:
                    pass

        return testing_event_sequence, anomaly_events, anomaly_positions, stable_states_dict

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