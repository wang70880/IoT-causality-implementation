from turtle import back
from src.event_processing import Hprocessor
import collections
import itertools
import numpy as np
import random
import src.event_processing as evt_proc
import src.background_generator as bk_generator
import statistics

def _lag_name(attr:'str', lag:'int'):
    assert(lag >= 0)
    new_name = '{}({})'.format(attr, -1 * lag) if lag > 0 else '{}'.format(attr)
    return new_name

class Evaluator():

    def __init__(self, dataset, event_processor, background_generator, tau_max) -> None:
        self.dataset = dataset
        self.tau_max = tau_max
        self.event_processor = event_processor
        self.background_generator = background_generator

        self.user_correlation_dict = self.construct_user_correlation_benchmark()
        self.physical_correlation_dict = None
        self.automation_correlation_dict = None

        self.correlation_dict = {
            'activity': self.user_correlation_dict,
            'physics': self.physical_correlation_dict,
            'automation': self.automation_correlation_dict
        }
    
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

    def construct_user_correlation_benchmark(self):
        user_correlation_dict = {}
        for frame_id in range(self.event_processor.frame_count):
            user_correlation_dict[frame_id] = {}
            for tau in range(1, self.tau_max + 1):
                temporal_array = self.background_generator.temporal_pair_dict[frame_id][tau] # Temporal information
                spatial_array = self.background_generator.spatial_pair_dict[frame_id][tau] # Spatial information
                functionality_array =  self.background_generator.functionality_pair_dict['activity'] # Functionality information
                user_correlation_array = temporal_array * spatial_array * functionality_array
                user_correlation_dict[frame_id][tau] = user_correlation_array
        return user_correlation_dict

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
        discovery_array = pcmci_array * self.background_generator.functionality_pair_dict['activity']; truth_array = self.user_correlation_dict[frame_id][tau]
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
    
    def inject_type1_anomalies(self, frame_id, n_anomalies, maximum_length):
        """_summary_

        Args:
            frame_id (_type_): _description_
            n_anomalies (_type_): _description_
            maximum_length (_type_): _description_

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
        
        print("Injected positions: {}, anomalies: {}".format(anomaly_positions, anomalous_sequences))
        print("Benign positions: {}".format(benign_position_dict.keys()))

        return testing_event_sequence, anomaly_positions, benign_position_dict 