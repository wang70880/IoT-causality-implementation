from turtle import back
from src.event_processing import Hprocessor
import collections
import itertools
import numpy as np
import src.event_processing as evt_proc
import src.background_generator as bk_generator

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

    def _adhoc_estimate_single_discovery_accuracy(self, frame_id, tau, pcmci_results):
        """This function estimates the discovery accuracy for only user activity correlations.

        Args:
            frame_id (_type_): _description_
            tau (_type_): _description_
            pcmci_results (_type_): _description_

        Returns:
            _type_: _description_
        """
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        pcmci_array = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
        for outcome_attr in pcmci_results.keys(): # Transform the pcmci_results dict into array format (given the specific time lag \tau)
            for (cause_attr, lag) in pcmci_results[outcome_attr]:
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

if __name__ == '__main__':
    # Parameter setting
    dataset = 'hh101'
    partition_config = 10
    tau_max = 1; tau_min = 1
    verbosity = 0  # -1: No debugging information; 0: Debugging information in parallel module; 1: Debugging info in PCMCI class; 2: Debugging info in CIT implementations
    ## For stable-pc
    pc_alpha = 0.2
    max_conds_dim = 10
    maximum_comb = 1
    ## For MCI
    alpha_level = 0.005
    pcmci_results = {'D001': [('D001', -1)], 'D002': [('D002', -1), ('M001', -1), ('M008', -1)], 'LS001': [('LS001', -1), ('D002', -1)], 'LS002': [('LS002', -1), ('M002', -1), ('M004', -1)], 'LS003':         [('LS003', -1), ('M002', -1)], 'LS004': [('LS004', -1), ('LS016', -1), ('M008', -1)], 'LS005': [('LS005', -1)], 'LS006': [('LS006', -1), ('M006', -1), ('M003', -1)], 'LS007': [('LS007', -1), ('M003', -   1), ('M007', -1)], 'LS008': [('LS008', -1), ('LS013', -1)], 'LS009': [('LS009', -1), ('LS012', -1)], 'LS010': [('LS010', -1), ('LS001', -1), ('M008', -1)], 'LS011': [('LS011', -1), ('M011', -1)],         'LS012': [('LS012', -1), ('LS009', -1)], 'LS013': [('LS013', -1), ('M010', -1)], 'LS014': [('LS014', -1), ('LS009', -1), ('M001', -1)], 'LS015': [('LS015', -1), ('M008', -1), ('M011', -1)], 'LS016':      [('LS016', -1), ('M008', -1)], 'M001': [('M001', -1), ('D002', -1), ('M010', -1), ('M008', -1), ('LS001', -1), ('M005', -1), ('M012', -1)], 'M002': [('M002', -1), ('M008', -1), ('M004', -1), ('M003', -   1), ('M012', -1), ('M009', -1), ('M011', -1)], 'M003': [('M003', -1), ('M006', -1), ('M008', -1), ('M002', -1), ('M004', -1), ('LS006', -1), ('M009', -1), ('M010', -1), ('M005', -1), ('M001', -1),        ('M007', -1)], 'M004': [('M004', -1), ('M008', -1), ('M002', -1), ('M003', -1), ('M012', -1), ('M009', -1), ('M011', -1)], 'M005': [('M005', -1), ('M008', -1), ('M001', -1), ('M010', -1), ('M012', -1),   ('M004', -1)], 'M006': [('M006', -1), ('LS006', -1), ('M003', -1)], 'M007': [('M007', -1), ('M003', -1), ('LS007', -1)], 'M008': [('M008', -1), ('M012', -1), ('M003', -1), ('M009', -1), ('M002', -1),     ('M004', -1), ('M010', -1), ('M006', -1), ('M011', -1), ('LS006', -1), ('M007', -1)], 'M009': [('M009', -1), ('M012', -1), ('M008', -1), ('M003', -1), ('M011', -1), ('M010', -1), ('M005', -1)], 'M010':   [('M010', -1), ('M001', -1), ('M008', -1), ('M005', -1), ('M012', -1), ('M009', -1), ('M003', -1), ('M011', -1)], 'M011': [('M008', -1), ('LS015', -1), ('M012', -1), ('M011', -1), ('M009', -1),           ('LS011', -1), ('M001', -1), ('M010', -1), ('M003', -1), ('M002', -1), ('M005', -1), ('M004', -1)], 'M012': [('M009', -1), ('M012', -1), ('M008', -1), ('M003', -1), ('M010', -1), ('M005', -1), ('M001', - 1), ('M011', -1)], 'T101': [('T101', -1), ('M008', -1), ('M012', -1)], 'T102': [('LS013', -1)], 'T103': [('T103', -1)], 'T104': [('T104', -1)], 'T105': [('T105', -1)]}

    event_preprocessor = evt_proc.Hprocessor(dataset)
    attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
    background_generator = bk_generator.BackgroundGenerator(dataset, event_preprocessor, partition_config, tau_max)

    evaluator = Evaluator(dataset=dataset, event_processor=event_preprocessor, background_generator=background_generator, tau_max=tau_max)
    evaluator._adhoc_estimate_single_discovery_accuracy(0, 1, pcmci_results)