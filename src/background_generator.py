import src.event_processing as evt_proc
import collections
import itertools
import numpy as np
import pandas as pd

from collections import defaultdict 

from src.event_processing import Hprocessor
from src.genetic_type import DevAttribute, AttrEvent, DataFrame
from src.tigramite.tigramite import data_processing as pp

class BackgroundGenerator():

    def __init__(self, dataset:'str', frame:'DataFrame', tau_max, filter_threshold) -> None:
        self.dataset = dataset; self.frame:'DataFrame' = frame
        self.tau_max = tau_max; self.filter_threshold = filter_threshold

        # Each background array is of shape (n_vars, n_vars, tau_max+1)
        self.frequency_array, self.activation_frequency_array, self.normalized_frequency_array = self._temporal_identification()
        self.spatial_array = self._spatial_identification()
        self.knowledge_dict = {
            'temporal': self.normalized_frequency_array,
            'spatial': self.spatial_array,
        }
    
    """ Functions for background knowledge identification """

    def _temporal_identification(self):
        """
        Many temporal pairs stem from the disorder of IoT logs. Therefore, we filter out pairs with low frequencies.
        The selected pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).
        """
        # Auxillary variables
        name_device_dict:'dict[DevAttribute]' = self.frame.name_device_dict
        var_names = self.frame.var_names; n_vars = len(var_names)
        event_sequence:'list[AttrEvent]' = [tup[0] for tup in self.frame.training_events_states]
        # Return variables
        frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)
        normalized_frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)
        activation_frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)

        last_act_dev = None; interval = 0
        for event in event_sequence: # Count sequentially-activated pairs
            if event.value == 1:
                if last_act_dev and interval <= self.tau_max:
                    activation_frequency_array[name_device_dict[last_act_dev].index, name_device_dict[event.dev].index, interval] += 1
                last_act_dev = event.dev
                interval = 1
            else:
                interval += 1

        for i in range(len(event_sequence)): # Count the occurrence of each lagged attr pair
            for lag in range (1, self.tau_max + 1):
                if i + lag >= len(event_sequence):
                    continue
                former = name_device_dict[event_sequence[i].dev].index; latter = name_device_dict[event_sequence[i + lag].dev].index
                frequency_array[(former, latter, lag)] += 1
        normalized_frequency_array[frequency_array>=self.filter_threshold] = 1

        return frequency_array, activation_frequency_array, normalized_frequency_array
    
    def _testbed_area_information(self):
        # Return variables
        area_list = None
        area_connectivity_array = None

        if self.dataset == 'hh101':
            # In the list, each element (which is also a list) represents a physical area and deployed devices in that area.
            area_list = [['T102', 'D002', 'M001', 'LS001',\
                          'M011', 'LS011', 'M009', 'LS009', 'MA014', 'LS014', 'M012', 'LS012',\
                          'M010', 'LS010', 'D001', 'T101', 'T104', 'T105', 'M005', 'LS005', 'MA013', 'LS013', 'M008', 'LS008', 'M004', 'LS004',\
                          'MA016', 'LS016', 'M003', 'LS003'], \
                         ['D003', 'T103', 'MA015', 'LS015'], \
                         ['M002', 'LS002', 'M007', 'LS007', 'M006', 'LS006']]
            num_areas = len(area_list)
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            for i in range(num_areas):
                area_connectivity_array[i, i] = 1
            area_connectivity_array[0, 1] = 1; area_connectivity_array[0, 2] = 1
            area_connectivity_array[1, 0] = 1; area_connectivity_array[2, 0] = 1
        elif self.dataset == 'hh130':
            area_list = [['D002', 'T102', 'M001', 'LS001', 'M003', 'M004', 'LS003', 'LS004', 'T106', 'M005', 'LS005', 'LS009', 'LS010', 'M006', 'LS006'], \
                         ['M002', 'LS002', 'T103', 'LS008'], \
                         ['M011', 'T104', 'LS011', 'LS007']]
            num_areas = len(area_list)
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            for i in range(num_areas):
                area_connectivity_array[i, i] = 1
            area_connectivity_array[0, 1] = 1; area_connectivity_array[0, 2] = 1
            area_connectivity_array[1, 0] = 1; area_connectivity_array[2, 0] = 1

        return area_list, area_connectivity_array

    def _spatial_identification(self):
        # Auxillary variables
        area_list, area_connectivity_array = self._testbed_area_information()
        name_device_dict:'dict[DevAttribute]' = self.frame.name_device_dict
        var_names = self.frame.var_names; n_vars = len(var_names)
        # Return variables
        spatial_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)

        if area_list:
            for index, x in np.ndenumerate(area_connectivity_array):
                if x == 1:
                    pre_area = area_list[index[0]]; con_area = area_list[index[1]]
                    for element in itertools.product(pre_area, con_area): # Get the cartesian product for devices in these two areas
                        if all([element[i] in var_names for i in range(2)]):
                            spatial_array[name_device_dict[element[0]].index, name_device_dict[element[1]].index, :] = 1
        else: # No area information: assuming all devices are spatially adjacent
            spatial_array = np.ones(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)

        return spatial_array

    def print_background_knowledge(self):
        var_names = self.frame.var_names
        for bk_type in self.knowledge_dict.keys():
            knowledge_array = self.knowledge_dict[bk_type]
            tau_free_knowledge_array = sum(
            [knowledge_array[:,:,tau] for tau in range(1, self.tau_max + 1)]
            )
            print("Candidate edges for background type {} (After lag aggregation):".format(bk_type))
            df = pd.DataFrame(tau_free_knowledge_array, columns=var_names, index=var_names)
            print(df)

    """ Functions for candidate edge generation """

    def apply_background_knowledge(self, selected_links, knowledge_type):
        assert(selected_links is not None)
        n_filtered_edges = 0; n_qualified_edges = 0
        filtered_edges = []
        for worker_index, link_dict in selected_links.items():
            for outcome, cause_list in link_dict.items():
                new_cause_list = []
                for (cause, lag) in cause_list:
                    background_array:'np.ndarray' = self.knowledge_dict[knowledge_type]
                    if background_array[cause, outcome, abs(lag)] == 1:
                        new_cause_list.append((cause, lag)); n_qualified_edges += 1
                    else:
                        filtered_edges.append((cause, outcome, lag)); n_filtered_edges += 1
                selected_links[worker_index][outcome] = new_cause_list
        #print("[Background Generator] By applying {} knowledge, CausalIoT filtered {} edges.".format(knowledge_type, n_filtered_edges))
        #print("[Background Generator] # of candidate edges: {}.".format(n_qualified_edges))
        return selected_links

    def generate_candidate_interactions(self, apply_bk, autocorrelation_flag=True):
        var_names = self.frame.var_names; n_vars = len(var_names)

        if autocorrelation_flag:
            selected_links = {n: {m: [(i, -t) for i in range(n_vars) for \
                t in range(1, self.tau_max+1)] if m == n else [] for m in range(n_vars)} for n in range(n_vars)}
        else:
            selected_links = {n: {m: [(i, -t) for i in range(n_vars) if i != m for \
                t in range(1, self.tau_max + 1)] if m == n else [] for m in range(n_vars)} for n in range(n_vars)}
        if apply_bk >= 1:
            selected_links = self.apply_background_knowledge(selected_links, 'temporal')
        if apply_bk >= 2:
            selected_links = self.apply_background_knowledge(selected_links, 'spatial')
        
        # Transform the selected_links to a matrix form
        candidate_matrix = np.zeros((n_vars, n_vars, self.tau_max+1))
        for index, link_dict in selected_links.items():
            for outcome, causes in link_dict.items():
                for (cause, lag) in causes:
                    candidate_matrix[(cause, outcome, abs(lag))] = 1

        return selected_links, candidate_matrix