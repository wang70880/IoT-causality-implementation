from tkinter import W
import src.event_processing as evt_proc
import collections
import itertools
import numpy as np
import pandas as pd

from collections import defaultdict 

from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
from src.genetic_type import DevAttribute, AttrEvent, DataFrame
from src.tigramite.tigramite import data_processing as pp

class BackgroundGenerator():

    def __init__(self, event_preprocessor:'GeneralProcessor', frame_id, tau_max) -> None:
        self.event_preprocessor = event_preprocessor
        self.dataset = event_preprocessor.dataset
        self.frame_id = frame_id; self.frame:'DataFrame' = event_preprocessor.frame_dict[frame_id]
        self.tau_max = tau_max

        # Each background array is of shape (n_vars, n_vars, tau_max+1)
        self.frequency_array, self.activation_frequency_array,self.normalized_frequency_array = self._temporal_identification()
        self.spatial_array = self._spatial_identification()
        self.user_array = self._user_activity_identification()
        self.physical_array = self._physical_identification()
        self.knowledge_dict = {
            'temporal': self.normalized_frequency_array,
            'spatial': self.spatial_array,
            'user': self.user_array,
            'physical': self.physical_array
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
        normalized_frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int8)
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

        normalized_frequency_array[frequency_array>=self.frame.n_days] = 1

        return frequency_array, activation_frequency_array, normalized_frequency_array
    
    def _testbed_area_information(self):
        # Return variables
        area_list = None
        area_connectivity_array = None

        if self.dataset == 'hh130':
            area_list = [['D002', 'T102', 'M001', 'LS001', 'M003', 'M004', 'LS003', 'LS004', 'T106', 'M005', 'LS005', 'LS009', 'LS010', 'M006', 'LS006'], \
                         ['M002', 'LS002', 'T103', 'LS008'], \
                         ['M011', 'T104', 'LS011', 'LS007']]
            num_areas = len(area_list)
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            for i in range(num_areas):
                area_connectivity_array[i, i] = 1
            area_connectivity_array[0, 1] = 1; area_connectivity_array[0, 2] = 1
            area_connectivity_array[1, 0] = 1; area_connectivity_array[2, 0] = 1
        elif self.dataset == 'contextact':
            # 1. Put all devices which are in the same location into a list, and construct the area_list
            device_description_dict:'dict' = self.event_preprocessor.device_description_dict
            location_devices_dict:'dict' = defaultdict(list)
            location_index_dict:'dict' = defaultdict(int)
            for dev in device_description_dict.keys():
                dev_entry = device_description_dict[dev]
                location_devices_dict[dev_entry['location']].append(dev)
            area_list = []
            for i, area in enumerate(location_devices_dict):
                location_index_dict[area] = i
                area_list.append(location_devices_dict[area])
            # 2. Manually identify the spatial adjacency of different locations.
            num_areas = len(area_list)
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            adjacent_areas_list = [['Kitchen', 'Stove', 'Dining Room', 'Living Room', 'Hallway First Floor', 'First Floor', 'Main Entrance'],\
                              ['Hallway First Floor', 'Stairway', 'Hallway Second Floor'],\
                              ['Hallway Second Floor', 'Bathroom', 'Study Room', 'Bedroom']]
            for adjacent_areas in adjacent_areas_list:
                for prior in adjacent_areas:
                    for latter in adjacent_areas:
                        area_connectivity_array[location_index_dict[prior],location_index_dict[latter]] = 1

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

    def _user_activity_identification(self):
        # Auxillary variables
        index_device_dict:'dict[DevAttribute]' = self.frame.index_device_dict
        var_names = self.frame.var_names; n_vars = len(var_names)
        # Return variables
        user_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)

        # 1. Construct the physical channels
        user_attributes = None
        if self.dataset == 'contextact':
            user_attributes = ['Infrared Movement Sensor', 'Switch']

        # 2. Leverage the physical channels to identify related device pairs
        if user_attributes:
            for i in range(n_vars):
                for j in range(n_vars):
                    prior, latter = (index_device_dict[i].attr, index_device_dict[j].attr)
                    if prior in user_attributes or latter in user_attributes:
                        user_array[i,j,:] = 1

        return user_array 

    def _physical_identification(self):
        # Auxillary variables
        index_device_dict:'dict[DevAttribute]' = self.frame.index_device_dict
        var_names = self.frame.var_names; n_vars = len(var_names)
        # Return variables
        physical_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)

        # 1. Construct the physical channels
        physical_channels = None
        if self.dataset == 'contextact':
            physical_channels = {
                'Brightness Sensor': ['Brightness Sensor', 'Dimmer', 'Rollershutter'],
                'Dimmer': ['Brightness Sensor', 'Power Sensor', 'Dimmer'],
                'Contact Sensor': ['Contact Sensor', 'Water Meter', 'Power Sensor'],
                'Humidity Sensor': ['Humidity Sensor', 'Water Meter', 'Contact Sensor'],
                'Power Sensor': ['Power Sensor', 'Contact Sensor', 'Rollershutter'],
                'Rollershutter': ['Rollershutter', 'Brightness Sensor', 'Dimmer', 'Power Sensor'],
                'Water Meter': ['Water Meter', 'Contact Sensor', 'Power Sensor'],
                'Switch': [],
                'Infrared Movement Sensor': []
            }

        # 2. Leverage the physical channels to identify related device pairs
        if physical_channels:
            for i in range(n_vars):
                for j in range(n_vars):
                    prior, latter = (index_device_dict[i].attr, index_device_dict[j].attr)
                    if latter in physical_channels[prior]:
                        #print("{}->{} ({}->{})".format(\
                        #index_device_dict[i].name, index_device_dict[j].name,
                        #index_device_dict[i].attr, index_device_dict[j].attr,))
                        physical_array[i,j,:] = 1

        return physical_array

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