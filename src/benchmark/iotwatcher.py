import collections
import itertools
import numpy as np
import random
import pandas as pd
from src.tigramite.tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from numpy import ndarray
from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
from src.drawer import Drawer
from src.benchmark.association_rule_miner import ARMMiner
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.tigramite.tigramite import plotting as ti_plotting
from collections import defaultdict
from pprint import pprint

from src.tigramite.tigramite import pcmci
from src.tigramite.tigramite.independence_tests.chi2 import ChiSquare

class BackgroundGenerator():

    def __init__(self, event_preprocessor:'GeneralProcessor', frame, tau_max):
        self.event_preprocessor = event_preprocessor
        self.dataset = event_preprocessor.dataset
        self.frame:'DataFrame' = frame
        self.tau_max = tau_max

        # Each background array is of shape (n_vars, n_vars, tau_max+1)
        self.frequency_array, self.normalized_frequency_array = self._temporal_identification()
        self.spatial_array = self._spatial_identification()
        self.user_array = self._user_activity_identification()
        self.physical_array = self._physical_identification()
        self.knowledge_dict = {
            'temporal': self.normalized_frequency_array,
            'spatial': self.spatial_array,
            'user': self.user_array,
            'physical': self.physical_array
        }

    def _temporal_identification(self):
        """
        Many temporal pairs stem from the disorder of IoT logs. Therefore, we filter out pairs with low frequencies.
        The selected pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).
        """
        name_device_dict:'dict[DevAttribute]' = self.frame.name_device_dict
        var_names = self.frame.var_names; n_vars = len(var_names)
        event_sequence:'list[AttrEvent]' = [tup[0] for tup in self.frame.training_events_states]

        #activation_frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)
        #last_act_dev = None; interval = 0
        #for event in event_sequence: # Count sequentially-activated pairs
        #    if event.value == 1:
        #        if last_act_dev and interval <= self.tau_max:
        #            activation_frequency_array[name_device_dict[last_act_dev].index, name_device_dict[event.dev].index, interval] += 1
        #        last_act_dev = event.dev
        #        interval = 1
        #    else:
        #        interval += 1

        frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int32)
        normalized_frequency_array = np.zeros(shape=(n_vars, n_vars, self.tau_max+1), dtype=np.int8)
        for i in range(len(event_sequence)): # Count the occurrence of each lagged attr pair
            for lag in range (1, self.tau_max + 1):
                if i + lag >= len(event_sequence):
                    continue
                former = name_device_dict[event_sequence[i].dev].index; latter = name_device_dict[event_sequence[i + lag].dev].index
                frequency_array[(former, latter, lag)] += 1
        normalized_frequency_array[frequency_array>=self.frame.n_days] = 1

        return frequency_array, normalized_frequency_array
    
    def _testbed_area_information(self):
        # Return variables
        area_list = None
        adjacent_areas_list = None
        area_connectivity_array = None

        if self.dataset == 'hh130':
            adjacent_areas_list = [['Washroom', 'Living'], ['Kitchen', 'Living']]
        elif self.dataset == 'contextact':
            adjacent_areas_list = [['Kitchen', 'Dining Room', 'Living Room', 'Hallway First Floor', 'First Floor', 'Main Entrance'],\
                              ['Hallway First Floor', 'Stairway', 'Hallway Second Floor'],\
                              ['Hallway Second Floor', 'Bathroom', 'Study Room', 'Bedroom']]

        if adjacent_areas_list:
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
        if self.dataset == 'hh130':
            user_attributes = ['Control4-Motion', 'Control4-Door']
        elif self.dataset == 'contextact':
            user_attributes = ['Infrared-Movement-Sensor', 'Switch', 'Dimmer']

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
                'Brightness-Sensor': ['Brightness-Sensor', 'Dimmer', 'Rollershutter'],
                'Dimmer': ['Brightness-Sensor', 'Power Sensor', 'Dimmer'],
                'Contact-Sensor': ['Contact-Sensor', 'Water-Meter', 'Power-Sensor'],
                'Humidity-Sensor': ['Humidity-Sensor', 'Water-Meter', 'Contact-Sensor'],
                'Power-Sensor': ['Power-Sensor', 'Contact-Sensor', 'Rollershutter', 'Water-Meter'],
                'Rollershutter': ['Rollershutter', 'Brightness-Sensor', 'Dimmer', 'Power-Sensor'],
                'Water-Meter': ['Water-Meter', 'Contact-Sensor', 'Power-Sensor'],
                'Switch': [],
                'Infrared-Movement-Sensor': []
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

class HAWatcher():

    def __init__(self, event_preprocessor, frame, tau_max, min_confidence=0.95):
        self.event_preprocessor:'GeneralProcessor' = event_preprocessor
        self.frame = frame
        self.var_names = self.frame.var_names; self.n_vars = len(self.var_names)
        self.background_generator:'BackgroundGenerator' = BackgroundGenerator(event_preprocessor, frame, tau_max)
        #self.association_miner:'AssociationMiner' = association_miner
        self.tau_max = tau_max
        self.min_confidence = min_confidence
        self.ground_truth_dict = self._construct_ground_truth()
        self.rule_dict, self.nor_mining_array = self._rule_mining()
        #self.bayesian_fitter = BayesianFitter(self.frame, self.tau_max, self.mining_edges, n_max_edges=self.n_max_edges, model_name='association-background')
    
    """Helper functions."""

    def _normalize_temporal_array(self, target_array:'np.ndaray', threshold=0):
        new_array = target_array.copy()
        if len(new_array.shape) == 3 and new_array.shape[-1] == self.tau_max+1:
            new_array = sum([new_array[:,:,tau] for tau in range(1, self.tau_max+1)])
            new_array[new_array>threshold] = 1
        return new_array
    
    """Function classes for background knowledge construction."""

    def _construct_ground_truth(self):
        ground_truth_dict = defaultdict(np.ndarray)
        ground_truth_dict['temporal']:'np.ndarray' = self.background_generator.normalized_frequency_array
        ground_truth_dict['spatial']:'np.ndarray' = self.background_generator.knowledge_dict['spatial']
        ground_truth_dict['user']:'np.ndarray' = self.background_generator.knowledge_dict['user']
        ground_truth_dict['physical']:'np.ndarray' = self.background_generator.knowledge_dict['physical']

        return ground_truth_dict

    """Function classes for interaction mining"""

    def _identify_user_interactions(self):
        """
        In the current frame, for any two devices, they have interactions iff (1) they are spatially adjacent, and (2) they are usually sequentially activated.
            (1) The identification of spatial adjacency is done by the background generator.
            (2) The identification of sequential activation is done by checking its occurrence within time lag tau_max.
        """
        # Return variables
        golden_user_array = self.ground_truth_dict['user'] + self.ground_truth_dict['spatial'] + self.ground_truth_dict['temporal']
        golden_user_array[golden_user_array<3] = 0; golden_user_array[golden_user_array==3] = 1

        return golden_user_array

    def _identify_physical_interactions(self):
        # Return variables
        golden_physical_array = self.ground_truth_dict['physical'] + self.ground_truth_dict['spatial'] + self.ground_truth_dict['temporal']
        golden_physical_array[golden_physical_array<3] = 0; golden_physical_array[golden_physical_array==3] = 1

        return golden_physical_array
    
    def _identify_automation_interactions(self):
        # Return variables
        golden_automation_array:'np.ndarray' = np.zeros((self.frame.n_vars, self.frame.n_vars, self.tau_max+1), dtype=np.int32)
        return golden_automation_array
    
    def _rule_mining(self):
        # Construct the ground truth based on HAWathcer's result
        interaction_dict = {}
        interaction_dict['user'] = self._identify_user_interactions(); interaction_dict['physics'] = self._identify_physical_interactions()
        interaction_dict['automation'] = self._identify_automation_interactions()

        # 1. First generate candidate interactions according to the background knowledge
        mining_array:'np.ndarray' = interaction_dict['user'] + interaction_dict['physics'] + interaction_dict['automation']
        mining_array[mining_array>0] = 1
        candidate_interactions:'np.ndarray' = self._normalize_temporal_array(mining_array)

        # 2. Then exmaine these interactions by checking the conditional probability. It is equivlent to an ARM problem with certain confidence about the rule
        # The support is set low because candidate interactions have been selected by the background knowledge.
        rule_dict = {}
        nor_mining_array:'np.ndarray' = np.zeros((self.n_vars, self.n_vars), dtype=np.int8)
        armer = ARMMiner(self.frame, self.tau_max, min_support=0.25, min_confidence=self.min_confidence)
        for c_situ in armer.rule_dict.keys():
            for p_situ in armer.rule_dict[c_situ]:
                # Add the p_situ only if (p_index, c_index) is a candidate interaction
                if candidate_interactions[p_situ[0], c_situ[0]]>0:
                    rule_dict[c_situ] = rule_dict[c_situ] if c_situ in rule_dict.keys() else []
                    rule_dict[c_situ].append(p_situ)
                    nor_mining_array[p_situ[0], c_situ[0]] = 1
        return rule_dict, nor_mining_array

    """Function classes for anomaly detection"""

    def anomaly_detection(self, testing_event_states, testing_benign_dict):
        alarm_position_events = []
        last_system_states = testing_benign_dict[0][1][:self.n_vars]
        for evt_id, (event, states) in enumerate(testing_event_states):
            c_index = self.var_names.index(event.dev)
            c_situ = (c_index, event.value)
            if c_situ not in self.rule_dict.keys():
                # If no rule is found about the current device, it is assumed to be normal.
                last_system_states = states.copy()
                continue
            # Otherwise, we find a list of rules self.rule_dict[c_index], which set the current device as the consecutive device
            p_indices = [p_index[0] for p_index in self.rule_dict[c_situ]]
            p_hypo_values = [p_index[1] for p_index in self.rule_dict[c_situ]]
            p_observed_values = [last_system_states[index] for index in p_indices]
            assert(len(p_hypo_values)==len(p_observed_values))
            if len([i for i in range(len(p_hypo_values)) if p_hypo_values[i]!=p_observed_values[i]])>0:
                # If any rule is violated, raise an alarm here
                alarm_position_events.append((evt_id, event))
                last_system_states = testing_benign_dict[evt_id][1][:self.n_vars]
            else:
                # If all rule testings are passed, it is a normal event, update the lagged system states according to the event
                last_system_states = states.copy()
        return alarm_position_events