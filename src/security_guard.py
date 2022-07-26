from tabulate import tabulate
from numpy import ndarray
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from statistics import mean
from collections import defaultdict

from src.genetic_type import DevAttribute, AttrEvent, DataFrame
from src.bayesian_fitter import BayesianFitter

NORMAL = 0
TYPE1_ANOMALY = 1
TYPE2_ANOMALY = 2

class PhantomStateMachine():

    def __init__(self, var_names, expanded_var_names) -> None:
        self.var_names = var_names; self.expanded_var_names = expanded_var_names
        self.n_vars = len(var_names); self.n_expanded_vars = len(self.expanded_var_names)
        assert(self.n_expanded_vars > self.n_vars and self.n_expanded_vars % self.n_vars == 0)
        self.phantom_states = [0] * (self.n_expanded_vars - self.n_vars) # suppose there is n attributes and time lag is l, the length of phantom_states is n * l

    def set_states(self, state_vector:'list[int]'):
        """
        Update the latest history state vectors (i.e., the vector for lag = -1).
        Args:
            state_vector (list[int]): The state vector to be renewed.
        """
        assert(len(state_vector) == self.n_vars)
        self.phantom_states = [*self.phantom_states[self.n_vars:], *state_vector]
    
    def get_lagged_states(self, lag = 1):
        """
        Get the history device states with respect to the time lag.

        Args:
            lag (int, optional): The time lag. Defaults to 1.

        Returns:
            sliced_phantom_states: The sliced phantom states (Not copied version)
        """
        assert(lag > 0)
        if lag == 1:
            return self.phantom_states[-lag * self.n_vars:]
        else:
            return self.phantom_states[-lag * self.n_vars: -(lag-1) * self.n_vars]

    def update(self, event:'AttrEvent'):
        """
        Update the phantom state machine according to the newly received event.
        """
        renewed_state_vector = self.get_lagged_states(lag = 1).copy()
        renewed_state_vector[self.var_names.index(event.dev)] = event.value
        self.set_states(renewed_state_vector)

    def get_device_states(self, expanded_devices:'list[DevAttribute]'):
        assert(all([device.index < self.n_expanded_vars - self.n_vars for device in expanded_devices])) # Since phantom state machine does not store the current state, the function cannot help to fetch the current state.
        return {device: self.phantom_states[device.index] for device in expanded_devices}
    
    def __str__(self):
        return tabulate(\
            list(zip(self.expanded_var_names[0: self.n_expanded_vars - self.n_vars], self.phantom_states)),\
            headers= ['Attr', 'State'])

class InteractionChain():

    def __init__(self, anomaly_flag, n_vars, expanded_var_names, expanded_causal_graph, expanded_attr_index) -> None:
        self.anomaly_flag = anomaly_flag
        self.n_vars = n_vars; self.expanded_var_names = expanded_var_names; self.expanded_causal_graph = expanded_causal_graph
        self.attr_index_chain = [expanded_attr_index - n_vars] # Adjust to the lagged attribute
        self.header_attr_index = self.attr_index_chain[-1]
    
    def match(self, expanded_attr_index:'int'):
        return self.expanded_causal_graph[self.header_attr_index, expanded_attr_index] > 0
    
    def get_header_attr(self):
        return self.expanded_var_names[self.header_attr_index]
    
    def update(self, expanded_attr_index:'int'):
        assert(self.match(expanded_attr_index))
        self.attr_index_chain.append(expanded_attr_index)
        self.attr_index_chain = [x - self.n_vars for x in self.attr_index_chain]
        self.header_attr_index = self.attr_index_chain[-1]
    
    def get_length(self):
        return len(self.attr_index_chain)

    def __str__(self):
        return "header_attr = {}, anomaly_flag = {}, len(chains) = {}\n"\
              .format(self.expanded_var_names[self.header_attr_index], self.anomaly_flag, len(self.attr_index_chain))

class ChainManager():
    
    def __init__(self, var_names, expanded_var_names, expanded_causal_graph) -> None:
        self.chain_pool = {}; self.n_chains = 0
        self.current_chain:'InteractionChain' = None
        self.var_names = var_names; self.n_vars = len(self.var_names)
        self.expanded_var_names = expanded_var_names; self.n_expanded_vars = len(self.expanded_var_names)
        self.expanded_causal_graph = expanded_causal_graph
        self.anomalous_interaction_dict = {}
    
    def match(self, expanded_attr_index:'int'):
        return self.current_chain is not None and self.current_chain.match(expanded_attr_index)
    
    def update(self, expanded_attr_index:'int'):
        assert(self.match(expanded_attr_index))
        self.current_chain.update(expanded_attr_index)
    
    def create(self, evt_id:'int', expanded_attr_index:'int', anomaly_flag):
        chain_id = evt_id
        self.chain_pool[chain_id] = InteractionChain(anomaly_flag, self.n_vars, self.expanded_var_names,\
                                    self.expanded_causal_graph, expanded_attr_index)
        self.n_chains += 1
        self.current_chain = self.chain_pool[chain_id]
        return chain_id

    def print_chains(self):
        print("Current chain stack with {} chains.".format(len(self.chain_pool.keys())))
        for index, chain in enumerate(self.chain_pool):
            print(" * Chain {}: {}".format(index, chain))

    def is_tracking_normal_chain(self):
        return self.current_chain.anomaly_flag == NORMAL

    def current_chain_length(self):
        return self.current_chain.get_length()

class SecurityGuard():

    def __init__(self, frame=None, bayesian_fitter:'BayesianFitter'=None, sig_level=0.9, verbosity=0) -> None:
        self.frame:'DataFrame' = frame
        self.verbosity = verbosity
        # The parameterized causal graph
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names, bayesian_fitter.expanded_causal_graph)
        # Recent devices
        self.recent_devices = []
        # Anomaly analyzer
        self.violation_dict = {}
        self.type1_debugging_dict = {}
        self.tp_debugging_dict = {}
        self.fn_debugging_dict = {}
        self.fp_debugging_dict = {}
        # The score threshold
        self.training_anomaly_scores, self.score_threshold = self._compute_anomaly_score_cutoff(sig_level=sig_level)
    
    def initialize(self, event_id:'int', event:'AttrEvent', state_vector:'np.ndarray'):
        # Auxillary variables
        extended_name_device_dict = self.bayesian_fitter.extended_name_device_dict
        # 1. Initialize the phantom state machine
        self.phantom_state_machine.set_states(state_vector)
        # 2. Update the chain manager
        index = extended_name_device_dict[event.dev].index
        if self.chain_manager.match(index):
            self.chain_manager.update(index)
        else:
            self.chain_manager.create(event_id, index, NORMAL)
        # 3. Update recent-devices list
        self._update_recent_devices(event.dev, True)
    
    def _update_recent_devices(self, dev, normality=True):
        # Auxillary variables
        tau_max = self.bayesian_fitter.tau_max
        if normality:
            if len(self.recent_devices) < tau_max:
                self.recent_devices.append(dev)
            else:
                self.recent_devices = [*self.recent_devices[1:], dev]

    def anomaly_detection(self, event_id, event, maximum_length):
        report_to_user = False
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        #print(self.phantom_state_machine)
        breakpoint_flag = self.breakpoint_detection(event)
        anomalous_score_flag, anomaly_score = self.state_validation(event=event)
        #print(" [Score Computation] The anomaly flag, score for {} becoming {} is ({}, {})".format(event[0], event[1], anomalous_score_flag, anomaly_score))
        if self.chain_manager.is_tracking_normal_chain():
            if not anomalous_score_flag: # A normal event
                self.phantom_state_machine.update(event)
                if not breakpoint_flag: # A normal propagation event
                    self.chain_manager.update(expanded_attr_index)
            else: # An abnormal event
                #print("[Anomaly Detection] Event {}: {}. Tracked chain's normality: {}.".format(event_id + self.frame['testing-start-index'] + 1, event, self.chain_manager.is_tracking_normal_chain()))
                self.violation_dict[event_id] = {}
                self.violation_dict[event_id]['attr'] = attr
                self.violation_dict[event_id]['interaction'] = (self.chain_manager.current_chain.get_header_attr(), attr)
                self.violation_dict[event_id]['breakpoint-flag'] = breakpoint_flag
                self.violation_dict[event_id]['anomaly-score'] = anomaly_score
                self.chain_manager.create(event_id, expanded_attr_index, TYPE2_ANOMALY)
        else:
            if breakpoint_flag or self.chain_manager.current_chain_length() >= maximum_length: # The propagation of abnormal chains ends.
                report_to_user = True # Finish tracking the current anomaly chain: Report to users
            else: 
                self.chain_manager.update(expanded_attr_index) # The current chain is still propagating.
        self.last_processed_event = event
        return report_to_user

    def score_anomaly_detection(self, event_id, event:'AttrEvent', debugging_id_list = []):
        parent_states, anomaly_score = self._compute_event_anomaly_score(event, self.phantom_state_machine, self.recent_devices)
        anomalous_score_flag = True if anomaly_score > self.score_threshold else False

        int_dict = None
        if anomalous_score_flag and event_id in debugging_id_list: # A tp is detected
            int_dict = self.tp_debugging_dict
        elif not anomalous_score_flag and event_id in debugging_id_list: # A fn is detected
            int_dict = self.fn_debugging_dict
        elif anomalous_score_flag and event_id not in debugging_id_list: # A fp is detected
            int_dict = self.fp_debugging_dict
        if int_dict is not None:
            anomaly_case = '{}={} under {}'.format(event.dev, event.value, ",".join(['{}={}'.format(k, v) for (k, v) in parent_states]))
            int_dict[anomaly_case] = [] if anomaly_case not in int_dict.keys() else int_dict[anomaly_case]
            int_dict[anomaly_case].append((event_id, anomaly_score))
        return anomalous_score_flag 
    
    def analyze_detection_results(self):
        tps = sum([len(tp_list) for tp_list in self.tp_debugging_dict.values()]); fps = sum([len(fp_list) for fp_list in self.fp_debugging_dict.values()]); fns = sum([len(fn_list) for fn_list in self.fn_debugging_dict.values()])
        sorted_tps = dict(sorted(self.tp_debugging_dict.items(), key=lambda x: len(x[1]), reverse=True))
        sorted_fps = dict(sorted(self.fp_debugging_dict.items(), key=lambda x: len(x[1]), reverse=True))
        sorted_fns = dict(sorted(self.fn_debugging_dict.items(), key=lambda x: len(x[1]), reverse=True))
        #print("************* # of FPs = {} *************".format(fps))
        #for fp_case in sorted_fps.keys():
        #    n_cases = len(sorted_fps[fp_case])
        #    if n_cases > 10:
        #        print("{} ---- Occurrence: {}, Avg score: {}".format(fp_case, n_cases, mean([sorted_fps[fp_case][x][1] for x in range(n_cases)])))
        #print("************* # of FNs = {} *************".format(fns))
        #for fn_case in sorted_fns.keys():
        #    n_cases = len(sorted_fns[fn_case])
        #    if n_cases > 10:
        #        print("{} ---- Occurrence: {}, Avg score: {}".format(fn_case, n_cases, mean([sorted_fns[fn_case][x][1] for x in range(n_cases)])))

        print("************* # of TPs = {} *************".format(tps))
        for tp_case in sorted_tps.keys():
            n_cases = len(sorted_tps[tp_case])
            #if n_cases > 10:
            print("{} ---- Occurrence: {}, Avg score: {}".format(tp_case, n_cases, mean([sorted_tps[tp_case][x][1] for x in range(n_cases)])))
        return fps, fns

    def calibrate(self, event_id:'int', testing_benign_dict:'dict[int]'):
        # Auxillary variables
        benign_event_states = self.frame.testing_events_states
        tau_max = self.bayesian_fitter.tau_max
        extended_name_device_dict = self.bayesian_fitter.extended_name_device_dict
        # Use the nearest tau_max benign events to calibrate the (1) phantom state machine, (2) recent_devices list, and (3) the currently tracking chain
        self.recent_devices = []
        benign_starting_index = testing_benign_dict[event_id]
        for i in reversed(range(tau_max)):
            lagged_event, lagged_benign_state_vector = benign_event_states[benign_starting_index - i]
            self.phantom_state_machine.set_states(lagged_benign_state_vector)
            self._update_recent_devices(lagged_event.dev, True)
            if i == 0:
                self.chain_manager.create(event_id, extended_name_device_dict[lagged_event.dev].index, NORMAL)

    def _fetch_parent_states(self, event:'AttrEvent', phantom_state_machine:'PhantomStateMachine'):
        # Auxillary variables
        extended_name_device_dict:'dict[str, DevAttribute]' = self.bayesian_fitter.extended_name_device_dict
        # 1. Get the list of parents
        expanded_parents:'list[DevAttribute]' = self.bayesian_fitter.get_expanded_parents(extended_name_device_dict[event.dev])
        # 2. Fetch the parents' states
        parent_state_dict:'dict[DevAttribute, int]' = phantom_state_machine.get_device_states(expanded_parents)
        parent_states:'list[tuple(str, int)]' = [(k.name, v) for k, v in parent_state_dict.items()]
        return parent_states

    def _compute_event_anomaly_score(self, event:'AttrEvent', phantom_state_machine:'PhantomStateMachine', recent_devices:'list[str]', verbosity=0):
        # Return variables
        anomaly_score = 0.
        # 1. Get the list of parents for event.dev, and fetch their states from the phantom state machine
        parent_states:'list[tuple(str, int)]' = self._fetch_parent_states(event, self.phantom_state_machine)
        # 2. Estimate the anomaly score for current event
        cond_prob = self.bayesian_fitter.estimate_cond_probability(event, parent_states, recent_devices)
        anomaly_score = 1. - cond_prob
        if verbosity:
            print("Event: {}\n\
                   Phantom state machine: {}\n\
                   recent devices: {}\n\
                   parent states: {}\n\
                   anomaly score: {}\n".format(event, self.phantom_state_machine, recent_devices, parent_states, cond_prob)
                  )
        return parent_states, anomaly_score

    def _compute_anomaly_score_cutoff(self, sig_level = 0.9):
        # Return variables
        self.score_threshold = 0.
        anomaly_scores = []
        # Auxillary variables
        var_names = self.bayesian_fitter.var_names; expanded_var_names = self.bayesian_fitter.expanded_var_names
        tau_max = self.bayesian_fitter.tau_max
        training_events_states:'list[tuple(AttrEvent, ndarray)]' = self.frame.training_events_states
        
        recent_devices = []
        training_phantom_machine = PhantomStateMachine(var_names, expanded_var_names)
        for index, (event, states) in enumerate(training_events_states):
            training_phantom_machine.set_states(states)
            if index > tau_max:
                parent_states, anomaly_score = self._compute_event_anomaly_score(event, training_phantom_machine, recent_devices)
                anomaly_scores.append(anomaly_score)
            if len(recent_devices) < tau_max:
                recent_devices.append(event.dev)
            else:
                recent_devices = [*recent_devices[1:], event.dev]
        score_threshold = np.percentile(np.array(anomaly_scores), sig_level * 100)
        # Draw the score distribution graph
        sns.displot(anomaly_scores, kde=False, color='red', bins=1000)
        plt.axvline(x=score_threshold)
        plt.title('Training score distribution')
        plt.xlabel('Scores')
        plt.ylabel('Occurrences')
        plt.savefig("temp/image/training-score-distribution.pdf")
        plt.close('all')
        return anomaly_scores, score_threshold

    def breakpoint_detection(self, event=()):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        return not self.chain_manager.match(expanded_attr_index)