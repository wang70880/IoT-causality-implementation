from tkinter import W
from numpy import ndarray
import numpy as np
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

    def update(self, event):
        """
        Update the phantom state machine according to the newly received event.
        """
        attr = event[0]; state = event[1]
        renewed_state_vector = self.get_lagged_states(lag = 1).copy()
        renewed_state_vector[self.var_names.index(attr)] = state
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
        self.last_processed_event = ()
        # The parameterized causal graph
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names, bayesian_fitter.expanded_causal_graph)
        # Anomaly analyzer
        self.violation_dict = {}
        self.type1_debugging_dict = {}
        self.fn_debugging_dict = {}
        self.fp_debugging_dict = {}
        self.large_pscore_dict = defaultdict(int)
        self.small_pscore_dict = defaultdict(int)
        # The score threshold
        self.training_anomaly_scores, self.score_threshold = self._compute_anomaly_score_cutoff(sig_level=sig_level)
    
    def initialize(self, event_id, event, state_vector):
        self.phantom_state_machine.set_states(state_vector) # Initialize the phantom state machine
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        if self.chain_manager.match(expanded_attr_index):
            self.chain_manager.update(expanded_attr_index)
        else:
            self.chain_manager.create(event_id, expanded_attr_index, NORMAL)
        self.last_processed_event = event

    def _proceed(self, event):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        if self.chain_manager.match(expanded_attr_index): # Maintain the currently tracked chain
            self.chain_manager.update(expanded_attr_index)
        self.phantom_state_machine.update(event) # Update the phantom state machine

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
                #print("     [Anomaly Tracking] Tracking finished at event {} with len(chain): {}. Prepare to calibrate it.".format(event_id + self.frame['testing-start-index'] + 1, self.chain_manager.current_chain_length()))
                report_to_user = True # Finish tracking the current anomaly chain: Report to users
            else: 
                self.chain_manager.update(expanded_attr_index) # The current chain is still propagating.
        self.last_processed_event = event
        return report_to_user

    def score_anomaly_detection(self, event_id, event, debugging_id_list = []):
        report_to_user = False
        attr = event[0]; state = event[1]
        anomalous_score_flag, anomaly_score = self.state_validation(event=event)
        if not anomalous_score_flag: # A normal event
            self.phantom_state_machine.update(event)
            if event_id in debugging_id_list: # A false negative is detected.
                self.fn_debugging_dict[event_id] = {}
                self.fn_debugging_dict[event_id]['attr'] = attr
                self.fn_debugging_dict[event_id]['state'] = state
                self.fn_debugging_dict[event_id]['anomaly-score'] = anomaly_score
        else: # An abnormal event
            #if event[0] == 'M001': # JC DEBUGGING 
            #    print("[Anomaly Detection] Anomalous score event {}: {}.\n".format(event_id + self.frame['testing-start-index'] + 1, event))
            if event_id not in debugging_id_list:
                self.fp_debugging_dict[event_id] = {}
                self.fp_debugging_dict[event_id]['attr'] = attr
                self.fp_debugging_dict[event_id]['state'] = state
                self.fp_debugging_dict[event_id]['anomaly-score'] = anomaly_score
            report_to_user = True
        self.last_processed_event = event
        return report_to_user
    
    def print_debugging_dict(self, fp_flag=True):
        target_dict = self.fp_debugging_dict if fp_flag else self.fn_debugging_dict
        attr_count_dict = {}; anomaly_score_lists = []; degree_list = []
        for evt_id, anomaly in target_dict.items():
            attr = anomaly['attr']; attr_degree = len(self.bayesian_fitter.get_parents(attr)); state = anomaly['state'];  score = anomaly['anomaly-score']
            print("     * ID, event, in-degree, score = {} ({}, {}) {} {}".format(evt_id, attr, state, attr_degree, score))
            attr_count_dict[attr] = 1 if attr not in attr_count_dict.keys() else attr_count_dict[attr] + 1
            anomaly_score_lists.append(score); degree_list.append(attr_degree)
        pprint(attr_count_dict)
        avg_score = 0 if len(anomaly_score_lists) == 0 else sum(anomaly_score_lists) * 1.0 / len(anomaly_score_lists)
        avg_degree = 0 if len(degree_list) == 0 else sum(degree_list) * 1.0 / len(degree_list)
        print("**Anomaly scores**:\n{}\nAverage: {}".format(anomaly_score_lists, avg_score))
        print("**Attr degrees**:\n{}\nAverage: {}".format(degree_list, avg_degree))

    def calibrate(self, event_id, stable_states_dict):
        """
        Find the latest normal event, then
            1. Create a new normal chain starting with the normal event.
            2. Set the state machine to the normal propagations.
        
        Param:
            stable_states_dict: The {event_id: (stable event, stable state vector)} dict
        """
        #print("     [Calibration] The nearest benign event is {}: {}".format(testing_event_id + self.frame['testing-start-index'] + 1, event))
        i = 0
        while i < self.tau_max:
            lagged_benign_state_vector = stable_states_dict[event_id - i][1]
            self.phantom_state_machine.set_states(lagged_benign_state_vector)
            i += 1
        last_stable_attr = stable_states_dict[event_id][0][0]; expanded_attr_index = self.expanded_var_names.index(last_stable_attr)
        #print(self.phantom_state_machine)
        self.chain_manager.create(event_id, expanded_attr_index, NORMAL)

    def compute_event_anomaly_score(self, event:'AttrEvent', phantom_state_machine:'PhantomStateMachine'):
        # Return variables
        anomaly_score = 0.
        # Auxillary variables
        extended_name_device_dict:'dict[str, DevAttribute]' = self.bayesian_fitter.extended_name_device_dict
        # 1. Get the list of parents
        expanded_parents:'list[DevAttribute]' = self.bayesian_fitter.get_expanded_parents(extended_name_device_dict[event.dev])
        if self.verbosity > 0:
            print("[Score Computation] Now handling device {} with parents ({})".format(event.dev,\
                    ','.join([parent.name for parent in expanded_parents])))
        # 2. Fetch the parents' states
        parent_state_dict:'dict[DevAttribute, int]' = phantom_state_machine.get_device_states(expanded_parents)
        device_states:'list[str, int]' = [(k.name, v) for k, v in parent_state_dict.items()]
        device_states.append((event.dev, event.value))
        # 3. Estimate the anomaly score for current event
        cond_prob = self.bayesian_fitter.estimate_cond_probability(event, device_states)
        anomaly_score = 1 - cond_prob
        if anomaly_score >= 0.9:
            self.large_pscore_dict['{}={} under {}'.format(event.dev, event.value, ",".join(['{}={}'.format(k.name, v) for k, v in parent_state_dict.items()]))] += 1
        else:
            self.small_pscore_dict['{}:{}'.format(event.dev, event.value)] += 1
        return anomaly_score

    def _compute_anomaly_score_cutoff(self, sig_level = 0.9):
        # Return variables
        self.score_threshold = 0.
        # Auxillary variables
        var_names = self.bayesian_fitter.var_names; expanded_var_names = self.bayesian_fitter.expanded_var_names
        tau_max = self.bayesian_fitter.tau_max
        training_events_states:'list[tuple(AttrEvent, ndarray)]' = self.frame.training_events_states

        anomaly_scores = []
        training_phantom_machine = PhantomStateMachine(var_names, expanded_var_names)
        for index, (event, states) in enumerate(training_events_states):
            training_phantom_machine.set_states(states)
            if index > tau_max:
                anomaly_score = self.compute_event_anomaly_score(event, training_phantom_machine)
                anomaly_scores.append(anomaly_score)
        score_threshold = np.percentile(np.array(anomaly_scores), sig_level * 100)
        return anomaly_scores, score_threshold

    def breakpoint_detection(self, event=()):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        return not self.chain_manager.match(expanded_attr_index)

    def state_validation(self, event=()):
        violation_flag = False
        anomaly_score = self.compute_event_anomaly_score(event, self.phantom_state_machine)
        if anomaly_score > self.score_threshold:
            violation_flag = True
        return violation_flag, anomaly_score