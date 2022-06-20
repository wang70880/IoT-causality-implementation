from itertools import chain
import numpy as np
from pprint import pprint
from tabulate import tabulate

NORMAL = 0
TYPE1_ANOMALY = 1
TYPE2_ANOMALY = 2

class PhantomStateMachine():
    """
    The phantom state machine stores the history device states (with respect to the time lag)
    That is, suppose there is n attributes and time lag is l, the length of phantom_states is n * l
    """

    def __init__(self, var_names, expanded_var_names) -> None:
        self.var_names = var_names; self.expanded_var_names = expanded_var_names
        self.n_vars = len(var_names); self.n_expanded_vars = len(self.expanded_var_names)
        self.phantom_states = [0] * (self.n_expanded_vars - self.n_vars) # The phantom state machine stores devices' history states.

    def set_states(self, state_vector):
        """
        Update the latest history state vectors (i.e., the vector for lag = -1).

        Args:
            state_vector (list[int]): The state vector to be renewed.
        """
        assert(len(state_vector) == self.n_vars)
        # Push the state machine backwards
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

    def get_attr_states(self, expanded_attr_indices, name_type = False):
        """
        Fetch the state of specified lagged attributes

        Args:
            expanded_attr_indices (list[int]): The attribute indices which are of interest
            name_type (bool, optional): Whether the key of returned state dict is the attribute name (Otherwise it is the index). Defaults to False.

        Returns:
            result_dict (dict[int:int] or dict[str:int]): The name-state dict
        """
        assert(all([x < self.n_expanded_vars - self.n_vars for x in expanded_attr_indices])) # Since phantom state machine does not store the current state, the function cannot help to fetch the current state.
        result_dict = {index: self.phantom_states[index] for index in expanded_attr_indices} if name_type else \
                            {self.expanded_var_names[index]: self.phantom_states[index] for index in expanded_attr_indices}
        return result_dict
    
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

    def __init__(self, frame=None, bayesian_fitter=None, verbosity=0, sig_level = 0.9) -> None:
        self.frame = frame
        self.verbosity = verbosity
        self.var_names: 'list[str]' = bayesian_fitter.var_names; self.expanded_var_names: 'list[str]' = bayesian_fitter.expanded_var_names
        self.tau_max = bayesian_fitter.tau_max
        self.last_processed_event = ()
        # The parameterized causal graph
        self.bayesian_fitter = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names, bayesian_fitter.expanded_causal_graph)
        # The score threshold
        self.score_threshold = 0.0
        self._compute_anomaly_score_cutoff(sig_level=sig_level)
        # Anomaly analyzer
        self.violation_dict = {}
        self.type1_debugging_dict = {}
    
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

    def score_anomaly_detection(self, event_id, event):
        report_to_user = False
        attr = event[0]
        anomalous_score_flag, anomaly_score = self.state_validation(event=event)
        if not anomalous_score_flag: # A normal event
            self.phantom_state_machine.update(event)
        else: # An abnormal event
            #if event[0] == 'M001': # JC DEBUGGING 
            #    print("[Anomaly Detection] Anomalous score event {}: {}.\n".format(event_id + self.frame['testing-start-index'] + 1, event))
            self.violation_dict[event_id] = {}
            self.violation_dict[event_id]['attr'] = attr
            self.violation_dict[event_id]['anomaly-score'] = anomaly_score
            report_to_user = True
        self.last_processed_event = event
        return report_to_user

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

    def compute_event_anomaly_score(self, event, phantom_state_machine):
        anomaly_score = 0
        attr = event[0]; observed_state = event[1]; expanded_attr_index = self.expanded_var_names.index(attr)
        expanded_parent_indices = self.bayesian_fitter.get_expanded_parent_indices(expanded_attr_index)
        #print(" [Score Computation] Now handling attr {} with parents ({})".format(attr, ','.join([self.expanded_var_names[i] for i in expanded_parent_indices])))
        parent_state_dict = phantom_state_machine.get_attr_states(expanded_parent_indices, True)
        if attr not in self.bayesian_fitter.nointeraction_attr_list: # JC NOTE: For all attributes which have no interactions or only aturocorrelations, they are not our focus and we assume they are benign.
            estimated_state = self.bayesian_fitter.predict_attr_state(attr, parent_state_dict)
            anomaly_score = 1.0 * (estimated_state - observed_state)**2
            if self.score_threshold > 0 and anomaly_score > self.score_threshold and event[0] == 'M001' and False: # JC DEBUGGING
                print("[Anomaly Detection] Event {} is an anomaly!".format(event))
                pprint(parent_state_dict)
                print("     (Estimated state, Observed state) = ({}, {})".format(estimated_state, observed_state))
                print("     (Abnormal score > threshold) = ({} > {})".format(anomaly_score, self.score_threshold))
        return anomaly_score

    def _compute_anomaly_score_cutoff(self, sig_level = 0.9):
        computed_anomaly_scores = []
        training_phantom_machine = PhantomStateMachine(self.var_names, self.expanded_var_names)
        training_events = list(zip(self.frame['attr-sequence'], self.frame['state-sequence']))
        training_frame = self.frame['training-data'] # A pp.dataframe object
        assert(len(training_events) == training_frame.T)
        for i in range(training_frame.T):
            cur_event = training_events[i]; cur_state_vector = training_frame.values[i]
            training_phantom_machine.set_states(cur_state_vector)
            if i > self.tau_max:
                anomaly_score = self.compute_event_anomaly_score(cur_event, training_phantom_machine)
                computed_anomaly_scores.append(anomaly_score)
        self.score_threshold = np.percentile(np.array(computed_anomaly_scores), sig_level * 100)
        return self.score_threshold

    def breakpoint_detection(self, event=()):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        return not self.chain_manager.match(expanded_attr_index)

    def state_validation(self, event=()):
        violation_flag = False
        anomaly_score = self.compute_event_anomaly_score(event, self.phantom_state_machine)
        if anomaly_score > self.score_threshold:
            violation_flag = True
        return violation_flag, anomaly_score