from itertools import chain
import numpy as np

NORMAL = 0
TYPE1_ANOMALY = 1
TYPE2_ANOMALY = 2

class PhantomStateMachine():

    def __init__(self, var_names, expanded_var_names) -> None:
        self.var_names = var_names; self.expanded_var_names = expanded_var_names
        self.n_vars = len(var_names); self.n_expanded_vars = len(self.expanded_var_names)
        self.phantom_states = [0] * self.n_expanded_vars

    def set_state(self, current_state_vector):
        assert(len(current_state_vector) == self.n_vars)
        self.phantom_states = [*self.phantom_states[self.n_vars:], *current_state_vector]

    def update(self, event):
        attr = event[0]; state = event[1]
        current_state_vector = self.phantom_states[-1*self.n_vars:].copy()
        current_state_vector[self.var_names.index(attr)] = state
        self.set_state(current_state_vector)

    def get_states(self, expanded_attr_indices, name_flag = 0):
        result_dict = {index: self.phantom_states[index] for index in expanded_attr_indices}\
                        if name_flag == 0 else\
                        {self.expanded_var_names[index]: self.phantom_states[index] for index in expanded_attr_indices}
        return result_dict

class InteractionChain():

    def __init__(self, anomaly_flag, n_vars, expanded_var_names, expanded_causal_graph, expanded_attr_index) -> None:
        self.anomaly_flag = anomaly_flag
        self.n_vars = n_vars; self.expanded_var_names = expanded_var_names; self.expanded_causal_graph = expanded_causal_graph
        self.attr_index_chain = [expanded_attr_index - n_vars] # Adjust to the lagged attribute
        self.header_attr_index = self.attr_index_chain[-1]
    
    def match(self, expanded_attr_index:'int'):
        return self.expanded_causal_graph[self.header_attr_index, expanded_attr_index] > 0
    
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
        self.score_threshold = self._compute_anomaly_score_cutoff(sig_level=sig_level)
        # Anomaly analyzer
        self.breakpoint_dict = {}
        self.violation_dict = {}
        self.type1_debugging_dict = {}
    
    def initialize(self, event_id, event, state_vector):
        self.phantom_state_machine.set_state(state_vector) # Initialize the phantom state machine
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
        breakpoint_flag = self.breakpoint_detection(event)
        anomalous_score_flag, anomaly_score = self.state_validation(event_id=event_id, event=event)
        if self.chain_manager.is_tracking_normal_chain():
            if not anomalous_score_flag: # A normal event
                self.phantom_state_machine.update(event)
                if not breakpoint_flag: # A normal propagation event
                    self.chain_manager.update(expanded_attr_index)
            else: # An abnormal event
                if breakpoint_flag: # Type 1 anomaly
                    self.breakpoint_dict[event_id] = {}
                    self.breakpoint_dict[event_id]['attr'] = attr
                    self.breakpoint_dict[event_id]['anomalous_interaction'] = \
                        (self.last_processed_event[0], attr)
                    self.chain_manager.create(event_id, expanded_attr_index, TYPE1_ANOMALY)
                else: # Type 2 anomaly
                    self.violation_dict[event_id] = {}
                    self.violation_dict[event_id]['attr'] = attr
                    self.violation_dict[event_id]['anomaly-score'] = anomaly_score
                    self.chain_manager.create(event_id, expanded_attr_index, TYPE2_ANOMALY)
        else:
            if breakpoint_flag or self.chain_manager.current_chain_length() >= maximum_length: # The propagation of abnormal chains ends.
                report_to_user = True # Finish tracking the current anomaly chain: Report to users
            else: 
                self.chain_manager.update(expanded_attr_index) # The current chain is still propagating.
        self.last_processed_event = event
        return report_to_user
    
    def calibrate(self, benign_event_id, testing_event_id, selected_event):
        """Find the latest normal event, then
            1. Create a new normal chain starting with the normal event.
            2. Set the state machine to the normal propagations.
        """
        event = (self.frame['testing-attr-sequence'][benign_event_id], self.frame['testing-state-sequence'][benign_event_id])
        assert(event == selected_event)
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        i = self.tau_max
        while i >= 0:
            lagged_benign_state_vector = self.frame['testing-data'].values[benign_event_id - i]
            self.phantom_state_machine.set_state(lagged_benign_state_vector)
            i -= 1
        self.chain_manager.create(testing_event_id, expanded_attr_index, NORMAL)

    def compute_event_anomaly_score(self, event, phantom_state_machine):
        anomaly_score = 0
        attr = event[0]; observed_state = event[1]; expanded_attr_index = self.expanded_var_names.index(attr)
        expanded_parent_indices = self.bayesian_fitter.get_expanded_parent_indices(expanded_attr_index)
        parent_state_dict = phantom_state_machine.get_states(expanded_parent_indices, 1)
        if len(parent_state_dict.keys()) > 0:
            estimated_state = self.bayesian_fitter.predict_attr_state(attr, parent_state_dict)
            anomaly_score = 1.0 * (estimated_state - observed_state)**2
        return anomaly_score

    def _compute_anomaly_score_cutoff(self, sig_level = 0.9):
        computed_anomaly_scores = []
        training_phantom_machine = PhantomStateMachine(self.var_names, self.expanded_var_names)
        training_events = list(zip(self.frame['attr-sequence'], self.frame['state-sequence']))
        training_frame = self.frame['training-data'] # A pp.dataframe object
        assert(len(training_events) == training_frame.T)
        for i in range(training_frame.T):
            cur_event = training_events[i]; cur_state_vector = training_frame.values[i]
            training_phantom_machine.set_state(cur_state_vector)
            if i > self.tau_max:
                anomaly_score = self.compute_event_anomaly_score(cur_event, training_phantom_machine)
                computed_anomaly_scores.append(anomaly_score)
        self.score_threshold = np.percentile(np.array(computed_anomaly_scores), sig_level * 100)
        print("The computed score threshold is {}".format(self.score_threshold))
        return self.score_threshold

    def breakpoint_detection(self, event=()):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        return not self.chain_manager.match(expanded_attr_index)

    def state_validation(self, event_id=-1, event=()):
        violation_flag = False
        anomaly_score = self.compute_event_anomaly_score(event, self.phantom_state_machine)
        if anomaly_score > self.score_threshold:
            violation_flag = True
        return violation_flag, anomaly_score