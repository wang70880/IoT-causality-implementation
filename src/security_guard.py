from itertools import chain
import numpy as np

NORMAL = 0
ABNORMAL = 1
NORMAL_EXO = 2
NORMAL_ENO = 3
ABNORMAL_EXO = 4
ABNORMAL_ENO = 5
anomaly_flag_dict = {NORMAL: [NORMAL_EXO, NORMAL_ENO], ABNORMAL: [ABNORMAL_EXO, ABNORMAL_ENO]}

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

    def get_states(self, attr_list):
        return {attr: self.phantom_states[self.expanded_var_names.index(attr)] for attr in attr_list}

class InteractionChain():

    def __init__(self, n_vars, expanded_var_names, expanded_causal_graph, expanded_attr_index) -> None:
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
    
    def __str__(self):
        return "header_attr = {}, len(chains) = {}\n"\
              .format(self.expanded_var_names[self.header_attr_index], len(self.attr_index_chain))

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
        self.current_chain.update(expanded_attr_index)
    
    def create(self, evt_id:'int', expanded_attr_index:'int'):
        chain_id = evt_id
        self.chain_pool[chain_id] = InteractionChain(self.n_vars, self.expanded_var_names,\
                                    self.expanded_causal_graph, expanded_attr_index)
        self.n_chains += 1
        self.current_chain = self.chain_pool[chain_id]
        return chain_id

    def print_chains(self):
        print("Current chain stack with {} chains.".format(len(self.chain_pool.keys())))
        for index, chain in enumerate(self.chain_pool):
            print(" * Chain {}: {}".format(index, chain))

class SecurityGuard():

    def __init__(self, bayesian_fitter=None, verbosity=0) -> None:
        self.verbosity = verbosity
        self.var_names: 'list[str]' = bayesian_fitter.var_names
        self.expanded_var_names: 'list[str]' = bayesian_fitter.expanded_var_names
        self.last_processed_event = ()
        # The score threshold
        self.score_threshold = 0.0
        # The parameterized causal graph
        self.bayesian_fitter = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names, bayesian_fitter.expanded_causal_graph)
        # Anomaly analyzer
        self.breakpoint_dict = {}
        self.type1_anomaly_dict = {}
        self.type1_debugging_dict = {}
    
    def initialize(self, event_id, event, state_vector):
        self.phantom_state_machine.set_state(state_vector) # Initialize the phantom state machine
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        if self.chain_manager.match(expanded_attr_index):
            self.chain_manager.update(expanded_attr_index)
        else:
            self.chain_manager.create(event_id, expanded_attr_index)
        self.last_processed_event = event

    def detect_type1_anomaly(self, event_id=0, event=(), debugging_list=[]):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        if self.chain_manager.match(expanded_attr_index):
            self.chain_manager.update(expanded_attr_index)
        else: # A breakpoint is detected
            chain_id = self.chain_manager.create(event_id, expanded_attr_index)
            # Save information about the breakpoint
            self.breakpoint_dict[event_id] = {}
            self.breakpoint_dict[event_id]['attr'] = attr
            self.breakpoint_dict[event_id]['anomalous_interaction'] = \
                (self.var_names.index(self.last_processed_event[0]), self.var_names.index(attr))
            self.breakpoint_dict[event_id]['chain_id'] = chain_id
            # Create a new chain in ChainManager
            print("Anomalous interactions in chain {}: {}".format(self.breakpoint_dict[event_id]['chain_id'],\
                                        (self.last_processed_event[0], attr)))
        self.last_processed_event = event

    def _compute_anomaly_score(self, state, predicted_state):
        # Here we take the quadratic loss as the anomaly score.
        # See paper "A Unifying Framework for Detecting Outliers and Change Points from Time Series"
        return 1.0 * (state - predicted_state) ** 2
    
    def anomaly_detection(self, event_id=0, event=(), debugging_list=[]):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        expanded_parent_indices = self.bayesian_fitter.get_expanded_parent_indices(expanded_attr_index)
        anomaly_flag = NORMAL; exo_flag = len(expanded_parent_indices) == 0
        # First initiate detections of type-1 attacks.
        if (not exo_flag) and (len(self.chain_manager.match(expanded_attr_index, NORMAL)) == 0):
            anomaly_flag = ABNORMAL
        
        if self.verbosity > 0:
            self.chain_manager.print_chains()
            if anomaly_flag == ABNORMAL:
                str = "Type-1 anomalies are detected!\n"\
                        + "  * Current event: {}\n".format(event)\
                        + "  * Exogenous attribute: {}\n".format(exo_flag)\
                        + "  * The parent set: {}\n".format([self.expanded_var_names[i] for i in expanded_parent_indices])
                print(str)

        # JC TODO: Initiate detections of type-2 attacks.
        # parent_state_dict = self.phantom_state_machine.get_states(expanded_parent_list)
        # predicted_state =  self.bayesian_fitter.predict_attr_state(attr, parent_state_dict)
        # anomaly_score = self._compute_anomaly_score(state, predicted_state)
        # anomaly_flag = anomaly_score > threshold

        # Update the chain pool and the phantom state machine according to the detection result.
        detailed_anomaly_flag, affected_chain_ids = self.chain_manager.update(expanded_attr_index, anomaly_flag)
        if detailed_anomaly_flag == ABNORMAL_EXO: # A type-1 anomaly is detected.
            self.type1_anomaly_dict[event_id] = {}
            self.type1_anomaly_dict[event_id]['anomalous_interaction'] = (self.var_names.index(self.last_processed_event[0]), self.var_names.index(event[0]))
            self.type1_anomaly_dict[event_id]['chain-id'] = affected_chain_ids[0]
        #self.phantom_state_machine.update(event)
        self.last_processed_event = event
        if len(debugging_list) > 0 and event_id in debugging_list:
            self.type1_debugging_dict[event_id] = detailed_anomaly_flag
        return detailed_anomaly_flag