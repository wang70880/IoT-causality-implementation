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

    def __init__(self, n_vars, expanded_var_names, expanded_causal_graph, anomaly_flag, expanded_attr_index) -> None:
        self.n_vars = n_vars; self.expanded_var_names = expanded_var_names; self.expanded_causal_graph = expanded_causal_graph
        self.anomaly_flag = anomaly_flag
        self.attr_index_chain = [expanded_attr_index]
        self.header_attr_index = self.attr_index_chain[-1]
    
    def match(self, expanded_attr_index:'int'):
        return self.expanded_causal_graph[self.header_attr_index, expanded_attr_index] > 0
    
    def update(self, expanded_attr_index:'int'):
        assert(self.match(expanded_attr_index))
        self.attr_index_chain.append(expanded_attr_index)
        self.attr_index_chain = [x - self.n_vars for x in self.attr_index_chain]
        self.header_attr_index = self.attr_index_chain[-1]
    
    def __str__(self):
        return "anomaly_flag = {}, header_attr = {}, len(chains) = {}\n"\
              .format(self.anomaly_flag, self.expanded_var_names[self.header_attr_index], len(self.attr_index_chain))

class ChainManager():
    
    def __init__(self, var_names, expanded_var_names, expanded_causal_graph) -> None:
        self.chain_pool:'list[InteractionChain]' = []
        self.var_names = var_names; self.n_vars = len(self.var_names)
        self.expanded_var_names = expanded_var_names; self.n_expanded_vars = len(self.expanded_var_names)
        self.expanded_causal_graph = expanded_causal_graph
    
    def match(self, expanded_attr_index:'int', anomaly_flag=-1):
        """Identify the set of normal/abnormal chains which can accommodate the new attribute
        That is, there exists a lagged variable in the candidate chain which is the parent of the new attribute.

        Args:
            expanded_attr_index (int): The new attribute

        Returns:
            matched_chain_indices: The list of indices for satisfied chains (in the chain pool)
        """
        satisfied_chain_indices = []
        if anomaly_flag == NORMAL:
            satisfied_chain_indices = [self.chain_pool.index(chain) for chain in self.chain_pool if chain.anomaly_flag == NORMAL and chain.match(expanded_attr_index)]
        elif anomaly_flag == ABNORMAL:
            satisfied_chain_indices = [self.chain_pool.index(chain) for chain in self.chain_pool if chain.anomaly_flag == ABNORMAL and chain.match(expanded_attr_index)]
        else:
            satisfied_chain_indices = [self.chain_pool.index(chain) for chain in self.chain_pool if chain.match(expanded_attr_index)]
        return satisfied_chain_indices
    
    def update(self, expanded_attr_index:'int', anomaly_flag):
        """Update the existing chains (or create a new chain) given the new arrived attribute event and the anomaly flag.

        Args:
            expanded_attr_index (int): The new attribute
            n_parents (_type_): Number of the new attributes' parents

        Returns:
            int: The number of updated chains
        """
        n_affected_chains = 0; detailed_anomaly_flag = 0
        matched_chain_indices = self.match(expanded_attr_index, anomaly_flag)
        if len(matched_chain_indices) > 0:
            n_affected_chains = len(matched_chain_indices); detailed_anomaly_flag = anomaly_flag_dict[anomaly_flag][1]
            for chain_index in matched_chain_indices:
                self.chain_pool[chain_index].update(expanded_attr_index)
        else:
            n_affected_chains = 1; detailed_anomaly_flag = anomaly_flag_dict[anomaly_flag][0]
            lagged_attr_index = expanded_attr_index - self.n_vars
            self.chain_pool.append(InteractionChain(self.n_vars, self.expanded_var_names, self.expanded_causal_graph, anomaly_flag, lagged_attr_index))
         
        return n_affected_chains, detailed_anomaly_flag

    def print_chains(self):
        normal_chains = [chain for chain in self.chain_pool if chain.anomaly_flag == NORMAL]
        abnormal_chains = [chain for chain in self.chain_pool if chain.anomaly_flag == ABNORMAL]
        print("Current chain stack with {} normal chains and {} abnormal chains".format(len(normal_chains), len(abnormal_chains)))
        for index, chain in enumerate(self.chain_pool):
            print(" * Chain {}: {}".format(index, chain))

class SecurityGuard():

    def __init__(self, bayesian_fitter, verbosity, sig_level) -> None:
        self.verbosity = verbosity
        self.var_names: 'list[str]' = bayesian_fitter.var_names
        self.expanded_var_names: 'list[str]' = bayesian_fitter.expanded_var_names
        self.last_processed_event = ()
        # The score threshold
        self.sig_level = sig_level
        self.score_threshold = 0.0
        # The parameterized causal graph
        self.bayesian_fitter = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names, bayesian_fitter.expanded_causal_graph)
        # Anomaly analyzer
        self.anomalous_interaction_dict = {}
    
    def get_score_threshold(self, training_frame):
        # JC TODO: Estimate the score threshold given the significance level (self.sig_level)
        pass
    
    def initialize(self, event, state_vector):
        self.phantom_state_machine.set_state(state_vector) # Initialize the phantom state machine
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        self.chain_manager.update(expanded_attr_index, NORMAL) # Initialize the chain manager
        self.last_processed_event = event
    
    def _compute_anomaly_score(self, state, predicted_state):
        # Here we take the quadratic loss as the anomaly score.
        # See paper "A Unifying Framework for Detecting Outliers and Change Points from Time Series"
        return 1.0 * (state - predicted_state) ** 2
    
    def anomaly_detection(self, event, threshold):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        expanded_parent_indices = self.bayesian_fitter.get_expanded_parent_indices(expanded_attr_index)
        anomaly_flag = NORMAL; exo_flag = len(expanded_parent_indices) == 0

        #if self.verbosity > 0:
        #    self.chain_manager.print_chains()
        #    str = "Status of current processing.\n"\
        #            + "  * Current event: {}\n".format(event)\
        #            + "  * Exogenous attribute: {}\n".format(exo_flag)
        #    if not exo_flag:
        #        parent_names = [self.expanded_var_names[i] for i in expanded_parent_indices]
        #        str += "    * The parent set: {}\n".format(parent_names)
        #    print(str)

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
        n_affected_chains, detailed_anomaly_flag = self.chain_manager.update(expanded_attr_index, anomaly_flag)
        if detailed_anomaly_flag == ABNORMAL_EXO: # A type-1 anomaly is detected.
            anomalous_interaction = (self.last_processed_event[0], event[0])
            self.anomalous_interaction_dict[anomalous_interaction] = 1 if anomalous_interaction not in self.anomalous_interaction_dict.keys()\
                    else self.anomalous_interaction_dict[anomalous_interaction] + 1
        #self.phantom_state_machine.update(event)
        self.last_processed_event = event
        return anomaly_flag