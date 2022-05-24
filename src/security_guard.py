import numpy as np

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

class ChainManager():
    
    def __init__(self, var_names, expanded_var_names, expanded_causal_graph) -> None:
        self.chain_pool:'list[list[int]]' = []
        self.var_names = var_names; self.n_vars = len(self.var_names)
        self.expanded_var_names = expanded_var_names; self.n_expanded_vars = len(self.expanded_var_names)
        self.expanded_causal_graph = expanded_causal_graph
    
    def insert(self, expanded_attr_index:'int'):
        if len(self.match(expanded_attr_index)) > 0: # If there exists matched chains (which also means that the attribute is not exogenous), just insert the attribute.
            self.update(expanded_attr_index, False)
        else: # No matched chains, just create a new chain for current attributes (no matter whether it is exogenous or endogenous)
            lagged_attr_index = expanded_attr_index - self.n_vars
            if [lagged_attr_index] not in self.chain_pool: 
                self.chain_pool.append([lagged_attr_index])
    
    def match(self, expanded_attr_index:'int'):
        """Identify the set of chains which can accommodate the new attribute
        That is, there exists a lagged variable in the candidate chain which is the parent of the new attribute.

        Args:
            expanded_attr_index (int): The new attribute

        Returns:
            matched_chain_indices: The list of indices for satisfied chains (in the chain pool)
        """
        satisfied_chain_indices: 'list[int]'  = []
        for chain in self.chain_pool:
            connection_list = [x for x in chain if self.expanded_causal_graph[x, expanded_attr_index] > 0]
            if len(connection_list) > 0:
                satisfied_chain_indices.append(self.chain_pool.index(chain))
        return satisfied_chain_indices
    
    def update(self, expanded_attr_index:'int', exo_flag):
        """Update the existing chains given the new arrived attribute event.

        Args:
            expanded_attr_index (int): The new attribute
            n_parents (_type_): Number of the new attributes' parents

        Returns:
            int: The number of updated chains
        """
        affected_chains = 0
        lagged_attr_index = expanded_attr_index - self.n_vars 
        if exo_flag and [lagged_attr_index] not in self.chain_pool: # If the current attribute is an exogenous attribute and does not exist in the current pool: Create a new chain for it.
            self.chain_pool.append([lagged_attr_index])
            affected_chains += 1
        elif not exo_flag:
            matched_chain_indices = self.match(expanded_attr_index)
            for chain_index in matched_chain_indices:
                self.chain_pool[chain_index].append(expanded_attr_index)
                self.chain_pool[chain_index] = [x - self.n_vars for x in self.chain_pool[chain_index] if x - self.n_vars >= 0] # Update the time lag for each chain
                affected_chains += 1
        return affected_chains

    def print_chains(self):
        for index, chain in enumerate(self.chain_pool):
            attr_name_chain = [self.expanded_var_names[i] for i in chain]
            print("Chain {}: {}".format(index, '->'.join(attr_name_chain)))

class SecurityGuard():

    def __init__(self, bayesian_fitter, verbosity) -> None:
        self.verbosity = verbosity
        self.var_names: 'list[str]' = bayesian_fitter.var_names
        self.expanded_var_names: 'list[str]' = bayesian_fitter.expanded_var_names
        # The parameterized causal graph
        self.bayesian_fitter = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names, bayesian_fitter.expanded_causal_graph)
    
    def initialize(self, event, state_vector):
        self.phantom_state_machine.set_state(state_vector) # Initialize the phantom state machine
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        self.chain_manager.insert(expanded_attr_index) # Initialize the chain manager
    
    def _compute_anomaly_score(self, state, predicted_state):
        # Here we take the quadratic loss as the anomaly score.
        # See paper "A Unifying Framework for Detecting Outliers and Change Points from Time Series"
        return 1.0 * (state - predicted_state) ** 2
    
    def anomaly_detection(self, event, threshold):
        attr = event[0]; expanded_attr_index = self.expanded_var_names.index(attr)
        anomaly_flag = 1
        expanded_parent_indices = self.bayesian_fitter.get_expanded_parent_indices(expanded_attr_index); exo_flag = len(expanded_parent_indices) > 0

        # First initiate detections of type-1 attacks.
        if exo_flag or len(self.chain_manager.match(expanded_attr_index)) > 0:
            anomaly_flag = 0

        if self.verbosity > 0 and anomaly_flag > 0:
            str = "\nType-1 anomalies are detected.\n"\
                    + "  * Current event: {}\n".format(event)\
                    + "  * Exogenous attribute: {}".format(exo_flag)
            print(str)
            self.chain_manager.print_chains()

        # JC TODO: Initiate detections of type-2 attacks.
        # parent_state_dict = self.phantom_state_machine.get_states(expanded_parent_list)
        # predicted_state =  self.bayesian_fitter.predict_attr_state(attr, parent_state_dict)
        # anomaly_score = self._compute_anomaly_score(state, predicted_state)
        # anomaly_flag = anomaly_score > threshold

        # Update the chain pool and the phantom state machine according to the detection result.
        if not anomaly_flag: # If the current event is not an anomaly: Update the chain pool and the phantom state machine
            self.chain_manager.update(expanded_attr_index, exo_flag)
            self.phantom_state_machine.update(event)
        return anomaly_flag