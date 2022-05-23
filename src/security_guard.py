import numpy as np

def _lag_name(attr:'str', lag:'int'):
    assert(lag >= 0)
    new_name = '{}({})'.format(attr, -1 * lag) if lag > 0 else '{}'.format(attr)
    return new_name

class PhantomStateMachine():

    def __init__(self, var_names, expanded_var_names) -> None:
        self.var_names = var_names; self.expanded_var_names = expanded_var_names
        self.n_vars = len(var_names); self.n_expanded_vars = len(self.expanded_var_names)
        self.phantom_states = [0] * self.n_expanded_vars

    def set_state(self, current_state_vector):
        assert(len(current_state_vector) == self.n_vars)
        self.phantom_state_machine = [*self.phantom_state_machine[self.n_vars:], *current_state_vector]

    def update_state(self, event):
        attr = event[0]; state = event[1]
        current_state_vector = self.phantom_state_machine[-1*self.n_vars:].copy()
        current_state_vector[self.var_names.index(attr)] = state
        self.set_state(current_state_vector)

    def get_states(self, attr_list):
        return {attr: self.phantom_states[self.expanded_var_names.index(attr)] for attr in attr_list}

class ChainManager():
    
    #JC TODO: Currently it only supports the pool check/update for tau =1.

    def __init__(self) -> None:
        self.chain_pool = {}
    
    def update_pool(self, attr, expanded_parent_list):
         
        lag_attr = _lag_name(attr, 1)
        if len(expanded_parent_list) == 0:
            self.chain_pool[attr] = [] if attr not in self.chain_pool.keys() else self.chain_pool[attr]
            self.chain_pool[attr].append(lag_attr)
        else:
            for chain in self.chain_pool.values():
                if chain[-1] in expanded_parent_list:
                    chain.append(lag_attr)

    def check_pool(self, expanded_parent_list):
        # An exogenous attribute or there is a chain matching current attr's parent.
        return len(expanded_parent_list) == 0 or any([chain[-1] in expanded_parent_list for chain in self.chain_pool.values()])

class SecurityGuard():

    def __init__(self, bayesian_fitter) -> None:
        # The parameterized causal graph
        self.bayesian_fitter = bayesian_fitter
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names, bayesian_fitter.expanded_var_names)
        # Chain manager
        self.chain_manager = ChainManager()
    
    def initialize(self, event):
        self.phantom_state_machine.update_state(event)
        self.chain_manager.update_pool(event[0], [])
    
    def _compute_anomaly_score(self, state, predicted_state):
        # Here we take the quadratic loss as the anomaly score.
        # See paper "A Unifying Framework for Detecting Outliers and Change Points from Time Series"
        return 1.0 * (state - predicted_state) ** 2
    
    def anomaly_detection(self, event, threshold):
        # JC TODO: Calculating the threshold is not implemented.
        attr = event[0]; state = event[1]
        anomaly_flag = 0
        expanded_parent_list = self.bayesian_fitter.get_parents(attr)
        # First initiate the pool check to identify the 1st-type violation
        anomaly_flag = self.chain_manager.check_pool(attr, expanded_parent_list)
        # Then initiate the probability check to identify the 2nd-type violation
        parent_state_dict = self.phantom_state_machine.get_states(expanded_parent_list)
        predicted_state =  self.bayesian_fitter.predict_attr_state(attr, parent_state_dict)
        anomaly_score = self._compute_anomaly_score(state, predicted_state)
        anomaly_flag = anomaly_score > threshold
        if not anomaly_flag: # If the current event is not an anomaly: Update the chain pool and the phantom state machine
            self.chain_manager.update_pool(attr, expanded_parent_list)
            self.phantom_state_machine.update_state(event)
        return anomaly_flag