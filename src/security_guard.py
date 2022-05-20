import numpy as np
class SecurityGuard():
    def __init__(self, bayesian_fitter) -> None:
        # The parameterized causal graph
        self.tau_max = bayesian_fitter.tau_max
        self.expanded_var_names = bayesian_fitter.expanded_var_names; self.n_expanded_vars = len(self.expanded_var_names)
        self.expanded_causal_graph = bayesian_fitter.expanded_causal_graph
        self.n_vars = int(self.n_expanded_vars / (self.tau_max + 1)); self.var_names = self.expanded_var_names[-self.n_vars:].copy()
        self.model = bayesian_fitter.model
        # Phantom state machine
        self.phantom_state_machine = [0] * self.n_expanded_vars
    
    def set_phantom_state_machine(self, current_state_vector):
        self.phantom_state_machine = [*self.phantom_state_machine[self.n_vars:], *current_state_vector]

    def update_phantom_state_machine(self, event):
        attr = event[0]; state = event[1]
        current_state_vector = self.phantom_state_machine[-1*self.n_vars:].copy()
        current_state_vector[self.var_names.index(attr)] = state
        self.set_phantom_state_machine(current_state_vector)
    
    def anomaly_detection(self, event):
        anomaly_flag = 0
        num_parent = self._num_parents(event[0])
        exo_flag = 0
        if num_parent > 0:
            posterior_prob = self._lookup_posterior_probability(event) # Get estimated probability
            print(posterior_prob)
            # JC TODO: Determine the anomaly flag according to the posterior probability
        else:
            # JC TODO: What if the causal discovery algorithm did not return any parents of the variable?
            exo_flag = 1
        if anomaly_flag == 0:
            self.update_phantom_state_machine(event)
        return exo_flag, anomaly_flag
    
    def _num_parents(self, attr:'str'):
        attr_expanded_index = self.expanded_var_names.index(attr)
        return sum(self.expanded_causal_graph[:,attr_expanded_index])
    
    def _lookup_posterior_probability(self, event):
        attr = event[0]; state = event[1]
        attr_expanded_index = self.expanded_var_names.index(attr)
        par_indices = np.where(self.expanded_causal_graph[:,attr_expanded_index] == 1)[0]; par_names = [self.expanded_var_names[i] for i in par_indices]
        phantom_states = [self.phantom_state_machine[x] for x in par_indices]
        phi = self.model.get_cpds(attr).to_factor()
        print(phi)
        print("\n\n\n\n\n")
        prob = 0
        return prob
        