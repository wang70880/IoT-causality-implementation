import numpy as np
class SecurityGuard():
    def __init__(self, tau_max, expanded_var_names, expanded_causal_graph) -> None:
        self.tau_max = tau_max
        self.expanded_var_names = expanded_var_names; self.n_expanded_vars = len(expanded_var_names)
        self.n_vars = self.n_expanded_vars / (self.tau_max + 1); self.var_names = self.expanded_var_names[-1*self.n_vars:].copy()
        self.expanded_causal_graph = expanded_causal_graph
        self.phantom_state_machine = [0] * self.n_expanded_vars
    
    def _update_phantom_state_machine(self, event):
        attr = event[0]; state = event[1]
        current_state_vector = self.phantom_state_machine[-1*self.n_vars:].copy()
        current_state_vector[self.var_names.index(attr)] = state
        self.phantom_state_machine = [*self.phantom_state_machine[self.n_vars:], *current_state_vector]

    def estimate_event_probability(self, event):
        attr = event[0]; state = event[1]
        attr_expanded_index = self.expanded_var_names.index(attr)
        anomaly_flag = 1
        num_parent = sum(self.expanded_causal_graph[:,attr_expanded_index])
        if num_parent > 0:
            parent_set = [self.expanded_var_names[x] for x in np.where(self.expanded_causal_graph[:,attr_expanded_index] == 1)]
            # JC TODO: Get the estimated probability and justify the event
        else:
            # JC TODO: What if the causal discovery algorithm did not return any parents of the variable?
            pass
        if anomaly_flag == 0:
            self._update_phantom_state_machine(event)