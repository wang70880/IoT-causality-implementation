import numpy as np
from pomegranate import *

def _lag_name(attr:'str', lag:'int'):
    return '{}({})'.format(attr, -1 * lag)

class BayesianPredictor:

    def __init__(self, dataframe, tau_max, link_dict) -> None:
        self.expanded_var_names, self.expanded_causal_graph, self.expanded_data_array =\
                     self._transform_materials(dataframe, tau_max, link_dict)
    
    def _transform_materials(self, dataframe, tau_max, link_dict):
        """
        This function transforms the original N variables into N(tau_max + 1) variables
        As a result,
            1. the resulting number of variables equals to N(tau_max + 1)
            2. The resulting causal graph is a N(tau_max + 1) * N(tau_max + 1) binary array
            3. The resulting data array is of shape (T - tau_max) * N(tau_max + 1)
        Args:
            dataframe (_type_): The original data frame
            tau_max (_type_): User-specified maximum time lag
            link_dict (_type_): Results returned by pc-stable algorithm, which is a dict recording the edges
        Returns:
            expanded_var_names (list[str]): Record the name of expanded variables (starting from t-tau_max to t)
        """
        expanded_var_names = dataframe.var_names.copy(); expanded_causal_graph = None; expanded_data_array = None
        for tau in range(1, tau_max + 1): # Construct expanded_var_names
            expanded_var_names = [*[_lag_name(x, tau) for x in dataframe.var_names], *expanded_var_names]
        expanded_causal_graph = np.zeros(shape=(len(expanded_var_names), len(expanded_var_names)), dtype=np.uint8)
        for outcome, cause_list in link_dict.items(): # Construct expanded causal graph (a binary array)
            for (cause, lag) in cause_list:
                expanded_causal_graph[expanded_var_names.index(_lag_name(cause, lag)), expanded_var_names.index(outcome)] = 1
        expanded_data_array = np.zeros(shape=(dataframe.T - tau_max, len(expanded_var_names)), dtype=np.uint8)
        for i in range(0, dataframe.T - tau_max): # Construct expanded data array
            expanded_data_array[i] = np.concatenate([dataframe.values[i+tau] for tau in range(0, tau_max+1)])
        return expanded_var_names, expanded_causal_graph, expanded_data_array

    def _construct_bayesian_model(self):
        edge_list = [(self.expanded_var_names[i], self.expanded_var_names[j])\
                        for (i, j), x in np.ndenumerate(self.expanded_causal_graph) if x == 1]
        print(edge_list)