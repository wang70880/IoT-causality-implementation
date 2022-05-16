import numpy as np
import bnlearn as bn
import pandas as pd
from time import time
import statistics

def _lag_name(attr:'str', lag:'int'):
    new_name = '{}({})'.format(attr, -1 * lag) if lag > 0 else '{}({})'.format(attr, lag)
    return new_name

class BayesianPredictor:

    def __init__(self, dataframe, tau_max, link_dict) -> None:
        self.expanded_var_names, self.expanded_causal_graph, self.expanded_data_array =\
                     self._transform_materials(dataframe, tau_max, link_dict)
        self.n_vars = len(self.expanded_var_names)
    
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

    def analyze_discovery_statistics(self):
        print("[BayesianPredictor] Analyzing discovery statistics.")
        outcoming_degree_list = [sum(self.expanded_causal_graph[i]) for i in range(self.n_vars)]
        incoming_degree_list = [sum(self.expanded_causal_graph[:,i]) for i in range(self.n_vars)]
        isolated_attr_list = [self.expanded_var_names[i] for i in range(self.n_vars)\
                                    if outcoming_degree_list[i] + incoming_degree_list[i] == 0]
        str = " * # isolated attrs: {}\n".format(len(isolated_attr_list))\
            + " * # no-out attrs: {}\n".format(outcoming_degree_list.count(0) - len(isolated_attr_list))\
            + " * # no-incoming attrs: {}\n".format(incoming_degree_list.count(0) - len(isolated_attr_list))\
            + " * (max, mean, min) for outcoming degrees: ({}, {}, {})\n".format(max(outcoming_degree_list),\
                        sum(outcoming_degree_list)* 1.0/(self.n_vars - len(isolated_attr_list)), min(outcoming_degree_list))\
            + " * (max, mean, min) for incoming degrees: ({}, {}, {})\n".format(max(incoming_degree_list),\
                        sum(incoming_degree_list)* 1.0/(self.n_vars - len(isolated_attr_list)), min(incoming_degree_list))
        print(str)

    def _construct_bayesian_model(self):
        start = time()
        edge_list = [(self.expanded_var_names[i], self.expanded_var_names[j])\
                        for (i, j), x in np.ndenumerate(self.expanded_causal_graph) if x == 1]
        print(edge_list)
        dag = bn.make_DAG(edge_list); df = pd.DataFrame(data=self.expanded_data_array, columns=self.expanded_var_names)
        model_mle = bn.parameter_learning.fit(dag, df, methodtype='maximumlikelihood')
        end = time()
        print("Consumption time for MLE: {} seconds".format((end-start) * 1.0 / 60))