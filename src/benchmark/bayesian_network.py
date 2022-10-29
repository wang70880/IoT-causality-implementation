import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore

class BayesianMiner():

    def __init__(self, dataframe:'pd.DataFrame', tau_max):
        self.dataframe = dataframe
        self.tau_max = tau_max
        self.expanded_var_names = list(dataframe.columns); self.n_expanded_vars = len(self.expanded_var_names)
        self.n_vars = int(self.n_expanded_vars/(self.tau_max+1)); self.var_names = list(self.expanded_var_names[:self.n_vars])

    def _normalize_temporal_array(self, target_array:'np.ndaray', tau_max):
        new_array = target_array.copy()
        if len(new_array.shape) == 3 and new_array.shape[-1] == tau_max+1:
            new_array = sum([new_array[:,:,tau] for tau in range(1, tau_max+1)])
            new_array[new_array>0] = 1
        return new_array

    def structure_learning(self):
        bayesian_edges = {}
        bayesian_array = np.zeros((self.n_vars, self.n_vars, self.tau_max+1), dtype=np.int8)
        nor_bayesian_array = np.zeros((self.n_vars, self.n_vars), dtype=np.int8)

        white_list = [(x, y) for x in list(self.expanded_var_names[self.n_vars:]) for y in self.var_names]
        est = HillClimbSearch(self.dataframe)
        best_model = est.estimate(scoring_method='bdsscore', white_list=white_list, epsilon=1e-3, show_progress=False)

        for edge in best_model.edges():
            cause = edge[0]; outcome = edge[1]
            cause_name = str(cause.split('(')[0]); c_lag = int(cause.split('(')[1].split(')')[0])
            c_index = self.var_names.index(cause_name); o_index = self.var_names.index(outcome)

            bayesian_edges[o_index] = bayesian_edges[o_index] if o_index in bayesian_edges.keys() else []
            bayesian_edges[o_index].append((c_index, c_lag))
            bayesian_array[c_index, o_index, abs(c_lag)] = 1
        nor_bayesian_array = self._normalize_temporal_array(bayesian_array, self.tau_max)
        return bayesian_edges, bayesian_array, nor_bayesian_array