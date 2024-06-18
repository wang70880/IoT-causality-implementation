import numpy as np
import pandas as pd
from collections import defaultdict
from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

from src.tigramite.tigramite import data_processing as pp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

class BayesianFitter:

    def __init__(self, frame, tau_max, link_dict) -> None:
        self.frame:'DataFrame' = frame; self.tau_max = tau_max
        self.dataframe:'pp.DataFrame' = self.frame.training_dataframe
        self.extended_name_device_dict:'dict[str, DevAttribute]' = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr name as the dict key
        self.extended_index_device_dict:'dict[int, DevAttribute]' = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr index as the dict key
        self.var_names = self.frame.var_names; self.n_vars = len(self.var_names)
        self.expanded_var_names, self.n_expanded_vars, self.expanded_causal_graph, self.expanded_data_array =\
                     self._transform_materials(self.frame.var_names, link_dict)
        self.model = None
        self.isolated_attr_list = []; self.exogenous_attr_list = []; self.stop_attr_list = []
        self._analyze_discovery_statistics()

    def _lag_name(self, attr:'str', lag:'int'): # Helper function to generate lagged names
        lag = -abs(lag)
        new_name = '{}({})'.format(attr, lag) if lag != 0 else '{}'.format(attr)
        return new_name

    def _transform_materials(self, var_names, link_dict):
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
        expanded_var_names = []; dev_count = 0
        expanded_causal_graph:'np.ndarray' = None; expanded_data_array:'np.ndarray' = None

        for tau in range(0, self.tau_max + 1): # Construct expanded_var_names and device info dict: Device with large lags are in the low indices
            lag = self.tau_max - tau
            for dev in var_names:
                extended_dev = DevAttribute(attr_name=self._lag_name(dev, lag), attr_index=dev_count, lag=lag)
                self.extended_name_device_dict[extended_dev.name] = extended_dev; self.extended_index_device_dict[extended_dev.index] = extended_dev
                expanded_var_names.append(extended_dev.name); dev_count += 1
        
        expanded_causal_graph = np.zeros((dev_count, dev_count), dtype=np.uint8)
        for outcome, cause_list in link_dict.items(): # Construct expanded causal graph (a binary array)
            for (cause, lag) in cause_list:
                cause_name = self._lag_name(cause, lag)
                expanded_causal_graph[self.extended_name_device_dict[cause_name].index,\
                                    self.extended_name_device_dict[outcome].index] = 1

        expanded_data_array = np.zeros(shape=(self.dataframe.T - self.tau_max, dev_count), dtype=np.uint8)
        for i in range(0, self.dataframe.T - self.tau_max): # Construct expanded data array: The concatenation guarantees that device with large lags are in the low indices
            expanded_data_array[i] = np.concatenate([self.dataframe.values[i+tau] for tau in range(0, self.tau_max+1)])

        return expanded_var_names, dev_count, expanded_causal_graph, expanded_data_array

    def construct_bayesian_model(self):
        """Construct a parameterized causal graph (i.e., a bayesian model)

        Returns:
            model: A pgmpy.model object
        """
        edge_list = [(self.extended_index_device_dict[i].name, self.extended_index_device_dict[j].name)\
                        for (i, j), x in np.ndenumerate(self.expanded_causal_graph) if x == 1]
        in_degrees = [sum(self.expanded_causal_graph[:, self.extended_name_device_dict[var_name].index]) for var_name in self.var_names]
        avg_degree = 0 if len(in_degrees)==0 else sum(in_degrees)*1.0/len(in_degrees); max_degree = max(in_degrees); max_attr = self.extended_index_device_dict[in_degrees.index(max_degree)].name
        print("[Bayesian Fitting] Total, Average, Max = {}, {}, {}".format(sum(in_degrees), avg_degree, max_degree))
        if max_degree > 10:
            print("[Bayesian Fitting] ALERT! The variable {} owns the maximum in-degree {} (larger than 10). This variable may slow down the fitting process!".format(max_attr, max_degree))
        self.model = BayesianNetwork(edge_list)
        df = pd.DataFrame(data=self.expanded_data_array, columns=self.expanded_var_names)
        #cpd = MaximumLikelihoodEstimator(self.model, df).estimate_cpd(corrs_attr) # JC TEST: The bayesian fitting consumes much time. Let's test the exact consumed time here..
        #print(cpd)
        self.model.fit(df, estimator= BayesianEstimator) 
    
    def estimate_cond_probability(self, event:'AttrEvent', parent_states:'list[tuple(str, int)]', selected_evidences:'list[str]'):
        """
        Predict the value of the target attribute given its parent states, i.e., E[attr|par(attr)]
        """
        # 1. Get the CPT for the interested variable
        cpd = self.model.get_cpds(event.dev).copy()
        parents = cpd.get_evidence()
        assert(len(parents) == len(parent_states))
        # 2. Get the list of variables to be marginalized, and marginalized the CPT
        marginalized_list = []
        for parent in parents:
            if not parent.startswith(tuple(set(selected_evidences))):
                marginalized_list.append(parent)
        cpd.marginalize(marginalized_list)
        # 3. Get the conditional probability (with only selected evidences)
        selected_parents = [parent for parent in parents if parent not in marginalized_list]
        val_dict = {k:v for (k,v) in parent_states if k in selected_parents}; val_dict[event.dev] = event.value
        cond_prob = 0 if len(selected_parents) == 0 else cpd.get_value(**val_dict)
        return cond_prob

    def get_expanded_parents(self, child_device: 'DevAttribute'):
        # Return variables
        parents = []
        for (i,), x in np.ndenumerate(self.expanded_causal_graph[:,child_device.index]):
            if x == 1:
                parents.append(self.extended_index_device_dict[i])
        return parents 

    def get_expanded_children(self, parents:'list[str]'):
        children = set()
        for parent in parents:
            for lag in range(1, self.tau_max + 1):
                lagged_parent = self._lag_name(parent, lag); lagged_parent_index = self.extended_name_device_dict[lagged_parent].index
                its_children_names = set([self.extended_index_device_dict[x].name for x in range(self.n_expanded_vars) if self.expanded_causal_graph[lagged_parent_index, x] == 1])
                children = children.union(its_children_names)
        return list(children)

    def _analyze_discovery_statistics(self):
        incoming_degree_dict = {var_name: sum(self.expanded_causal_graph[:,self.extended_name_device_dict[var_name].index])\
                                for var_name in self.var_names}
        outcoming_degree_dict = {}
        for var_name in self.var_names:
            outcoming_degree = 0
            for tau in range(1, self.tau_max + 1):
                extended_dev = self._lag_name(var_name, tau)
                outcoming_degree += sum(self.expanded_causal_graph[self.extended_name_device_dict[extended_dev].index])
            outcoming_degree_dict[var_name] = outcoming_degree
        #nointeraction_attr_list = []
        #for var_name in self.var_names:
        #    var_index = self.extended_name_device_dict[var_name].index
        #    parents_index = [k for k in range(self.n_expanded_vars) if self.expanded_causal_graph[k, var_index] > 0]
        #    if all([(p - var_index) % self.n_vars == 0 for p in parents_index]): # If the current variable only contains autocorrelated parents
        #        nointeraction_attr_list.append(var_name)
        isolated_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] + outcoming_degree_dict[var_name] == 0]
        exogenous_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] == 0 and outcoming_degree_dict[var_name] > 0]
        stop_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] > 0 and outcoming_degree_dict[var_name] == 0]
        outcoming_degree_list = list(outcoming_degree_dict.values()); incomming_degree_list = list(incoming_degree_dict.values())

        str = " * isolated attrs, #: {}, {}\n".format(isolated_attr_list, len(isolated_attr_list))\
            + " * stop attrs, #: {}, {}\n".format(stop_attr_list, len(stop_attr_list))\
            + " * exogenous attrs, #: {}, {}\n".format(exogenous_attr_list, len(exogenous_attr_list))\
            + " * # edges: {}\n".format(np.sum(self.expanded_causal_graph))\
            + " * (max, mean, min) for outcoming degrees: ({}, {}, {})\n".format(max(outcoming_degree_list),\
                        sum(outcoming_degree_list)*1.0/(self.n_vars - len(isolated_attr_list)), min(outcoming_degree_list))\
            + " * (max, mean, min) for incoming degrees: ({}, {}, {})\n".format(max(incomming_degree_list),\
                        sum(incomming_degree_list)*1.0/(self.n_vars - len(isolated_attr_list)), min(incomming_degree_list))\
            #+ " * no-interaction attrs, #: {}, {}\n".format(nointeraction_attr_list, len(nointeraction_attr_list))\

        self.isolated_attr_list = isolated_attr_list; self.exogenous_attr_list = exogenous_attr_list; self.stop_attr_list = stop_attr_list
