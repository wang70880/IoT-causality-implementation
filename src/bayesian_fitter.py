import numpy as np
import pandas as pd
from collections import defaultdict
from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import ExpectationMaximization as EM
from pprint import pprint
from time import time

from src.tigramite.tigramite import data_processing as pp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

def _elapsed_minutes(start):
    return (time()-start) * 1.0 / 60

class BayesianFitter:

    def __init__(self, frame, tau_max, link_dict) -> None:
        self.frame:'DataFrame' = frame; self.tau_max = tau_max
        self.dataframe:'pp.DataFrame' = self.frame.training_dataframe
        self.var_names = self.frame.var_names; self.n_vars = len(self.var_names)

        self.extended_name_device_dict:'dict[str, DevAttribute]' = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr name as the dict key
        self.extended_index_device_dict:'dict[int, DevAttribute]' = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr index as the dict key
        self.expanded_var_names, self.n_expanded_vars, self.extended_edges, self.expanded_causal_graph, self.expanded_data_array = self._transform_materials(link_dict)
        self.pd_dataframe = pd.DataFrame(data=self.expanded_data_array, columns=self.expanded_var_names)

        self.isolated_attr_list = []; self.exogenous_attr_list = []; self.stop_attr_list = []
        self._analyze_discovery_statistics()

        self.model = BayesianNetwork(self.extended_edges)
        self.probability_model = self._construct_probability_model()
        self.causal_model = None

    def _lag_name(self, dev:'str', lag:'int'): # Helper function to generate lagged names
        lag = -abs(lag)
        new_name = '{}({})'.format(dev, lag)
        return new_name

    def _transform_materials(self, link_dict):
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

        var_names = self.frame.var_names

        expanded_var_names = []; dev_count = 0
        expanded_causal_graph:'np.ndarray' = None
        expanded_data_array:'np.ndarray' = None
        name_device_dict:'dict[DevAttribute]' = self.frame.name_device_dict
        index_device_dict:'dict[DevAttribute]' = self.frame.index_device_dict

        # Construct expanded_var_names and device info dict: Device with large lags are in the low indices
        for tau in range(0, 2*self.tau_max+1):
            for dev in var_names:
                lagged_dev_name = self._lag_name(dev, -tau)
                extended_dev = DevAttribute(name=lagged_dev_name, index=dev_count, attr=name_device_dict[dev].attr,\
                                        location=name_device_dict[dev].location, lag=-tau)
                self.extended_name_device_dict[lagged_dev_name] = extended_dev; self.extended_index_device_dict[extended_dev.index] = extended_dev
                expanded_var_names.append(extended_dev.name); dev_count += 1
        
        expanded_causal_graph = np.zeros((dev_count,dev_count), dtype=np.uint8)
        extended_edges = []
        for tau in range(0, self.tau_max+1):
            for outcome, cause_list in link_dict.items(): # Construct expanded causal graph (a binary array)
                outcome = outcome
                for cause in cause_list:
                    lagged_cause_name = self._lag_name(index_device_dict[cause[0]].name, cause[1]-tau)
                    lagged_outcome_name = self._lag_name(index_device_dict[outcome].name, -tau)
                    assert(lagged_cause_name in expanded_var_names and lagged_outcome_name in expanded_var_names)
                    extended_edges.append((lagged_cause_name, lagged_outcome_name))
                    expanded_causal_graph[self.extended_name_device_dict[lagged_cause_name].index,\
                                            self.extended_name_device_dict[lagged_outcome_name].index] = 1

        # Construct expanded data array: The concatenation guarantees that device with large lags are in the low indices
        expanded_data_array = np.zeros(shape=(self.dataframe.T-2*self.tau_max, dev_count), dtype=np.uint8)
        for i in range(0, self.dataframe.T - 2*self.tau_max):
            expanded_data_array[i] = np.concatenate([self.dataframe.values[i+tau] for tau in range(2*self.tau_max, -1, -1)])

        return expanded_var_names, dev_count, extended_edges, expanded_causal_graph, expanded_data_array

    def _construct_probability_model(self):
        """
            For each edge X -> Y, we estimate the expected value (E[Y|X=0] and E[Y|X=1])
            Specifically, E[Y|X=0] = 0*P(Y=0|X=0) + 1*P(Y=1|X=0) = P(Y=1|X=0)
        """
        resulting_probability_model = {}
        estimator = MaximumLikelihoodEstimator(data=self.pd_dataframe, model=self.model)
        for edge in self.extended_edges:
            parents = self.model.get_parents(edge[0])
            state_counts = estimator.state_counts(variable=edge[1], parents=[edge[0]])
            count_array = state_counts.values
            resulting_probability_model[edge] = (1.*count_array[1, 0]/np.sum(count_array[:,0]), 1.*count_array[1, 1]/np.sum(count_array[:,1]))
        return resulting_probability_model

    def construct_bayesian_model(self):
        """Construct a parameterized causal graph (i.e., a bayesian model)
        Returns:
            model: A pgmpy.model object
        """
        print("Prepare to initate Bayesian parameter estimation")
        start_time = time()
        edge_list = self.extended_edges
        self.model = BayesianNetwork(edge_list)
        df = pd.DataFrame(data=self.expanded_data_array, columns=self.expanded_var_names)
        #cpd = MaximumLikelihoodEstimator(self.model, df).estimate_cpd(corrs_attr) # JC TEST: The bayesian fitting consumes much time. Let's test the exact consumed time here..
        #print(cpd)
        self.model.fit(df, estimator=EM, n_jobs=50) 
        elapsed_time = _elapsed_minutes(start_time)
        print("Construction finished. Elapsed time: {} mins".format(elapsed_time))
    
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

    def _analyze_discovery_statistics(self):
        """
        This function analyzes
            (1) the statistics of the degree, and
            (2) variable types according to its degrees.
        """

        # 1. Get the degree statistics
        incoming_degree_dict = {var_name: sum(self.expanded_causal_graph[:,self.extended_name_device_dict[self._lag_name(var_name,0)].index])\
                                for var_name in self.var_names}; in_degrees = list(incoming_degree_dict.values())
        avg_in_degree = 0 if len(in_degrees)==0 else sum(in_degrees)*1.0/len(in_degrees)
        max_in_degree = max(in_degrees); max_in_attr = self.extended_index_device_dict[in_degrees.index(max_in_degree)].name
        min_in_degree = min(in_degrees); min_in_attr = self.extended_index_device_dict[in_degrees.index(min_in_degree)].name

        # 2. Get the variable statistics
        outcoming_degree_dict = {}
        for var_name in self.var_names:
            outcoming_degree = 0
            for tau in range(1, self.tau_max + 1):
                extended_dev = self._lag_name(var_name, tau)
                outcoming_degree += sum(self.expanded_causal_graph[self.extended_name_device_dict[extended_dev].index])
            outcoming_degree_dict[var_name] = outcoming_degree
        isolated_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] + outcoming_degree_dict[var_name] == 0]
        exogenous_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] == 0 and outcoming_degree_dict[var_name] > 0]
        stop_attr_list = [var_name for var_name in self.var_names if incoming_degree_dict[var_name] > 0 and outcoming_degree_dict[var_name] == 0]
        outcoming_degree_list = list(outcoming_degree_dict.values()); incomming_degree_list = list(incoming_degree_dict.values())

        str = " [Bayesian Fitting]\n"\
            + " * isolated attrs, #: {}, {}\n".format(isolated_attr_list, len(isolated_attr_list))\
            + " * stop attrs, #: {}, {}\n".format(stop_attr_list, len(stop_attr_list))\
            + " * exogenous attrs, #: {}, {}\n".format(exogenous_attr_list, len(exogenous_attr_list))\
            + " * # edges: {}\n".format(np.sum(self.expanded_causal_graph))\
            + " * (max, mean, min) for outcoming degrees: ({}, {}, {})\n".format(max(outcoming_degree_list),\
                        sum(outcoming_degree_list)*1.0/(self.n_vars - len(isolated_attr_list)), min(outcoming_degree_list))\
            + " * (max, mean, min) for incoming degrees: ({}, {}, {})\n".format(max_in_degree, avg_in_degree, min_in_degree)
        
        print(str)

        if max_in_degree > 10:
            print("[Bayesian Fitting] ALERT! The variable {} owns the maximum in-degree {} (larger than 10). This variable may slow down the fitting process!".format(max_in_attr, max_in_degree))

        self.isolated_attr_list = isolated_attr_list; self.exogenous_attr_list = exogenous_attr_list; self.stop_attr_list = stop_attr_list
