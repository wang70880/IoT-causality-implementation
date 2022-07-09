from src.event_processing import Hprocessor
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb
from src.tigramite.tigramite import plotting as tp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

from collections import defaultdict
from torch import var_mean
from pprint import pprint
from time import time

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# PARAM SETTING
dataset = 'hh130'

class DataDebugger():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.preprocessor = Hprocessor(dataset=dataset)

    def initiate_preprocessing(self):
        self.preprocessor.initiate_data_preprocessing(partition_config=30, training_ratio=1.0)
    
    def validate_discretization(self):
        for dev, tup in self.preprocessor.discretization_dict.items():
            val_list = tup[0]; seg_point = tup[1]
            sns.displot(val_list, kde=False, color='red', bins=1000)
            plt.axvline(seg_point, 0, 1)
            plt.title('State distributions of device {}'.format(dev))
            plt.xlabel('State')
            plt.ylabel('Frequency')
            plt.savefig("temp/image/{}_{}_states.pdf".format(self.dataset, dev))

    def validate_ci_testing(self):

        # Load data and related attributes
        index_device_dict:'dict[DevAttribute]' = self.preprocessor.index_device_dict
        name_device_dict:'dict[DevAttribute]' = self.preprocessor.name_device_dict
        tested_frame:'DataFrame' = self.preprocessor.frame_dict[0]
        int_device_indices = [k for k in index_device_dict.keys() if index_device_dict[k].name.startswith('M')]
        n_vars = len(int_device_indices)

        # Initialize the conditional independence tester
        pc_alpha = 0.01
        tau_max = 4; int_tau = 4
        assert(int_tau <= tau_max)
        selected_links = {n: [(i, int(-int_tau)) for i in int_device_indices] for n in int_device_indices}
        pcmci = PCMCI(
            dataframe=tested_frame.training_dataframe,
            cond_ind_test=CMIsymb(),
            verbosity=-1)

        # Test the dependency relationship among motion sensors
        all_parents = defaultdict(dict); all_vals = defaultdict(dict); all_pvals = defaultdict(dict)

        var_parent = 'M001'; tau = -2
        var_child = 'M003'
        cond = True

        val, pval = pcmci.cond_ind_test.run_test(
            X = [(name_device_dict[var_parent].index, tau)],
            Y = [(name_device_dict[var_child].index, 0)],
            #Z = [],
            Z = [(name_device_dict[var_parent].index, tau+1)] if cond else [],
            tau_max = tau_max
        )
        print("{} {} {}".format(val, pval, pval <= pc_alpha))
        exit()

        for i in int_device_indices: # Calculate the statistic values and the p value
            parents = []
            val_dict = defaultdict(float)
            pval_dict = defaultdict(float)
            for j in int_device_indices:
                for tau in range(1, tau_max + 1):
                    lag = int(-tau)
                    Z = []
                    #Z = [(j, int(-inter_tau)) for inter_tau in range(1, tau-1)] # If conditioning on the previous variables
                    val, pval = pcmci.cond_ind_test.run_test(X=[(j, lag)],
                                                        Y=[(i, 0)],
                                                        Z=Z,
                                                        tau_max=tau_max)
                    parents.append((j, lag))
                    val_dict[(j, lag)] = val
                    pval_dict[(j, lag)] = pval
            all_parents[i] = parents
            all_vals[i] = val_dict
            all_pvals[i] = pval_dict
        
        def dict_to_matrix(all_vals, tau_max, int_attrs, default=1): # Inner helpful functions for output format transformation
            n_vars = len(int_attrs)
            matrix = np.ones((n_vars, n_vars, tau_max + 1))
            matrix *= default
            for j in all_vals.keys():
                for link in all_vals[j].keys():
                    k, tau = link
                    matrix[int_attrs.index(k), int_attrs.index(j), abs(tau)] = all_vals[j][link]
            return matrix
        
        val_matrix = dict_to_matrix(all_vals, tau_max, int_device_indices, default=0.)
        p_matrix = dict_to_matrix(all_pvals, tau_max, int_device_indices, default=1.)
        selected_links_matrix = np.empty((n_vars, n_vars, tau_max + 1)); selected_links_matrix.fill(1.1)
        for j in selected_links.keys():
            for link in selected_links[j]:
                k, tau = link
                selected_links_matrix[int_device_indices.index(k), int_device_indices.index(j), abs(tau)] = 0.
        final_graph = p_matrix + selected_links_matrix # Adjust graph according to the selected links: For non-selected links, penalize its p-value by adding 1.1
        final_graph = final_graph <= pc_alpha # Adjust graph according to hypothesis testing
        graph = pcmci.convert_to_string_graph(final_graph)

        # Sorting all parents according to the CMI value
        for dev_index in int_device_indices:
            print("Sorted parents for device {}:".format(index_device_dict[dev_index].name))
            parents = [x for x in int_device_indices if final_graph[int_device_indices.index(x), int_device_indices.index(dev_index), int_tau]]
            parent_vals = [(x, val_matrix[int_device_indices.index(x) , int_device_indices.index(dev_index), int_tau])
                                 for x in parents]
            parent_vals.sort(key=lambda tup: tup[1], reverse=True)
            print(" -> ".join( [index_device_dict[x[0]].name for x in parent_vals] ))
        
        # Plotting
        var_names = [index_device_dict[k].name for k in int_device_indices]
        tp.plot_time_series_graph(
            figsize=(6, 4),
            val_matrix=val_matrix,
            graph=graph,
            var_names=var_names,
            link_colorbar_label='MCI'
        )
        plt.savefig("temp/image/{}cmi_test_tau{}.pdf".format(self.dataset, int_tau))
    
    def cpt_testing(self, x:'str', lag:'int', y:'str'):
        assert(lag > 0)
        # Load data and related attributes
        tested_frame:'DataFrame' = self.preprocessor.frame_dict[0]
        name_device_dict = self.preprocessor.name_device_dict
        x_index = name_device_dict[x].index; y_index = name_device_dict[y].index

        # We hope to get the result of CPT Pr(y|(x,lag))
        devices = ['{}-{}'.format(x, lag), y]
        data_array, xyz = tested_frame.training_dataframe.construct_array(
            X=[(x_index, int(-lag))],
            Y=[(y_index, 0)],
            Z=[],
            tau_max=lag,
            mask_type=None,
            return_cleaned_xyz=False,
            do_checks=True,
            cut_off='2xtau_max'
        )
        df = pd.DataFrame(data=np.transpose(data_array), columns=devices)
        model = BayesianNetwork([(devices[0], devices[1])])
        model.fit(df, estimator=MaximumLikelihoodEstimator)
        print(model.get_cpds(devices[0]))
        print(model.get_cpds(devices[1]))

if __name__ == '__main__':
    data_debugger = DataDebugger(dataset=dataset)
    data_debugger.initiate_preprocessing()
    data_debugger.validate_discretization()
    data_debugger.validate_ci_testing()
    #data_debugger.cpt_testing('M001', 2, 'M001')