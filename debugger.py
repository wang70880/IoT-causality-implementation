from src.event_processing import Hprocessor
from src.tigramite.tigramite.pcmci import PCMCI
from src.tigramite.tigramite.independence_tests import CMIsymb
from src.tigramite.tigramite import plotting as tp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

from collections import defaultdict
from torch import var_mean
from pprint import pprint
from time import time

import seaborn as sns
import numpy as np
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
        tested_frame:'DataFrame' = self.preprocessor.frame_dict[0]
        motion_sensor_indices = [k for k in index_device_dict.keys() if index_device_dict[k].name.startswith('M')]

        # Initialize the conditional independence tester
        pc_alpha = 0.0001
        tau_max = 1
        pcmci = PCMCI(
            dataframe=tested_frame.training_dataframe,
            cond_ind_test=CMIsymb(),
            verbosity=-1)
        cond_ind_test = CMIsymb()
        cond_ind_test.set_dataframe(tested_frame.training_dataframe)

        # Test the dependency relationship among motion sensors
        all_parents = defaultdict(dict)
        all_vals = defaultdict(dict)
        all_pvals = defaultdict(dict)

        for i in motion_sensor_indices:
            parents = []
            val_dict = defaultdict(float)
            pval_dict = defaultdict(float)
            for j in motion_sensor_indices:
                parents.append((j, -1))
                val, pval = cond_ind_test.run_test(X=[(j, -1)],
                                                    Y=[(i, 0)],
                                                    Z=[],
                                                    tau_max=tau_max,
                                                    # verbosity=self.verbosity
                                                    )
                val_dict[(j, -1)] = val
                pval_dict[(j, -1)] = pval
            all_parents[i] = parents
            all_vals[i] = val_dict
            all_pvals[i] = pval_dict
        
        def dict_to_matrix(all_vals, tau_max, int_attrs, default=1):
            n_vars = len(int_attrs)
            matrix = np.ones((n_vars, n_vars, tau_max + 1))
            matrix *= default
            for j in all_vals.keys():
                for link in all_vals[j].keys():
                    k, tau = link
                    matrix[int_attrs.index(k), int_attrs.index(j), abs(tau)] = all_vals[j][link]
            return matrix
        
        val_matrix = dict_to_matrix(all_vals, tau_max, motion_sensor_indices, default=0.)
        p_matrix = dict_to_matrix(all_pvals, tau_max, motion_sensor_indices, default=1.)
        final_graph = p_matrix <= pc_alpha
        graph = pcmci.convert_to_string_graph(final_graph)
        #selected_links = {n: [(i, -1) for i in motion_sensor_indices] for n in motion_sensor_indices}
        #symmetrized_results = pcmci.symmetrize_p_and_val_matrix(
        #    p_matrix=p_matrix,
        #    val_matrix=val_matrix,
        #    selected_links=selected_links,
        #    conf_matrix=None)
        
        # Plotting 
        print(val_matrix)
        var_names = [index_device_dict[k].name for k in motion_sensor_indices]
        tp.plot_time_series_graph(
            figsize=(6, 4),
            val_matrix=val_matrix,
            graph=graph,
            var_names=var_names,
            link_colorbar_label='MCI'
        ); plt.show()
                                                

if __name__ == '__main__':
    data_debugger = DataDebugger(dataset=dataset)
    data_debugger.initiate_preprocessing()
    data_debugger.validate_discretization()
    data_debugger.validate_ci_testing()