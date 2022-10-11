import collections
import itertools
import numpy as np
import random
import pandas as pd
from src.tigramite.tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from numpy import ndarray
from src.event_processing import Hprocessor, Cprocessor, GeneralProcessor
from src.drawer import Drawer
from src.bayesian_fitter import BayesianFitter
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.tigramite.tigramite import plotting as ti_plotting
from collections import defaultdict
from pprint import pprint

from src.tigramite.tigramite import pcmci
from src.tigramite.tigramite.independence_tests.chi2 import ChiSquare

class AssociationMiner():

    def __init__(self, event_preprocessor, frame, tau_max, pc_alpha, use_training=True):
        self.event_preprocessor:'GeneralProcessor' = event_preprocessor
        self.frame:'DataFrame' = frame
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.use_training = use_training
        self.mining_edges, self.mining_array, self.nor_mining_array = self.interaction_mining()

    def _normalize_temporal_array(self, target_array:'np.ndaray', threshold=0):
        new_array = target_array.copy()
        if len(new_array.shape) == 3 and new_array.shape[-1] == self.tau_max+1:
            new_array = sum([new_array[:,:,tau] for tau in range(1, self.tau_max+1)])
            new_array[new_array>threshold] = 1
        return new_array
    
    def interaction_mining(self, selected_links=None):
        # Auxillary variables
        n_vars = self.frame.n_vars
        int_dataframe = self.frame.training_dataframe if self.use_training else self.frame.testing_dataframe
        pcmci = PCMCI(
            dataframe=int_dataframe,
            cond_ind_test=ChiSquare()
        )

        # Return variable
        mining_edges = defaultdict(list)
        mining_array:'np.ndarray' = np.zeros((n_vars, n_vars, self.tau_max+1), dtype=np.int8)
        if not selected_links:
            selected_links = {n: [(i, -t) for i in range(n_vars) for \
                    t in range(1, self.tau_max+1)] for n in range(n_vars)}
        for outcome, candidate_causes in selected_links.items():
            for cause in candidate_causes:
                val, pval = pcmci.cond_ind_test.run_test(X=[cause], Y=[(outcome, 0)], Z=[], tau_max=self.tau_max)
                if pval <= self.pc_alpha:
                    mining_array[cause[0], outcome, abs(cause[1])] = 1
                    mining_edges[outcome].append((cause, val))
        nor_mining_array:'np.ndarray' = self._normalize_temporal_array(mining_array)
        
        # Sort the mined edges in the decreasing order of the edge strength
        for outcome, causes in mining_edges.items():
            mining_edges[outcome] = sorted(causes, key=lambda x: x[1], reverse=True)
        
        # Remove the statistical value information from the dict, and only store the edge info into the dict
        for outcome, causes in mining_edges.items():
            mining_edges[outcome] = [cause[0] for cause in causes]
        return mining_edges, mining_array, nor_mining_array