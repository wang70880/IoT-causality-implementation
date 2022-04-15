from src.tigramite.pcmci import PCMCI
from src.tigramite.causal_effects import CausalEffects
from src.tigramite.independence_tests.cmisymb import CMIsymb
from src.tigramite import plotting as tp
import numpy as np
from matplotlib import pyplot as plt

class CausalDiscovery():

    def __init__(self, dataframe,
                 tau_max,
                 cond_ind_test,
                 pc_alpha = 0,
                 verbosity=0):
        self.dataframe = dataframe
        self.tau_max = tau_max
        self.cond_ind_test = cond_ind_test
        self.pc_alpha = pc_alpha
        self.cond_ind_test.set_dataframe(self.dataframe)
        self.verbosity = verbosity
        self.var_names = self.dataframe.var_names
        self.T = self.dataframe.T
        self.N = self.dataframe.N
    
    def initiate_stablePC(self):
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=self.cond_ind_test
        )
        all_parents, results = pcmci.run_pc_stable(pc_alpha=self.pc_alpha, tau_max=self.tau_max, max_combinations=5)
        if self.verbosity > 0:
            print("# of records: {}".format(pcmci.T))
            pcmci.print_significant_links(
                graph = results['graph'],
                p_matrix = results['p_matrix'], 
                val_matrix = results['val_matrix'],
                conf_matrix = results['conf_matrix'],
                alpha_level = 0.1
            )
            print(all_parents)
        # NOTE: Followings are testing codes
        return all_parents, results

    def initiate_PCMCI(self):
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=self.cond_ind_test
        )
        results = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=0.1, alpha_level=0.05)
        if self.verbosity > 0:
            print("# of records: {}".format(pcmci.T))
            pcmci.print_significant_links(
                graph = results['graph'],
                p_matrix = results['p_matrix'], 
                val_matrix = results['val_matrix'],
                conf_matrix = results['conf_matrix'],
                alpha_level = 0.05
            )
        # NOTE: Followings are testing codes
        return results

class CausalInference():

    def __init__(self, dataframe, time_series_graph, X, Y, S):
        self.dataframe = dataframe
        self.time_series_graph = time_series_graph
        self.X = X
        self.Y = Y
        self.S = S
        self.causal_effects = CausalEffects(time_series_graph, graph_type='stationary_dag', X=self.X, Y=self.Y, S=self.S,
                           hidden_variables=None,
                           verbosity=0)
        
        self.opt = None
    
    def check__XYS_paths(self):
        newX, newY = self.causal_effects.check_XYS_paths()
        # self.X = newX
        # self.Y = newY

    def get_optimal_set(self):
        opt = self.causal_effects.get_optimal_set()
        # print("Oset = ", [(self.dataframe.var_names[v[0]], v[1]) for v in opt])
        self.opt = opt
    
    def check_optimality(self):
        optimality = self.causal_effects.check_optimality()
        return optimality

    def intervention_effect_prediction(self, intervention_data):
        """
        Calculate the effect on Y given the intervention of X (with conditioning set self.S and the optimal adjustment set self.opt)
        For instance of Z + instance of X (according to intervention_data), calculate the expectation value of Y.

        Note that we only consider the case of single outcome. Specifically, here we assume len(Y) == 1.

        Args:
            intervention_data (list[int]): The intervention value for all variables in X
        """
        self.check__XYS_paths()
        self.get_optimal_set()
        assert(self.check_optimality())
        assert(len(self.X) == len(intervention_data))
        assert(len(self.Y) == 1)

        array, xyz =  \
            self.dataframe.construct_array(X=self.X, Y=self.Y, # + self.Z, 
                                           Z=self.S,
                                           extraZ=self.opt,
                                           tau_max=self.causal_effects.tau_max,
                                           mask_type=None,
                                           cut_off='tau_max',
                                           verbosity=1)
        array = array.T # transpose from N * T matrix to T * N matrix
        X_indices = list(np.where(xyz==0)[0])
        for i in range(len(X_indices)): # For each intervention variable indexed by X_indices[i], its value is set to be intervention_data[i]
            array = array[ array[:, X_indices[i]] == intervention_data[i] ] # Select those columns which cause variables are fixed to the intervention value
        array = array.T # transpose back to N * T matrix
        predictor_indices =  list(np.where(xyz==3)[0]) \
                           + list(np.where(xyz==2)[0])
        predictor_array = array[predictor_indices, :].T
        # Given the intervention_data, further process the predictor_array
        target_array = array[np.where(xyz==1)[0][0], :].T
        fre_dict = {}
        row_count = 0
        for row in predictor_array:
            key = ''.join([str(x) for x in list(row)]) # The key is instance of Z + extraZ
            fre_dict[key] = [0, 0] if key not in fre_dict.keys() else fre_dict[key]
            fre_dict[key][0] += 1; fre_dict[key][1] += target_array[row_count] # Update the [instance #, summed y] pair
            row_count += 1
        frequency_table = list(fre_dict.values())
        weighted_frequency_table = [(entry[0] * 1.0 / row_count, entry[1] * 1.0 / entry[0]) for entry in frequency_table]
        return sum([val[0] * val[1] for val in weighted_frequency_table])

class CorrelationMiner():

    def __init__(self, dataframe, discovery_method='stable-pc'):
        self.dataframe = dataframe
        self.discovery_method = discovery_method 
        self.all_parents = None
        self.discovery_results = None

    def initiate_causal_discovery(self, tau_max, pc_alpha):
        all_parents = None; results = None
        causal_miner = CausalDiscovery(dataframe=self.dataframe, tau_max=tau_max, cond_ind_test=CMIsymb(
            significance='shuffle_test', n_symbs= None
            ), pc_alpha=pc_alpha, verbosity=1
            )
        if self.discovery_method == 'stable-pc':
            all_parents, results = causal_miner.initiate_stablePC()
            self.all_parents = all_parents
            self.discovery_results = results

    def initiate_causal_inference(self, tau_max):
        time_series_graph = None # First derive the DAG time_series graph
        # if self.discovery_method == 'stable-pc':
        #     assert((self.all_parents is not None) and (self.discovery_results is not None))
        #     time_series_graph = CausalEffects.get_graph_from_dict(self.all_parents, tau_max = tau_max)
        # assert(time_series_graph is not None)
        self.all_parents = {0: [(0, -1), (1, -1), (15, -1), (16, -1)], 1: [(1, -1), (31, -1), (13, -1), (4, -1), (18, -1)], 2: [(2, -1), (1, -1), (18, -1), (14, -1), (32, -1)], 3: [(3, -1), (17, -1), (4, -1)], 4: [(4, -1), (3, -1), (17, -1), (6, -1)], 5: [(5, -1), (17, -1), (6, -1), (31, -1)], 6: [(6, -1), (9, -1), (14, -1), (10, -1)], 7: [(7, -1), (23, -1), (16, -1)], 8: [(8, -1), (24, -1)], 9: [(9, -1)], 10: [(10, -1), (9, -1), (15, -1), (13, -1), (1, -1)], 11: [(11, -1), (2, -1), (27, -1), (14, -1), (16, -1)], 12: [(12, -1), (15, -1), (13, -1), (1, -1)], 13: [(13, -1), (10, -1), (14, -1)], 14: [(14, -1), (1, -1), (9, -1), (11, -1), (31, -1)], 15: [(15, -1), (10, -1), (1, -1), (16, -1)], 16: [(16, -1), (33, -1), (28, -1)], 17: [(17, -1), (4, -1)], 18: [(18, -1), (1, -1), (27, -1), (30, -1), (2, -1), (16, -1)], 19: [(19, -1), (21, -1), (25, -1), (32, -1), (27, -1), (13, -1)], 20: [(20, -1), (7, -1), (25, -1), (17, -1), (23, -1), (19, -1), (1, -1), (32, -1)], 21: [(21, -1), (19, -1), (25, -1), (32, -1), (27, -1)], 22: [(1, -1), (9, -1), (22, -1), (25, -1), (18, -1), (16, -1), (12, -1), (21, -1), (28, -1)], 23: [(23, -1), (7, -1), (20, -1)], 24: [(24, -1), (8, -1)], 25: [(25, -1), (1, -1), (20, -1), (29, -1), (27, -1), (19, -1), (18, -1), (21, -1), (26, -1), (28, -1), (32, -1), (22, -1), (7, -1), (23, -1)], 26: [(26, -1), (25, -1), (20, -1), (27, -1), (18, -1)], 27: [(27, -1), (1, -1), (18, -1), (25, -1), (0, -1), (11, -1)], 28: [(28, -1), (1, -1), (25, -1), (4, -1), (27, -1)], 29: [(25, -1), (17, -1), (1, -1), (15, -1), (27, -1), (28, -1), (26, -1), (34, -1), (29, -1)], 30: [(30, -1), (25, -1), (1, -1), (15, -1)], 31: [(31, -1)], 32: [(32, -1), (31, -1)], 33: [(33, -1), (34, -1)], 34: [(34, -1), (33, -1)]} # NOTE: This is the testing code. 
        time_series_graph = CausalEffects.get_graph_from_dict(self.all_parents, tau_max = tau_max)
        effects_dict = {} # effects_dict[cause_attr][outcome_attr][intervention] stores the corresponding effect.
        for key in self.all_parents.keys(): # Traverse each causal-outcome pair in the graph and estimate the causal effects.
            outcome_attr = (key, 0)
            cause_attrs = self.all_parents[key]
            for cause_attr in cause_attrs:
                causal_inferencer = CausalInference(dataframe=self.dataframe, \
                    time_series_graph=time_series_graph, X=[cause_attr], Y=[outcome_attr], S=[]) # We did not test conditional ATEs; Moreover, we assume that X and Y only contains single attribute
                causal_inferencer.check__XYS_paths()
                assert(causal_inferencer.check_optimality()) # For DAGs, the optimality always holds.
                causal_inferencer.get_optimal_set()
                effects_dict[cause_attr] = {} if cause_attr not in effects_dict.keys() else effects_dict[cause_attr]
                effects_dict[cause_attr][outcome_attr] = {} if outcome_attr not in effects_dict[cause_attr].keys() else effects_dict[cause_attr][outcome_attr]
                effects_dict[cause_attr][outcome_attr][1] = causal_inferencer.intervention_effect_prediction([1])
                effects_dict[cause_attr][outcome_attr][0] = causal_inferencer.intervention_effect_prediction([0])
        print(effects_dict)
        return effects_dict