from pymining import itemmining, assocrules, perftesting
from collections import defaultdict
import numpy as np
from pprint import pprint

class ARMMiner():

    def __init__(self, auxillary_bayesian_fitter, frame, min_support=0.2, min_confidence=0.5, verbosity=0) -> None:
        self.auxillary_bayesian_fitter = auxillary_bayesian_fitter
        self.tau_max = self.auxillary_bayesian_fitter.tau_max
        self.n_vars = self.auxillary_bayesian_fitter.n_vars
        self.frame = frame
        self.min_support = min_support; self.min_confidence = min_confidence
        self.verbosity = verbosity

        self.transactions = self._transactions_generation()
        self.mining_edges, self.mining_array, self.nor_mining_array = self.interaction_mining()
    
    def _transactions_generation(self):
        transactions = []
        training_events_states = self.frame.training_events_states
        for i in range(len(training_events_states)-self.tau_max-1):
            transaction = []
            for j in range(0, self.tau_max+1):
                (event, state) = training_events_states[i+j]
                transaction.append(self.auxillary_bayesian_fitter._lag_name(event.dev, j-self.tau_max))
            transactions.append(tuple(transaction))
        if self.verbosity:
            print("[ARM] # transactions = {}".format(len(transactions)))
        return transactions

    def _normalize_temporal_array(self, target_array:'np.ndaray', threshold=0):
        new_array = target_array.copy()
        if len(new_array.shape) == 3 and new_array.shape[-1] == self.tau_max+1:
            new_array = sum([new_array[:,:,tau] for tau in range(1, self.tau_max+1)])
            new_array[new_array>threshold] = 1
        return new_array

    def interaction_mining(self):
        expanded_var_names = self.auxillary_bayesian_fitter.expanded_var_names
        relim_input = itemmining.get_relim_input(self.transactions)
        item_sets = itemmining.relim(relim_input, min_support=self.min_support)
        if self.verbosity:
            print("[ARM] # identified item sets: {}".format(len(list(item_sets.keys()))))
        rules = assocrules.mine_assoc_rules(item_sets, min_support=self.min_support, min_confidence=self.min_confidence)
        rules = sorted(rules, key=lambda x:x[3], reverse=True) # Sort the rule according to the confidence level
        if self.verbosity:
            print("[ARM] # identified rules: {}".format(len(rules)))
        mining_edges = defaultdict(list)
        mining_array:'np.ndarray' = np.zeros((self.n_vars, self.n_vars, self.tau_max+1), dtype=np.int8)
        for rule in rules:
            preceding_devices = rule[0]
            consecutive_devices = rule[1]
            interactions = [(x, y) for x in preceding_devices if expanded_var_names.index(x)>=self.n_vars\
                                    for y in consecutive_devices if expanded_var_names.index(y)<self.n_vars]
            for (cause, outcome) in interactions:
                cause_index = int(expanded_var_names.index(cause)%self.n_vars); cause_lag = int(expanded_var_names.index(cause)/self.n_vars)
                outcome_index = expanded_var_names.index(outcome)
                mining_edges[outcome_index].append((cause_index, -cause_lag))
                mining_array[cause_index, outcome_index, cause_lag] = 1
        nor_mining_array:'np.ndarray' = self._normalize_temporal_array(mining_array)
        return mining_edges, mining_array, nor_mining_array