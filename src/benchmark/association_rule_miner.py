from pymining import itemmining, assocrules, perftesting
from collections import defaultdict
import numpy as np
from pprint import pprint

class ARMMiner():

    def __init__(self, frame, tau_max, min_support=0.2, min_confidence=0.95, verbosity=0) -> None:
        self.tau_max = tau_max
        self.frame = frame
        self.var_names = self.frame.var_names; self.n_vars = len(self.var_names)
        self.min_support = min_support; self.min_confidence = min_confidence
        self.verbosity = verbosity

        self.transactions = self._transactions_generation()
        self.rule_dict, self.nor_mining_array = self._rule_mining()
        #self.mining_edges, self.mining_array, self.nor_mining_array = self.interaction_mining()
    
    def _encode_name(self, dev_name, dev_value):
        return '{}:{}'.format(dev_name, dev_value)
    
    def _decode_name(self, encoded_name:'str'):
        dev_infos = encoded_name.split(':')
        return (dev_infos[0], int(dev_infos[1]))
    
    def _transactions_generation(self):
        transactions = []
        training_events_states = self.frame.training_events_states
        for i in range(len(training_events_states)-self.tau_max-1):
            transaction = []
            for j in range(0, self.tau_max+1):
                (event, state) = training_events_states[i+j]
                transaction.append(self._encode_name(event.dev, event.value))
            transactions.append(tuple(transaction))
        if self.verbosity:
            print("[ARM] # transactions = {}".format(len(transactions)))
        return transactions

    def _rule_mining(self):
        """The rule dict is in the format (cond-dev,cond-state) -> (action-dev, action-state)"""
        relim_input = itemmining.get_relim_input(self.transactions)
        item_sets = itemmining.relim(relim_input, min_support=self.min_support)
        if self.verbosity:
            print("[ARM] # identified item sets: {}".format(len(list(item_sets.keys()))))
        rules = assocrules.mine_assoc_rules(item_sets, min_support=self.min_support, min_confidence=self.min_confidence)
        rules = sorted(rules, key=lambda x:x[3], reverse=True) # Sort the rule according to the confidence level
        if self.verbosity:
            print("[ARM] # identified rules: {}".format(len(rules)))

        nor_mining_array:'np.ndarray' = np.zeros((self.n_vars, self.n_vars), dtype=np.int8)
        rule_dict = {} # For each outcome, the dict stores all its prerequisites, i.e., the list of causes
        for rule in rules:
            preceding_device_infos = list(map(self._decode_name, rule[0]))
            consecutive_device_infos = list(map(self._decode_name, rule[1]))
            for p_info in preceding_device_infos:
                for c_info in consecutive_device_infos:
                    p_index = self.var_names.index(p_info[0]); c_index = self.var_names.index(c_info[0])
                    c_situ = (c_index, c_info[1]); p_situ = (p_index, p_info[1])
                    rule_dict[c_situ] = rule_dict[c_situ] if c_situ in rule_dict.keys() else []
                    rule_dict[c_situ].append(p_situ)
                    nor_mining_array[p_index, c_index] = 1
        return rule_dict, nor_mining_array
    
    def anomaly_detection(self, testing_event_states, testing_benign_dict):
        alarm_position_events = []
        last_system_states = testing_benign_dict[0][1][:self.n_vars]
        for evt_id, (event, states) in enumerate(testing_event_states):
            c_index = self.var_names.index(event.dev)
            c_situ = (c_index, event.value)
            if c_situ not in self.rule_dict.keys():
                # If no rule is found about the current device, it is assumed to be normal.
                last_system_states = states.copy()
                continue
            # Otherwise, we find a list of rules self.rule_dict[c_index], which set the current device as the consecutive device
            p_indices = [p_index[0] for p_index in self.rule_dict[c_situ]]
            p_hypo_values = [p_index[1] for p_index in self.rule_dict[c_situ]]
            p_observed_values = [last_system_states[index] for index in p_indices]
            assert(len(p_hypo_values)==len(p_observed_values))
            if len([i for i in range(len(p_hypo_values)) if p_hypo_values[i]!=p_observed_values[i]])>0:
                # If any rule is violated, raise an alarm here
                alarm_position_events.append((evt_id, event))
                last_system_states = testing_benign_dict[evt_id][1][:self.n_vars]
            else:
                # If all rule testings are passed, it is a normal event, update the lagged system states according to the event
                last_system_states = states.copy()
        return alarm_position_events