from tabulate import tabulate
from numpy import indices, ndarray
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
from pprint import pprint
from statistics import mean
from collections import defaultdict

from src.genetic_type import DevAttribute, AttrEvent, DataFrame
from src.bayesian_fitter import BayesianFitter

NORMAL = 0
TYPE1_ANOMALY = 1
TYPE2_ANOMALY = 2

class PhantomStateMachine():

    def __init__(self, var_names, lagged_var_names, tau_max) -> None:
        self.var_names = var_names; self.n_vars = len(var_names)
        self.lagged_var_names = lagged_var_names; self.n_lagged_vars = len(self.lagged_var_names)
        self.tau_max = tau_max
        # Suppose there is n attributes and time lag is l, the length of phantom_states is n*l
        # The latest variable (i.e., lag=1) has the smallest index
        self.phantom_states = [0] * (self.n_vars * self.tau_max)
        assert(len(self.phantom_states) == self.n_lagged_vars)
    
    def equal(self, test_states):
        return len(test_states)==len(self.phantom_states) and all([test_states[i]==self.phantom_states[i] for i in range(len(test_states))])

    def set_latest_states(self, state_vector:'list[int]'):
        """
        Update the latest history state vectors (i.e., the vector for lag = -1).
        Args:
            state_vector (list[int]): The state vector to be renewed.
        """
        assert(len(state_vector) == self.n_vars)
        self.phantom_states = [*state_vector, *self.phantom_states[:-self.n_vars]]
    
    def get_lagged_states(self, lag=1):
        """
        Get the history device states with respect to the time lag.
        Args:
            lag (int, optional): The time lag. Defaults to 1.
        Returns:
            sliced_phantom_states: The sliced phantom states
        """
        assert(lag > 0 and lag <= self.tau_max)
        return self.phantom_states[self.n_vars*(lag-1):self.n_vars*lag]

    def get_indices_states(self, indices:'list[int]'):
        # The parameter indices is the variable index in the bayesian_fitter.expanded_var_names
        # However, since bayesian_fitter.expanded_var_names is of size (tau_max+1)*n_vars, here the index in the phantom machien should substract n_vars
        return [self.phantom_states[x] for x in indices]

    def update(self, event:'AttrEvent'):
        """
        Update the phantom state machine according to the newly received event.
        """
        renewed_state_vector = self.get_lagged_states(lag=1).copy()
        renewed_state_vector[self.var_names.index(event.dev)] = event.value
        self.set_latest_states(renewed_state_vector)
    
    def flush(self, laggged_states:'list[int]'):
        #print("laggged_states, self.phantom_states: {} v.s. {}".format(laggged_states, self.phantom_states))
        #print("LEN of the above sequence: {} v.s. {}".format(len(laggged_states), len(self.phantom_states)))
        assert(len(laggged_states) == len(self.phantom_states))
        self.phantom_states = laggged_states.copy()

    def __str__(self):
        return tabulate(\
            list(zip(self.lagged_var_names, self.phantom_states)),\
            headers= ['Attr', 'State'])

class InteractionChain():

    def __init__(self, anomaly_flag, n_vars, expanded_var_names, expanded_causal_graph, expanded_attr_index) -> None:
        self.anomaly_flag = anomaly_flag
        self.n_vars = n_vars; self.expanded_var_names = expanded_var_names; self.expanded_causal_graph = expanded_causal_graph
        self.attr_index_chain = [expanded_attr_index - n_vars] # Adjust to the lagged attribute
        self.header_attr_index = self.attr_index_chain[-1]
    
    def match(self, expanded_attr_index:'int'):
        return self.expanded_causal_graph[self.header_attr_index, expanded_attr_index] > 0
    
    def get_header_attr(self):
        return self.expanded_var_names[self.header_attr_index]
    
    def update(self, expanded_attr_index:'int'):
        assert(self.match(expanded_attr_index))
        self.attr_index_chain.append(expanded_attr_index)
        self.attr_index_chain = [x - self.n_vars for x in self.attr_index_chain]
        self.header_attr_index = self.attr_index_chain[-1]
    
    def get_length(self):
        return len(self.attr_index_chain)

    def __str__(self):
        return "header_attr = {}, anomaly_flag = {}, len(chains) = {}\n"\
              .format(self.expanded_var_names[self.header_attr_index], self.anomaly_flag, len(self.attr_index_chain))

class ChainManager():
    
    def __init__(self, var_names, expanded_var_names, expanded_causal_graph) -> None:
        self.chain_pool = {}; self.n_chains = 0
        self.current_chain:'InteractionChain' = None
        self.var_names = var_names; self.n_vars = len(self.var_names)
        self.expanded_var_names = expanded_var_names; self.n_expanded_vars = len(self.expanded_var_names)
        self.expanded_causal_graph = expanded_causal_graph
        self.anomalous_interaction_dict = {}
    
    def match(self, expanded_attr_index:'int'):
        return self.current_chain is not None and self.current_chain.match(expanded_attr_index)
    
    def update(self, expanded_attr_index:'int'):
        assert(self.match(expanded_attr_index))
        self.current_chain.update(expanded_attr_index)
    
    def create(self, evt_id:'int', expanded_attr_index:'int', anomaly_flag):
        chain_id = evt_id
        self.chain_pool[chain_id] = InteractionChain(anomaly_flag, self.n_vars, self.expanded_var_names,\
                                    self.expanded_causal_graph, expanded_attr_index)
        self.n_chains += 1
        self.current_chain = self.chain_pool[chain_id]
        return chain_id

    def print_chains(self):
        print("Current chain stack with {} chains.".format(len(self.chain_pool.keys())))
        for index, chain in enumerate(self.chain_pool):
            print(" * Chain {}: {}".format(index, chain))

    def is_tracking_normal_chain(self):
        return self.current_chain.anomaly_flag == NORMAL

    def current_chain_length(self):
        return self.current_chain.get_length()

class SecurityGuard():

    def __init__(self, frame=None, bayesian_fitter:'BayesianFitter'=None, sig_level=0.8, verbosity=0) -> None:
        self.frame:'DataFrame' = frame
        self.bayesian_fitter:'BayesianFitter' = bayesian_fitter
        self.var_names = self.bayesian_fitter.var_names; self.tau_max = self.bayesian_fitter.tau_max
        self.verbosity = verbosity
        # Phantom state machine
        self.phantom_state_machine = PhantomStateMachine(bayesian_fitter.var_names,\
            bayesian_fitter.expanded_var_names[len(self.var_names):len(self.var_names)*(self.tau_max+1)], bayesian_fitter.tau_max)

        # Anomaly analyzer
        self.violation_dict = defaultdict(list)
        self.tp_debugging_dict = defaultdict(list)
        self.fn_debugging_dict = defaultdict(list)
        self.fp_debugging_dict = defaultdict(list)
        # The score threshold
        self.training_anomaly_scores, self.score_threshold = self._compute_anomaly_score_cutoff(sig_level=sig_level)
    
    def initialize_phantom_machine(self):
        latest_event_states = [self.frame.training_events_states[-self.tau_max+i] for i in range(0, self.tau_max)]
        machine_initial_states = [event_state[1] for event_state in latest_event_states]
        assert(len(machine_initial_states)==self.tau_max)
        # Initialize the phantom state machine
        for state_vector in machine_initial_states:
            self.phantom_state_machine.set_latest_states(state_vector)

    def contextual_anomaly_detection(self, testing_event_states, testing_benign_dict):
        alarm_position_events = []
        self.initialize_phantom_machine()
        for evt_id, (event, states) in enumerate(testing_event_states):
            anomaly_score = self._compute_event_anomaly_score(event, self.phantom_state_machine)
            if anomaly_score >= self.score_threshold:
                # Raise an alarm and report to users
                alarm_position_events.append((evt_id, event))
            # Restore the lagged system state (the lagged system states after the latest benign event happens)
            lagged_system_states = testing_benign_dict[evt_id][1]
            self.phantom_state_machine.flush(lagged_system_states)
        return alarm_position_events

    def kmax_anomaly_detection(self, testing_event_states, testing_benign_dict, kmax, debugging_dict=None):
        """
        This function is responsible for determining the contextual anomaly or collective anomaly.
        Return values:
        alarm_position_chains (list[(int, list[AttrEvent])])
        """
        alarm_start_positions = []
        alarm_chains = []
        self.initialize_phantom_machine()
        anomaly_chain = []
        print("Initializing kmax anomaly detection.")
        cur_tracking_contextual_id = 0
        for evt_id, (event, states) in enumerate(testing_event_states):

            anomaly_score = self._compute_event_anomaly_score(event, self.phantom_state_machine)
            contextual_anomaly_flag = anomaly_score >= self.score_threshold

            is_tracking = True if 0<len(anomaly_chain)<kmax else False

            #if evt_id in list(debugging_dict.keys()):
            #    print("     Contextual anomaly at position {}, anomaly flag: {}, tracking state: {}".format(evt_id, contextual_anomaly_flag, is_tracking))
            #    cur_tracking_contextual_id = evt_id
            #elif evt_id-cur_tracking_contextual_id+1 < debugging_dict[cur_tracking_contextual_id]:
            #    print("         Collective anomaly at position {}, anomaly flag: {}, tracking state: {}".format(evt_id, contextual_anomaly_flag, is_tracking))
            if is_tracking and anomaly_score < self.score_threshold:
                anomaly_chain.append((evt_id, event))
            elif is_tracking and anomaly_score >= self.score_threshold:
                #print("     Chain breaks with length: {}".format(len(anomaly_chain)))
                alarm_chains.append(anomaly_chain.copy())
                anomaly_chain = []
                latest_stable_lagged_states = testing_benign_dict[evt_id][1]
                self.phantom_state_machine.flush(latest_stable_lagged_states)
            elif len(anomaly_chain)==0 and anomaly_score < self.score_threshold:
                pass
            elif len(anomaly_chain)==0 and anomaly_score >= self.score_threshold:
                # Record the position of the contextual anomaly, and start maintaining the anomaly chain
                alarm_start_positions.append(evt_id)
                anomaly_chain = [(evt_id, event)]

            self.phantom_state_machine.set_latest_states(states)

            if (len(anomaly_chain) == kmax) or (len(anomaly_chain)>0 and evt_id == len(testing_event_states)-1):
                alarm_chains.append(anomaly_chain.copy())
                anomaly_chain = []
                latest_stable_lagged_states = testing_benign_dict[evt_id][1]
                self.phantom_state_machine.flush(latest_stable_lagged_states)

        #print("TP, FP = {}, {}".format(tp, fp))
        assert(len(alarm_start_positions) == len(alarm_chains))
        return alarm_chains

    def analyze_detection_results(self):
        tps = sum([len(tp_list) for tp_list in self.tp_debugging_dict.values()])
        fps = sum([len(fp_list) for fp_list in self.fp_debugging_dict.values()])
        fns = sum([len(fn_list) for fn_list in self.fn_debugging_dict.values()])
        sorted_tps = dict(sorted(self.tp_debugging_dict.items(), key=lambda x: len(x[1]), reverse=True))
        sorted_fps = dict(sorted(self.fp_debugging_dict.items(), key=lambda x: len(x[1]), reverse=True))
        sorted_fns = dict(sorted(self.fn_debugging_dict.items(), key=lambda x: len(x[1]), reverse=True))
        print("# testing events = {}".format(len(self.frame.testing_events_states)))
        print("# FPs, FNs, TPs = {}, {}, {} *************".format(fps, fns, tps))
        for fp_case in sorted_fps.keys():
            n_cases = len(sorted_fps[fp_case])
            print("{} ---- Occurrence: {}, Avg score: {}".format(fp_case, n_cases, mean([sorted_fps[fp_case][x][1] for x in range(n_cases)])))
        return tps, fps, fns

    def _compute_event_anomaly_score(self, event:'AttrEvent', phantom_state_machine:'PhantomStateMachine', verbosity=0):
        prob_scheme = True
        # Return variables
        anomaly_score = 1.
        # 1. Get the list of parents for event.dev, and fetch their states from the phantom state machine
        try:
            parents = self.bayesian_fitter.model.get_parents(event.dev)
        except: # If current event device has no parents, just return 
            return anomaly_score
        indices = [self.bayesian_fitter.expanded_var_names.index(parent)-self.bayesian_fitter.n_vars for parent in parents]
        parent_states = phantom_state_machine.get_indices_states(indices)
        parent_states:'list[tuple(str, int)]' = list(zip(parents, parent_states))
        # 2. Estimate the anomaly score for current event
        if prob_scheme:
            cond_prob = self.bayesian_fitter.estimate_cond_probability(event, parent_states)
            anomaly_score = 1.-cond_prob
        else:
            act_event = AttrEvent(event.date, event.time, event.dev, event.attr, 1)
            act_prob = self.bayesian_fitter.estimate_cond_probability(act_event, parent_states)
            anomaly_score = abs(event.value - act_prob)
        if verbosity:
            print("Event: {}\n\
                   Phantom state machine:\n{}\n\
                   parent states: {}\n\
                   anomaly score: {}\n".format(event, phantom_state_machine, parent_states, anomaly_score)
                  )
        return anomaly_score

    def _compute_anomaly_score_cutoff(self, sig_level):
        # Return variables
        self.score_threshold = 0.
        anomaly_scores = []
        # Auxillary variables
        training_events_states:'list[tuple(AttrEvent, ndarray)]' = self.frame.training_events_states
        training_phantom_machine:'PhantomStateMachine' = PhantomStateMachine(self.var_names,\
                                self.bayesian_fitter.expanded_var_names[len(self.var_names):len(self.var_names)*(self.tau_max+1)], self.tau_max)

        for index, (event, states) in enumerate(training_events_states):
            # An initialization is needed using the first tau_max events.
            if index < self.tau_max:
                training_phantom_machine.set_latest_states(states)
                continue
            anomaly_score = self._compute_event_anomaly_score(event, training_phantom_machine, verbosity=0)
            anomaly_scores.append(anomaly_score)
            training_phantom_machine.set_latest_states(states)

        score_threshold = np.percentile(np.array(anomaly_scores), sig_level * 100)
        # Draw the score distribution graph
        sns.displot(anomaly_scores, kde=False, color='red', bins=1000)
        plt.axvline(x=score_threshold)
        plt.title('Training score distribution')
        plt.xlabel('Scores')
        plt.ylabel('Occurrences')
        plt.savefig("temp/image/training-score-distribution-{}.pdf".format(int(sig_level*100)))
        plt.close('all')
        #print("Score threshold for sig-level={}: {}".format(sig_level, score_threshold))
        return anomaly_scores, score_threshold
