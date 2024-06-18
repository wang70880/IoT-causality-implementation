import pandas as pd
import numpy as np
import itertools
import functools
from itertools import permutations
from pprint import pprint
from collections import defaultdict
from src.genetic_type import DataFrame, AttrEvent, DevAttribute

class MarkovMiner():

    def __init__(self, frame, tau_max, sig_level, training_kmax=5) -> None:
        self.frame:'DataFrame' = frame
        self.tau_max = tau_max
        self.sig_level = sig_level
        self.var_names = self.frame.var_names
        self.training_kmax = training_kmax
        self.state_space = self._get_state_space()
        self.transition_df = self._generate_transition_df()
        self.score_threshold_dict = self.train_transition_model()
    
    def _state_name(self,state:'tuple'):
        return '{}{}'.format(state[0], state[1])

    def _states_name(self, states:'tuple'):
        return ','.join(states)

    def _get_state_space(self):
        """
        Suppose there are two devices a and b, the state space is [a0, a1, b0, b1]
        """
        value_range = [0, 1]
        state_space = [(x, y) for x in self.var_names for y in value_range]
        str_state_space = list(map(self._state_name, state_space))
        return str_state_space
    
    def _generate_transition_df(self):
        """
        The generated dataframe is in the following format.
        Row: a string with string index like 'Light1, Lock0'. The length of the index equals to tau_max
        Col: a string with index like 'Light1'. The length is simply 1.
        Therefore, each value in the final transition matrix represents the probability that P(Col|Row)
        """
        row_names = self.state_space.copy()
        cur_row_names = [[x] for x in self.state_space]
        for i in range(self.tau_max-1):
            cur_row_names = [[*cur_comb, y] for cur_comb in cur_row_names for y in row_names]
        row_names = [self._states_name(tuple(x)) for x in cur_row_names]
        col_names = self.state_space
        #print("Size of the transition matrix: {} X {}".format(len(row_names), len(col_names)))
        transition_matrix = np.zeros(shape=(len(row_names), len(col_names)))
        transition_df = pd.DataFrame(data=transition_matrix, index=row_names, columns=col_names)
        return transition_df
    
    def train_transition_model(self):
        training_event_states = self.frame.training_events_states
        history_states = ()
        for i, (event, states) in enumerate(training_event_states):
            cur_state = self._state_name((event.dev, event.value))
            if i < self.tau_max:
                history_states = (*history_states[-(self.tau_max-1):], cur_state)
                continue
            # Update the transiiton probability here
            self.transition_df.loc[self._states_name(history_states), cur_state] += 1
            # Always keep the latest lagged state in the end of the history states
            history_states = (*history_states[-(self.tau_max-1):], cur_state)
        # Finally, normalize the transition matrix
        self.transition_df = self.transition_df.div(self.transition_df.sum(axis=1), axis=0).fillna(0)

        # Now derive the score threshold for each k, which is the length of the sequence
        anomaly_scores_dict = defaultdict(list)
        score_threshold_dict = {}
        transition_probs = []
        history_states = ()
        for i, (event, states) in enumerate(training_event_states):
            cur_state = self._state_name((event.dev, event.value))
            if i < self.tau_max:
                history_states = (*history_states[-(self.tau_max-1):], cur_state)
                continue
            transition_prob = self.transition_df.loc[self._states_name(history_states), cur_state]
            if len(transition_probs) < self.training_kmax:
                transition_probs.append(transition_prob)
                history_states = (*history_states[-(self.tau_max-1):], cur_state)
                continue
            else:
                transition_probs = [*transition_probs[:-1], transition_prob]
            for l in range(1, self.training_kmax+1):
                if l == 1:
                    seq_k_prob = transition_probs[0]
                else:
                    seq_k_prob = functools.reduce(lambda a, b: a*b, transition_probs[:l])
                seq_k_anomaly_score = 1 - seq_k_prob
                anomaly_scores_dict[l].append(seq_k_anomaly_score)
            #print(anomaly_score)
            history_states = (*history_states[-(self.tau_max-1):], cur_state)
        
        #print(anomaly_scores)
        for l, anomaly_scores in anomaly_scores_dict.items():
            score_threshold_dict[l] = np.percentile(np.array(anomaly_scores), self.sig_level * 100)
        #print("Score threshold for the markov model:")
        #pprint(score_threshold_dict)
        return score_threshold_dict
    
    def anomaly_detection(self, testing_event_states, testing_benign_dict, kmax, verbosity=0):
        assert(1<=kmax<=self.training_kmax)
        score_threshold = self.score_threshold_dict[kmax]
        alarm_position_events = []
        # Initial history states
        history_states = [self._state_name((event.dev, event.value)) for event in testing_benign_dict[0][2]]
        transition_probs = []
        for evt_id, (event, states) in enumerate(testing_event_states):
            cur_state = self._state_name((event.dev, event.value))
            transition_prob  = self.transition_df.loc[self._states_name(history_states), cur_state]
            if len(transition_probs) < kmax:
                transition_probs.append(transition_prob)
            else:
                transition_probs = [*transition_probs[:-1], transition_prob]
            if len(transition_probs) == kmax:
                seq_k_prob = transition_probs[0] if kmax==1 else functools.reduce(lambda a, b: a*b, transition_probs)
                anomaly_score = 1. - seq_k_prob
                # Record the whole anomalous sequence which ends at current sequence
                # JC NOTE: Here we did not use the score threshold, and 0 instead.
                if anomaly_score == 1.0:
                #if anomaly_score >= score_threshold:
                    start_position = evt_id-(kmax-1)
                    for i in range(kmax):
                        alarm_position_events.append((start_position+i, testing_event_states[start_position+i][0]))
            # Update the history states for calculating the next transition probability
            history_states = (*history_states[-(self.tau_max-1):], cur_state)
        return alarm_position_events