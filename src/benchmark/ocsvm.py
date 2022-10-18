import sys
sys.path.append('../src')
import numpy as np
from src.genetic_type import AttrEvent, DataFrame
from sklearn.svm import OneClassSVM

class OCSVMer():
    
    def __init__(self, frame, window_size, verbosity=0):
        self.frame:'DataFrame' = frame
        self.window_size = window_size
        self.verbosity = verbosity
        self.ocsvm = self._initialize_svm()
    
    def _initialize_svm(self):
        training_event_states = self.frame.training_events_states
        ocsvm = OneClassSVM(kernel='rbf', gamma='auto')
        training_vectors = self.construct_vectors(training_event_states)
        ocsvm.fit(training_vectors)
        return ocsvm
    
    def construct_vectors(self, event_states:'tuple(AttrEvent, np.ndarray)'):
        n_vars = len(event_states[0][1])
        vectors = []
        cur_vector = [0]*n_vars*self.window_size
        for i, (event, state) in enumerate(event_states):
            cur_vector = [*list(state), *cur_vector[:-n_vars]]
            vectors.append(cur_vector)
        return np.array(vectors)
    
    def contextual_anomaly_detection(self, testing_event_states):
        testing_event_states = self.frame.testing_events_states
        testing_vectors = self.construct_vectors(testing_event_states)
        prediction_result = self.ocsvm.predict(testing_vectors)
        outlier_positiosn = [pos for pos in range(len(prediction_result)) if prediction_result[pos]==-1]
        alarm_position_events = [(pos, testing_event_states[pos][0]) for pos in outlier_positiosn]
        return alarm_position_events