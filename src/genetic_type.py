from numpy import ndarray
from src.tigramite.tigramite import data_processing as pp
from collections import defaultdict

class AttrEvent():

    def __init__(self, date, time, dev, attr, value):
        # Event in the format of [date, time, dev, dev_attr, value]
        self.date:'str' = date; self.time:'str' = time; self.dev:'str' = dev
        self.attr:'str' = attr; self.value:'int' = value
    
    def __str__(self) -> str:
        return ' '.join([self.date, self.time, self.dev, self.attr, str(self.value)])

class DataFrame():

    def __init__(self, id, var_names, n_events, n_days) -> None:
        self.id = id; self.n_events = n_events; self.n_days = n_days
        self.var_names = var_names; self.n_vars = len(self.var_names)
        self.training_events_states:'list[tuple(AttrEvent, ndarray)]' = None
        self.testing_events_states:'list[tuple(AttrEvent, ndarray)]' = None
        self.name_device_dict:'dict[DevAttribute]' = None # The str-DevAttribute dict using the attr name as the dict key
        self.index_device_dict:'dict[DevAttribute]' = None # The str-DevAttribute dict using the attr index as the dict key
        self.training_dataframe:'pp.DataFrame' = None
        self.testing_dataframe:'pp.DataFrame' = None
    
    def set_training_data(self, events_states, dataframe):
        self.training_events_states = events_states
        self.training_dataframe:'pp.DataFrame' = dataframe
        assert(self.training_dataframe.T == len(self.training_events_states))
        assert(self.training_dataframe.N == self.n_vars)

    def set_testing_data(self, events_states, dataframe):
        self.testing_events_states = events_states
        self.testing_dataframe:'pp.DataFrame' = dataframe
        assert(self.testing_dataframe.T == len(self.testing_events_states))
        assert(self.training_dataframe.N == self.n_vars)
    
    def set_device_info(self, name_device_dict, index_device_dict):
        self.name_device_dict:'dict[DevAttribute]' = name_device_dict
        self.index_device_dict:'dict[DevAttribute]' = index_device_dict
        assert(set(self.name_device_dict.keys()) == set(self.var_names) == set([dev.name for dev in self.index_device_dict.values()]))

class DevAttribute():

    def __init__(self, attr_name=None, attr_index=None, lag=0):
        self.index = attr_index
        self.name = attr_name
        self.lag = abs(lag)