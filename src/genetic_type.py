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

    def __init__(self, id, var_names, attr_names, n_events, n_days) -> None:
        self.id = id; self.n_events = n_events; self.n_days = n_days
        self.var_names = var_names; self.n_vars = len(self.var_names)
        self.attr_names = attr_names; self.n_attrs = len(self.attr_names)
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
    
    def get_cpt_and_ate(self, prior_vars:'list', latter_vars:'list', cond_vars:'list', tau_max, use_training=True):
        assert(len(latter_vars)==len(prior_vars)==1) # Currently only support the debugging for single x and y, i.e., P(y|x)
        assert(len(cond_vars)==0) # Currently does not support additional conditioning variables
        int_dataframe = self.training_dataframe if use_training else self.testing_dataframe
        assert(int_dataframe is not None)
        array, xyz = int_dataframe.construct_array(X=prior_vars, Y=latter_vars, Z=cond_vars, tau_max=tau_max, do_checks=True)
        n_cols = array.shape[1]
        xy_00 = len([col for col in range(n_cols) if array[0,col]+array[1,col]==0])
        xy_11 = len([col for col in range(n_cols) if array[0,col]+array[1,col]==2])
        xy_01 = len([col for col in range(n_cols) if array[0,col]==0 and array[1,col]==1])
        xy_10 = len([col for col in range(n_cols) if array[0,col]==1 and array[1,col]==0])
        cpt = {0: [round(xy_00*1./(xy_00+xy_01), 3), round(xy_01*1./(xy_00+xy_01), 3)], 1:[round(xy_10*1./(xy_10+xy_11), 3), round(xy_11*1./(xy_10+xy_11), 3)]}
        ate = cpt[1][1]-cpt[0][1]
        return cpt, ate

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

    def __init__(self, name, index, attr=None, location=None, lag=0):
        self.index = index 
        self.name = name 
        self.attr = attr
        self.location = location
        self.lag = abs(lag)