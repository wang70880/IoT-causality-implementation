import src.event_processing as evt_proc
import collections
import itertools
import numpy as np

class BackgroundGenerator():

    def __init__(self, dataset, event_processor, partition_config, tau_max) -> None:
        self.dataset = dataset
        self.partition_config = partition_config
        self.tau_max = tau_max
        self.event_processor = event_processor

        self.temporal_pair_dict, self.heuristic_temporal_pair_dict = self._temporal_pair_identification()
        self.spatial_pair_dict = self._spatial_pair_identification()
        self.functionality_pair_dict = self._functionality_pair_identification()

        self.correlation_dict = {
            'heuristic-temporal': self.heuristic_temporal_pair_dict,
            'temporal': self.temporal_pair_dict,
            'spatial': self.spatial_pair_dict,
            'functionality': self.functionality_pair_dict
        }
    
    def _temporal_pair_identification(self):
        """Identification of the temporal correlation pair in the following format.

            These pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).

            Note that many temporal pairs stem from the disorder of IoT logs. Therefore, we filter out pairs with low frequencies.

        Args:
            partition_config (int): _description_
            tau_max (int, optional): _description_. Defaults to 1.
        Returns:

        """
        temporal_pair_dict = {} # First index: frame_id. Second index: lag. Value: an integer array of shape num_attrs X num_attrs
        heuristic_temporal_pair_dict = {}
        for frame_id in range( len(self.event_processor.frame_dict.keys())): # Collect all lagged pairs in all frames
            temporal_pair_dict[frame_id] = {}
            heuristic_temporal_pair_dict[frame_id] = {}
            attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
            attr_sequence = self.event_processor.frame_dict[frame_id]['attr-sequence']
            state_sequence = self.event_processor.frame_dict[frame_id]['state-sequence']
            assert(len(attr_sequence) == len(state_sequence))
            num_event = len(attr_sequence)
            for event_id in range(num_event): # Count the occurrence of each attr pair and update the corresponding array
                for lag in range (1, self.tau_max + 1):
                    if event_id + lag >= num_event:
                        continue
                    temporal_pair_dict[frame_id][lag] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64) if lag not in temporal_pair_dict[frame_id].keys() else temporal_pair_dict[frame_id][lag]
                    heuristic_temporal_pair_dict[frame_id][lag] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64) if lag not in heuristic_temporal_pair_dict[frame_id].keys() else heuristic_temporal_pair_dict[frame_id][lag]
                    prior_attr = attr_sequence[event_id]; con_attr = attr_sequence[event_id + lag]
                    temporal_pair_dict[frame_id][lag][attr_names.index(prior_attr), attr_names.index(con_attr)] += 1
        for frame_id in range(len(self.event_processor.frame_dict.keys())): # Construct the array
            for lag in range (1, self.tau_max + 1):
                attr_array = temporal_pair_dict[frame_id][lag]
                heuristic_attr_array = heuristic_temporal_pair_dict[frame_id][lag]
                for idx, x in np.ndenumerate(attr_array):
                    """JC TODO: Explain here why we set 5 as the golden standard and why our heuristic standard is set to 4."""
                    heuristic_attr_array[idx] = 0 if x < 3 * self.partition_config else 1
                    attr_array[idx] = 0 if x < 4 * self.partition_config else 1  

        return temporal_pair_dict, heuristic_temporal_pair_dict
    
    def _testbed_area_infomation(self):
        area_list = []
        if self.dataset == 'hh101':
            # In the list, each element (which is also a list) represents a physical area and deployed devices in that area.
            area_list = [['T102', 'D002', 'M001', 'LS001', 'M010', 'LS010'], \
                         ['M011', 'LS011', 'D003', 'T103', 'MA015', 'LS015'], \
                         ['M009', 'LS009', 'MA014', 'LS014', 'M012', 'LS012'], \
                         ['D001', 'T101', 'T104', 'T105', 'M005', 'LS005', 'MA013', 'LS013', 'M008', 'LS008', 'M004', 'LS004'], \
                         ['MA016', 'LS016', 'M003', 'LS003', 'M002', 'LS002', 'M007', 'LS007', 'M006', 'LS006']]
            num_areas = 5
            area_connectivity_array = np.zeros(shape=(num_areas, num_areas), dtype=np.int64)
            for i in range(num_areas):
                area_connectivity_array[i, i] = 1
            area_connectivity_array[0, 1] = 1; area_connectivity_array[1, 2] = 1; area_connectivity_array[0, 3] = 1; area_connectivity_array[3, 4] = 1
            area_connectivity_array[1, 0] = 1; area_connectivity_array[2, 1] = 1; area_connectivity_array[3, 0] = 1; area_connectivity_array[4, 3] = 1
        return area_list, area_connectivity_array

    def _spatial_pair_identification(self):
        spatial_pair_dict = {} # First index: frame_id. Second index: lag. Value: an integer array of shape num_attrs * num_attrs
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        area_list, area_connectivity_array = self._testbed_area_infomation()
        spatial_array_list = []
        for lag in range (1, self.tau_max + 1): # The spatial array is only concerned with the lag and the testbed_area_list
            spatial_array_for_cur_lag = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
            # Given lags, compute the transitioned coherency, i.e., whether lag steps are enough to move from one area to another
            transition_area_array_for_cur_lag = area_connectivity_array if lag == 1 else np.linalg.matrix_power(area_connectivity_array, lag)
            for index, x in np.ndenumerate(transition_area_array_for_cur_lag):
                if x == 0:
                    continue
                area0 = area_list[index[0]]; area1 = area_list[index[1]]
                for element in itertools.product(area0, area1): # Get the cartesian product for devices in these two areas
                    if element[0] in attr_names and element[1] in attr_names:
                        spatial_array_for_cur_lag[attr_names.index(element[0]), attr_names.index(element[1])] = 1
            spatial_array_list.append(spatial_array_for_cur_lag)

        for frame_id in range(len(self.event_processor.frame_dict.keys())):
            spatial_pair_dict[frame_id] = {}
            for lag in range (1, self.tau_max + 1): 
                spatial_pair_dict[frame_id][lag] = spatial_array_list[lag-1]
        
        return spatial_pair_dict 

    def _functionality_pair_identification(self):
        functionality_pair_dict = {} # First index: keyword in {'activity', 'physics'}. Value: an integer array of shape num_attrs * num_attrs
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        functionality_pair_dict['activity'] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)
        functionality_pair_dict['physics'] = np.zeros(shape=(num_attrs, num_attrs), dtype=np.int64)

        for i in range(num_attrs):
            for j in range(num_attrs):
                i_activity_flag = attr_names[i].startswith(('M', 'D')); j_activity_flag = attr_names[j].startswith(('M', 'D')) # Identify attributes which are related with user activities 
                i_physics_flag = attr_names[i].startswith(('M', 'T', 'LS')); j_physics_flag = attr_names[i].startswith(('M', 'T', 'LS')) # Identify attributes which are related with physics
                functionality_pair_dict['activity'][i, j] = 1 if i_activity_flag and j_activity_flag else 0
                functionality_pair_dict['physics'][i, j] = 1 if i_physics_flag and j_physics_flag else 0
        return functionality_pair_dict

    def _print_pair_list(self, interested_array):
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        pair_list = []
        for index, x in np.ndenumerate(interested_array):
            if x == 1:
                pair_list.append((attr_names[index[0]], attr_names[index[1]]))
        print("Pair list with lens {}: {}".format(len(pair_list), pair_list))

    def print_benchmark_info(self,frame_id=0, tau=1, type = ''):
        """Print out the identified device correlations.

        Args:
            frame_id (int, optional): _description_. Defaults to 0.
            tau (int, optional): _description_. Defaults to 1.
            type (str, optional): 'temporal' or 'spatial' or 'functionality'
        """
        print("The {} corrleation dict for frame_id = {}, tau = {}: ".format(type, frame_id, tau))
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)
        if type != 'functionality':
            self._print_pair_list(self.correlation_dict[type][frame_id][tau])
        else:
            self._print_pair_list(self.correlation_dict[type]['activity'])
            self._print_pair_list(self.correlation_dict[type]['physics'])

    def apply_background_knowledge(self, selected_links=None, knowledge_type='', frame_id=0):
        assert(selected_links is not None)
        # print(attr_names)
        for tau in range(1, self.tau_max + 1):
            background_array = self.correlation_dict[knowledge_type][frame_id][tau] \
                    if knowledge_type != 'functionality' else self.correlation_dict[knowledge_type]['activity']
            for worker_index, link_dict in selected_links.items():
                # print("Job id: {}".format(worker_index))
                for outcome, cause_list in link_dict.items():
                    new_cause_list = []
                    for (cause, lag) in cause_list:
                        if abs(lag) == tau and background_array[cause, outcome] > 0:
                            new_cause_list.append((cause, lag))
                            # print(" Identified edge: ({},{}) -> {} / ({}, {}) -> {}".format(attr_names[cause], lag, attr_names[outcome], cause, lag, outcome))
                    selected_links[worker_index][outcome] = new_cause_list
        return selected_links

    def generate_candidate_interactions(self, apply_bk, frame_id, N):
        selected_links = {n: {m: [(i, -t) for i in range(N) for \
                t in range(1, self.tau_max + 1)] if m == n else [] for m in range(N)} for n in range(N)}
        if apply_bk >= 1:
            selected_links = self.apply_background_knowledge(selected_links, 'heuristic-temporal', frame_id)
        if apply_bk >= 2:
            selected_links = self.apply_background_knowledge(selected_links, 'spatial', frame_id)
            selected_links = self.apply_background_knowledge(selected_links, 'functionality', frame_id)
        return selected_links

if __name__ == '__main__':
    # Parameter setting
    dataset = 'hh101'
    partition_config = 10
    tau_max = 2; tau_min = 1
    event_preprocessor = evt_proc.Hprocessor(dataset)
    attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
    background_generator = BackgroundGenerator(dataset, event_preprocessor, partition_config, tau_max)
    background_generator.print_benchmark_info(frame_id=3, tau=2, type='spatial')
