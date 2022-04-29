from event_processing import Hprocessor
import collections
import itertools
import numpy as np

class Evaluator():

    def __init__(self, dataset, partition_config, tau_max) -> None:
        self.dataset = dataset
        self.partition_config = partition_config
        self.tau_max = tau_max
        self.event_processor = Hprocessor(dataset)

        self.temporal_pair_dict = None
        self.spatial_pair_dict = None
        self.user_correlation_dict = None
        self.physical_correlation_dict = None
        self.automation_correlation_dict = None

        self.event_processor.initiate_data_preprocessing(partition_config=self.partition_config)
        self._temporal_pair_identification()
        self._spatial_pair_identification()
        self.construct_user_correlation_benchmark()

        self.correlation_dict = {
            0: self.temporal_pair_dict,
            1: self.spatial_pair_dict,
            2: self.user_correlation_dict,
            3: self.physical_correlation_dict,
            4: self.automation_correlation_dict
        }
    
    def _temporal_pair_identification(self):
        """Identification of the temporal correlation pair in the following format.

            These pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).

            Note that many temporal pairs stem from the disorder of IoT logs. Therefore, we filter out pairs with low frequencies.

        Args:
            partition_config (tuple): _description_
            tau_max (int, optional): _description_. Defaults to 1.
        Returns:

        """
        temporal_pair_dict = {} # First index: frame_id. Second index: lag. Value: an integer array of shape num_attrs X num_attrs
        for frame_id in range( len(self.event_processor.frame_dict.keys())): # Collect all lagged pairs in all frames
            temporal_pair_dict[frame_id] = {}
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
                    prior_attr = attr_sequence[event_id]; con_attr = attr_sequence[event_id + lag]
                    temporal_pair_dict[frame_id][lag][attr_names.index(prior_attr), attr_names.index(con_attr)] += 1

        for frame_id in range( len(self.event_processor.frame_dict.keys())):
            for lag in range (1, self.tau_max + 1):
                attr_array = temporal_pair_dict[frame_id][lag]
                for idx, x in np.ndenumerate(attr_array):
                    # attr_array[idx] = 0 if x == 0 else 1
                    attr_array[idx] = 0 if x < self.partition_config[1] else 1 # JC NOTE: Filter out pairs with low frequencies (threshold: the number of days for current frame)

        self.temporal_pair_dict = temporal_pair_dict
    
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
        spatial_pair_dict = {} # First index: frame_id. Second index: lag. Value: an integer array of shape num_attrs X num_attrs
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

        for frame_id in range( len(self.event_processor.frame_dict.keys())):
            spatial_pair_dict[frame_id] = {}
            for lag in range (1, self.tau_max + 1): 
                spatial_pair_dict[frame_id][lag] = spatial_array_list[lag-1]
        
        self.spatial_pair_dict = spatial_pair_dict 

    def _physical_correlation_identification(self):
        pass

    def _automation_correlation_identification(self):
        pass

    def construct_user_correlation_benchmark(self):
        self.user_correlation_dict = {}
        for frame_id in range(self.event_processor.frame_count):
            self.user_correlation_dict[frame_id] = {}
            for tau in range(1, self.tau_max + 1):
                temporal_array = self.temporal_pair_dict[frame_id][tau]
                spatial_array = self.spatial_pair_dict[frame_id][tau]
                # The user-activity correlation is a spatial-temporal dependency pair.
                user_correlation_array = temporal_array * spatial_array # The user correlation should be satisfy the temporal and spatial coherence.
                self.user_correlation_dict[frame_id][tau] = user_correlation_array

    def print_benchmark_info(self,frame_id=0, tau=1, type = 0):
        """Print out the identified device correlations.

        Args:
            frame_id (int, optional): _description_. Defaults to 0.
            tau (int, optional): _description_. Defaults to 1.
            type (int, optional): {
                0: Temporal correlation, 1: Spatial correlation
                2: User-activity correlation, 3: Physical-channel correlation
                4: Automation rule correlation
            }
        """
        interested_array = self.correlation_dict[type][frame_id][tau]
        pair_list = []
        attr_names = self.event_processor.attr_names; num_attrs = len(attr_names)

        for index, x in np.ndenumerate(interested_array):
            if x == 0:
                continue
            pair_list.append((attr_names[index[0]], attr_names[index[1]]))
        
        print("Number of pairs: {}".format(len(pair_list)))
        print("Pair list: {}".format(pair_list))
        
        return pair_list

    def _estimate_single_discovery_accuracy(self, frame_id, tau, discovered_links_dict):
        pc_results = discovered_links_dict[frame_id]
        attr_names = self.event_processor.attr_names
        n_discovery = len( [x for x in sum(list(pc_results.values()), []) if x[1] == -1 * tau ] ) # The number of discovered causal links
        truth_array = self.user_correlation_dict[frame_id][tau]; truth_count = np.sum(truth_array) # The number of true causal links
        tp = 0; fn = 0; fp = 0
        for idx, x in np.ndenumerate(truth_array):
            if x == 1: # We only check those causal links
                if (attr_names[idx[0]] in pc_results.keys()) and ( (attr_names[idx[1]], -1 * tau) in pc_results[attr_names[idx[0]]]):
                    tp += 1
                else:
                    fn += 1
                    print("Missing {} -> {}".format( self.event_processor.attr_names[idx[0]], self.event_processor.attr_names[idx[1]] ))

        fp = n_discovery - tp
        print("\n##\n## [frame_id={}, tau={}] Evaluating stable-pc algorithm\n##".format(frame_id, tau)
                  + "\n\nParameters:")
        print(  "\nn_discovery = %d" % n_discovery
                  + "\ntruth_count = %s" % truth_count 
                  + "\ntp = %d" % tp
                  + "\nfn = %d" % fn 
                  + "\nfp = %d" % fp)

if __name__ == '__main__':
    # Parameter setting
    dataset = 'hh101'
    partition_config = (1, 10)
    tau_max = 1; tau_min = 1
    verbosity = 0  # -1: No debugging information; 0: Debugging information in parallel module; 1: Debugging info in PCMCI class; 2: Debugging info in CIT implementations
    ## For stable-pc
    pc_alpha = 0.2
    max_conds_dim = 10
    maximum_comb = 1
    ## For MCI
    alpha_level = 0.005

    evaluator = Evaluator(dataset=dataset, partition_config=partition_config, tau_max=tau_max)