from os import stat, path
from tkinter import W
import jenkspy
import math
import numpy as np
from datetime import datetime
from collections import defaultdict
from pprint import pprint

from sklearn.linear_model import PassiveAggressiveClassifier
from src.tigramite.tigramite import data_processing as pp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

class Processor:

	def __init__(self, dataset):
		self.dataset = dataset
		self.data_path = './data'
		self.origin_data = '{}/{}/data-origin'.format(self.data_path, self.dataset)
		self.transition_data = '{}/{}/data-transition'.format(self.data_path, self.dataset) # The default name of the original data file should be "data"
		self.training_data = '{}/{}/data-traning'.format(self.data_path, self.dataset)
		self.testing_data = '{}/{}/data-testing'.format(self.data_path, self.dataset)
		# Path variables
		self.adj_path_prefix = '{}/{}/adj/'.format(self.data_path, self.dataset)
		self.armadj_path_prefix = '{}/{}/armadj/'.format(self.data_path, self.dataset)
		self.stableadj_path_prefix = '{}/{}/stableadj/'.format(self.data_path, self.dataset)
		self.mat_path_prefix = '{}/{}/mat/'.format(self.data_path, self.dataset)
		self.partition_path_prefix = '{}/{}/partition/'.format(self.data_path, self.dataset)
		self.timeorder_path_prefix = '{}/{}/timeorder/'.format(self.data_path, self.dataset)
		self.priortruth_path_prefix = '{}/{}/priortruth/'.format(self.data_path, self.dataset)
		self.plot_path_prefix = '{}/{}/rplot/'.format(self.data_path, self.dataset)
		self.truth_path_prefix = '{}/{}/truth/'.format(self.data_path, self.dataset)

		# Golden standards
		self.instrumented_automation_rule_dict: 'dict[str, str]' = {}
		self.true_interactions_dict: 'dict[str, np.ndarray]' = {}

		# Inferred variables
		# self.device_count_dict: 'dict[str, int]' = {}; self.sensor_count_dict: 'dict[str, int]' = {}; self.actuator_count_dict: 'dict[str, int]' = {}
		# self.interaction_graphs: 'dict[str, InteractionGraph]' = defaultdict(InteractionGraph)
		# self.dependency_interval_dict: 'dict[str, list[float]]' = {}

	def time_based_partition(self, act_label: str = ''): # Set the time interval as 1 hour. Therefore, the whole dataset will be partitioned to 24 parts.
		start_time = act_label.split("_")[0]
		end_time = act_label.split("_")[1]
		start_time = datetime.datetime.strptime(start_time, '%H')
		end_time = datetime.datetime.strptime(end_time, '%H')
		fin = open(self.training_data, 'r') # The training dataset
		Path(self.partition_path_prefix + act_label).mkdir(parents=True, exist_ok=True) # Create partition directories if not exist
		fout_time = open(self.partition_path_prefix + act_label + '/data', 'w+') # The interested dataset given the activity label
		involved_sensors = {}
		for line in fin.readlines():
			inp = line.strip().split(' ')
			rec_time = datetime.datetime.strptime(inp[1].split('.')[0], '%H:%M:%S')
			sensor_name = inp[2]
			if (rec_time >= start_time and rec_time < end_time):
				fout_time.write(line) # write data to act-label dataset
				try: # Get the usage statistics for each sensor in the activity
					involved_sensors[sensor_name] += 1
				except:
					involved_sensors[sensor_name] = 1
			involved_sensors = dict(sorted(involved_sensors.items(), key = lambda item: item[1]))
		fin.close()
		fout_time.close()
		return involved_sensors

	def build_timerel_and_truth(self, act_label:'str' = '', devices_list:'list[str]' = [], process_index_dict:'dict[str, int]'= {}):
		mixed_truth_mat = np.zeros( (len(devices_list), len(devices_list)) )
		for process, process_index in process_index_dict.items(): # The 3rd step: Build the corresponding time order file and the truth file
			time_mat = np.zeros( (len(devices_list), len(devices_list)) )
			truth_mat = np.zeros( (len(devices_list), len(devices_list)) )
			sensors = list(process.split(" ")); num_device = len(sensors)
			for i in range(num_device): # Build the time order file
				for j in range(num_device):
					if i < j: 
						time_mat[devices_list.index(sensors[i]), devices_list.index(sensors[j]) ] = 2
						time_mat[devices_list.index(sensors[j]), devices_list.index(sensors[i]) ] = 1
					elif i > j: 
						time_mat[devices_list.index(sensors[i]), devices_list.index(sensors[j]) ] = 1
						time_mat[devices_list.index(sensors[j]), devices_list.index(sensors[i]) ] = 2
					else:
						pass
			for i in range(num_device - 1): # Build the truth file: Simply according to the time order
				truth_mat[devices_list.index(sensors[i]), devices_list.index(sensors[i+1]) ] = 1; mixed_truth_mat[devices_list.index(sensors[i]), devices_list.index(sensors[i+1]) ] = 1
			for trigger, responser in self.instrumented_automation_rule_dict.items(): # Build the truth file according to the instrumented automaiton rule
					truth_mat[devices_list.index(trigger), devices_list.index(responser) ] = 1; mixed_truth_mat[devices_list.index(trigger), devices_list.index(responser) ] = 1				
			df = pd.DataFrame(data=time_mat.astype(int))
			df.to_csv(self.timeorder_path_prefix + act_label + '/{}.mat'.format(process_index), sep = ' ', header = devices_list, index = False) # write to time order file
			df = pd.DataFrame(data=truth_mat.astype(int))
			df.to_csv(self.priortruth_path_prefix + act_label + '/{}.mat'.format(process_index), sep = ' ', header = devices_list, index = False) # write to truth file
		self.true_interactions_dict[act_label] = mixed_truth_mat
		df = pd.DataFrame(data=mixed_truth_mat.astype(int))
		df.to_csv(self.priortruth_path_prefix + act_label + '/mixed.mat', sep = ' ', header = devices_list, index = False) # write to truth file

	def association_rule_mining(self, act_label_list, min_sup=2, min_conf=0.1):
		for act_label in act_label_list:
			sensors = list(pd.read_csv("{}{}/{}".format(self.adj_path_prefix, act_label, "mixed.mat"), delim_whitespace=True).columns.values)
			mat_files = list(Path(self.mat_path_prefix + act_label).glob('*.mat'))
			mixed_matrix = np.zeros((len(sensors), len(sensors)))
			for process_file in mat_files:
				df = pd.read_csv(process_file, delim_whitespace=True)
				transactions = [] # The list of itemsets
				tmp = df.mul(df.columns).apply(lambda x: ','.join(filter(bool, x)), 1) # Check those itemsets
				for row in tmp:
					if len(row) == 0:
						continue
					transactions.append(list(row.split(',')))
				relim_input = itemmining.get_relim_input(transactions)
				item_sets = itemmining.relim(relim_input, min_support=min_sup)
				rules = assocrules.mine_assoc_rules(item_sets, min_support=min_sup, min_confidence=min_conf)
				process_adj_matrix = np.zeros((len(sensors), len(sensors)))
				for rule in rules:
					trigger_sensors_set = rule[0]
					action_sensors_set = rule[1]
					for trigger_sensor in trigger_sensors_set:
						for action_sensor in action_sensors_set:
							# print("{} -> {}".format(trigger_sensor, action_sensor))
							mixed_matrix[sensors.index(trigger_sensor), sensors.index(action_sensor)] = 1
							process_adj_matrix[sensors.index(trigger_sensor), sensors.index(action_sensor)] = 1
				final_df = pd.DataFrame(data=mixed_matrix.astype(int))
				Path("{}{}".format(self.armadj_path_prefix, act_label)).mkdir(parents=True, exist_ok=True)
				final_df.to_csv("{}{}/{}".format(self.armadj_path_prefix, act_label, "mixed.mat"), sep = ' ', header = sensors, index = False)

class Hprocessor(Processor):

	def __init__(self, dataset, verbosity=0):
		super().__init__(dataset)
		self.attr_names = None; self.n_vars = 0
		self.name_device_dict = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr name as the dict key
		self.index_device_dict = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr index as the dict key
		self.attr_count_dict = defaultdict(int); self.dev_count_dict = defaultdict(int)
		self.transition_events_states = None # Lists of all (event, state array) tuple
		self.frame_dict:'defaultdict(DataFrame)' = None # A dict with key, value = (frame_id, dict['number', 'day-interval', 'start-date', 'end-date', 'attr-sequence', 'attr-type-sequence', 'state-sequence'])
		self.frame_count = None
		self.tau_max = -1
		self.verbosity = verbosity
		# Variables for testing purposes
		self.discretization_dict:'dict[tuple]' = {}

	def _parse_raw_events(self, raw_event: "str"):
		"""Transform raw events into well-formed tuples

		Args:
			raw_event (str): The raw event logs

		Returns:
			event_tuple (AttrEvent): AttrEvent(date, time, dev, dev, value)
		"""
		raw_event = ' '.join(raw_event.split())
		inp = raw_event.strip().split(' ')
		return AttrEvent(inp[0], inp[1], inp[2], inp[6], inp[5])

	def _enum_unification(self, val: 'str') -> 'str':
		act_list = ["ON", "OPEN", "HIGH", "1"]
		de_list = ["OFF", "CLOSE", "LOW", "0"]
		unified_val = ''
		if val in act_list:
			unified_val = 1
		elif val in de_list:
			unified_val = 0
		else:
			print("Unknown values detected: {}".format(val))
			assert(0)
		return unified_val

	def _timestr2Seconds(timestr):
		timestamp = timestr.split('.')[0]
		hh, mm, ss = timestamp.split(':')
		sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
		return sec

	def sanitize_raw_events(self):
		"""This function aims to filter unnecessary attributes and imperfect devices.

		Returns:
			qualified_events: list[AttrEvent]: The list of qualified parsed events
		"""
		qualified_events: list[AttrEvent] = []
		device_state_dict = defaultdict(str)
		if path.isfile(self.transition_data): # If we have parsed the file: No need to re-parse it.
			fin = open(self.transition_data, 'r')
			for line in fin.readlines():
				inp = ' '.join(line.split()).strip().split(' ')
				qualified_events.append(AttrEvent(inp[0], inp[1], inp[2], inp[3], inp[4]))
		else:
			fin = open(self.origin_data, 'r')
			missed_attr_dicts = defaultdict(int)
			for line in fin.readlines():
				parsed_event:'AttrEvent' = self._parse_raw_events(line) # (date, time, dev_name, dev_attr, value)
				'''
				0. Filter noisy events.
					Some datasets contain noisy events including typos and setup events.
					As a result, the preprocessor should remove them.
				'''
				if self.dataset == 'hh101' and datetime.strptime(parsed_event.date, '%Y-%m-%d') <= datetime.strptime('2012-07-18', '%Y-%m-%d'): # The events before the date 07-18 are all setup events.
					continue
				elif self.dataset == 'hh130' and datetime.strptime(parsed_event.date, '%Y-%m-%d') <= datetime.strptime('2014-04-20', '%Y-%m-%d'): # The events before the date 07-18 are all setup events.
					continue
				'''
				1. Filter periodic attribute events.
					In our work, we only consider response attribute events.
					Specifically, we identify a list of attributes from SmartThings website.
					The information about the attributes can be also obtained from the dataset readme file.
				'''
				# 1. Select interested attributes.
				# int_attrs = ['Control4-Motion',  'Control4-Door', 'Control4-LightSensor', 'Control4-Temperature', 'Control4-Light']
				int_attrs = ['Control4-Motion',  'Control4-Door'] # JC NOTE: Adjust interested nodes here
				if parsed_event.attr not in int_attrs:
					missed_attr_dicts[parsed_event.attr] += 1
					continue
				if self.dataset == 'hh130':
					if parsed_event.attr == 'Control4-LightSensor' and parsed_event.dev not in ['LS007', 'LS008', 'LS009', 'LS010']: # Filter those imperfect light sensor (motion sensors with stickers)
						continue
					if parsed_event.attr == 'Control4-Temperature' and parsed_event.dev not in ['T103', 'T104']:
						continue
				# 2. Remove redundant events (which implies no state transitions)
				if device_state_dict[parsed_event.dev] == parsed_event.value:
					continue
				# 3. Collect legitimate events
				device_state_dict[parsed_event.dev] = parsed_event.value
				qualified_events.append(parsed_event)
			if self.verbosity:
				print("[Preprocessing] Missed attr dict during data preprocessing:")
				pprint(missed_attr_dicts)
		return qualified_events

	def unify_value_type(self, parsed_events: "list[AttrEvent]") -> "list[AttrEvent]":
		"""This function unifies the attribute values by the following two steps.
		1. Initiate the numeric-enum conversion.
		2. Unify the enum variables and map their ranges to {0, 1}

		Args:
			parsed_events (list[list[str]]): The parsed sanitized events (derived from the sanitize_data function)

		Returns:
			unified_parsed_events (list[list[str]]): The unified events
		"""

		# 1. Numeric-to-enum conversion
		continuous_attr_dict = defaultdict(list);  numeric_attr_dict = defaultdict(float)
		if path.isfile(self.transition_data):
			pass
		else:
			for parsed_event in parsed_events: # First collect all float values for each numeric attribute
				try:
					float_val = float(parsed_event.value)
					continuous_attr_dict[parsed_event.dev].append(float_val) # In this dataset, the device name is actually the attribute name.
				except:
					continue
			for k, v in continuous_attr_dict.items(): # Then call natural breaks algorithms to get the break for each attribute
				numeric_attr_dict[k] = jenkspy.jenks_breaks(v, nb_class=2)[1]
				self.discretization_dict[k] = (v, numeric_attr_dict[k])
			for parsed_event in parsed_events: # Finally, transform the numeric attribute to low-high enum attribute 
				if parsed_event.dev in numeric_attr_dict.keys():
					parsed_event.value = "HIGH" if float(parsed_event.value) > numeric_attr_dict[parsed_event.dev] else "LOW"
		# 2. Enum unification
		for parsed_event in parsed_events: # Unify the range of all enum variables to {0, 1}
			parsed_event.value = self._enum_unification(parsed_event.value)
		return parsed_events
	
	def create_data_frame(self, unified_parsed_events: "list[AttrEvent]"):
		"""This function takes unified events as inputs, and filters non-transition events
		Moreover, it creates the data frame for the transition events
		Finally, this function writes these transition events into data files

		Args:
			unified_parsed_events (list[AttrEvent]): The unified events (from the unify_value_type function)

		Returns:
			attr_names (list[str]): The list of involved attributes 
			transition_events_states (list[tuple(list[str], ndarray[int])]): The list of tuple (transition events, transited states)
		"""
		attr_names = set()
		transition_events_states = []
		for unified_event in unified_parsed_events: # Get the list of attributes
			attr_names.add(unified_event.dev)
		attr_names = list(attr_names); attr_names.sort()
		for i in range(len(attr_names)): # Build the index for each attribute
			device = DevAttribute(attr_name=attr_names[i], attr_index=i, lag=0)
			self.name_device_dict[attr_names[i]] = device; self.index_device_dict[i] = device
		assert(len(self.name_device_dict.keys()) == len(self.index_device_dict.keys())) # Otherwise, the violation indicates that there exists devices with the same name
		last_states = [0] * len(attr_names) # The initial states are all 0
		for unified_event in unified_parsed_events:
			cur_states = last_states.copy()
			# Filter redundant events which do not imply state changes
			if cur_states[self.name_device_dict[unified_event.dev].index] == unified_event.value:
				continue
			cur_states[self.name_device_dict[unified_event.dev].index] = unified_event.value
			transition_events_states.append((unified_event, np.array(cur_states)))  # Record legitimate events
			self.dev_count_dict[unified_event.dev] += 1; self.attr_count_dict[unified_event.attr] += 1
			last_states = cur_states
		if self.verbosity:
			print("[Preprocessing] Event preprocessing finished. Target attribute dict")
			pprint(self.attr_count_dict)
			print("[Preprocessing] Event preprocessing finished. Target device dict")
			pprint(self.dev_count_dict)
		if not path.isfile(self.transition_data):
			fout = open(self.transition_data, 'w+') # Finally, write these transition events into the data file
			for tup in transition_events_states: 
				fout.write(tup[0].__str__() + '\n')
			fout.close()
		self.attr_names = attr_names; self.n_vars = len(attr_names)
		return transition_events_states
	
	def partition_data_frame(self, transition_events_states, partition_config, training_ratio):
		"""Partition the data frame according to the set of triggered attributes

		Args:
			transition_events_states (list, optional): _description_. Defaults to [].
			partition_config (int, must): The partitioned days
		
		Returns:
			dataframes (list[Dataframe]): The separated data frames
		"""
		frame_dict = defaultdict(DataFrame)
		states_array = np.stack([tup[1] for tup in transition_events_states], axis=0)

		last_timestamp = ''; count = 0
		seg_points = []
		for tup in transition_events_states: # First get the segmentation points
			transition_event:'AttrEvent' = tup[0]
			cur_timestamp = '{} {}'.format(transition_event.date, transition_event.time)
			last_timestamp = cur_timestamp if last_timestamp == '' else last_timestamp
			past_days = ((datetime.fromisoformat(cur_timestamp) - datetime.fromisoformat(last_timestamp)).total_seconds()) * 1.0 / 86400
			if past_days >= partition_config:
				seg_points.append(count)
				last_timestamp = cur_timestamp
			count += 1

		last_point = 0; frame_count = 0
		for seg_point in seg_points: # Get the data frame with range [last_point, seg_point]
			if seg_point - last_point < 100: # If current day contains too few records, append the current day's record to the next day.
				continue
			testing_start_point = math.floor(last_point + training_ratio * (seg_point - last_point))
			training_data = states_array[last_point:testing_start_point, ]; testing_data = states_array[testing_start_point: seg_point, ]
			dataframe = pp.DataFrame(data=training_data, var_names=self.attr_names); testing_dataframe = pp.DataFrame(data=testing_data, var_names=self.attr_names)
			dframe = DataFrame(id=frame_count, var_names=self.attr_names, n_events=seg_point-last_point)
			dframe.set_training_data(transition_events_states[last_point:testing_start_point], dataframe)
			dframe.set_testing_data(transition_events_states[testing_start_point:seg_point], testing_dataframe)
			frame_dict[frame_count] = dframe

			frame_count += 1
			last_point = seg_point

		self.transition_events_states = transition_events_states
		self.frame_dict = frame_dict
		self.frame_count = frame_count
	
	def select_suitable_tau(self, transition_events_states):
		transition_events:'list[AttrEvent]' = [tup[0] for tup in transition_events_states]
		intervals = []
		act_detected = False; count = 0.0
		for evt in transition_events:
			if int(evt.value) == 1: # An activation event is detected.
				act_detected = True
				if count > 0:
					intervals.append(count + 1)
				count = 1.0
			elif act_detected:
				count += 1
		avg_interval = round(sum(intervals) / len(intervals))
		print("The average of activation intervals is {}".format(avg_interval))

		self.tau_max = avg_interval - 1 # The tau_max did not count the current timestamp

	def initiate_data_preprocessing(self):
		"""
		The entrance function for preprocessing data
		"""
		parsed_events = self.sanitize_raw_events()
		unified_parsed_events = self.unify_value_type(parsed_events)
		self.create_data_frame(unified_parsed_events) # Create transition file

	def data_loading(self, partition_config=30, training_ratio=0.9):
		unified_parsed_events = []
		fin = open(self.transition_data, 'r')
		for line in fin.readlines(): # Transform each preprocessed line into the AttrEvent
			inp = line.strip().split(' ')
			unified_parsed_events.append(AttrEvent(inp[0], inp[1], inp[2], inp[3], int(inp[4])))
		transition_events_states = self.create_data_frame(unified_parsed_events)
		self.partition_data_frame(transition_events_states, partition_config, training_ratio)
		return self.frame_dict