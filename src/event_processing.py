import jenkspy
import math
import statistics
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pprint import pprint

from src.tigramite.tigramite import data_processing as pp
from src.genetic_type import DevAttribute, AttrEvent, DataFrame

def _enum_unification(val: 'str') -> 'int':
	act_list = ["ON", "OPEN", "HIGH", "1"]
	de_list = ["OFF", "CLOSE", "CLOSED", "LOW", "0"]
	unified_val = ''
	if val in act_list:
		unified_val = 1
	elif val in de_list:
		unified_val = 0
	else:
		print("Unknown values detected: {}".format(val))
		assert(0)
	return unified_val

class GeneralProcessor():

	def __init__(self, dataset, partition_days, training_ratio, verbosity=0):
		self.dataset = dataset
		self.data_path = './data/{}/'.format(self.dataset)
		self.origin_data = '{}data-origin'.format(self.data_path)
		self.transition_data = '{}data-transition'.format(self.data_path)

		self.partition_days = partition_days
		self.training_ratio = training_ratio
		self.verbosity = verbosity

		# Initialized in function read_preprocessed_data_file()
		self.var_names = []; self.n_vars = 0
		self.attr_names = []; self.n_attrs = 0
		self.name_device_dict = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr name as the dict key
		self.index_device_dict = defaultdict(DevAttribute) # The str-DevAttribute dict using the attr index as the dict key
		self.attr_count_dict = defaultdict(int); self.dev_count_dict = defaultdict(int)
		self.transition_events_states = None # Lists of all (event, state array) tuple
		self.frame_dict:'defaultdict(DataFrame)' = None # A dict with key, value = (frame_id, dict['number', 'day-interval', 'start-date', 'end-date', 'attr-sequence', 'attr-type-sequence', 'state-sequence'])
		self.frame_count = None
		# Variables for testing purposes
		self.discretization_dict:'dict[tuple]' = {}

	def read_preprocessed_data_file(self):
		# Return variables
		transition_events_states = []
		# Debugging variables
		var_names = set()
		name_device_dict = defaultdict(DevAttribute); index_device_dict = defaultdict(DevAttribute)
		attr_count_dict = defaultdict(int); dev_count_dict = defaultdict(int)

		# 1. Read data file and create AttrEvent object for each event
		unified_parsed_events = []
		fin = open(self.transition_data, 'r')
		for line in fin.readlines():
			inp = line.strip().split(' ')
			unified_parsed_events.append(AttrEvent(inp[0], inp[1], inp[2], inp[3], int(inp[4])))

		# 2. Construct the device list and corresponding index dictionary
		for unified_event in unified_parsed_events:
			var_names.add(unified_event.dev)
		var_names = list(var_names); var_names.sort()
		for i in range(len(var_names)):
			device = DevAttribute(attr_name=var_names[i], attr_index=i, lag=0)
			name_device_dict[device.name] = device; index_device_dict[device.index] = device
		assert(len(name_device_dict.keys()) == len(index_device_dict.keys())) # Otherwise, the violation indicates that there exists devices with the same name

		# 3. Construct the state vector for each event, and return the result
		last_states = [0] * len(var_names)
		for unified_event in unified_parsed_events:
			cur_states = last_states.copy(); cur_states[name_device_dict[unified_event.dev].index] = unified_event.value
			transition_events_states.append((unified_event, np.array(cur_states)))
			dev_count_dict[unified_event.dev] += 1; attr_count_dict[unified_event.attr] += 1
			last_states = cur_states

		# 4. Store the device information into the class
		self.var_names = var_names; self.n_vars = len(var_names)
		self.name_device_dict = name_device_dict; self.index_device_dict = index_device_dict
		self.attr_count_dict = attr_count_dict; self.dev_count_dict = dev_count_dict
		if self.verbosity > 0:
			print("[Data Loading] # records, attrs, devices = {}, {}, {}".format(
				len(unified_parsed_events), len(self.attr_count_dict.keys()), len(self.dev_count_dict.keys())
			))

		return transition_events_states

	def partition_data_frame(self, transition_events_states):
		frame_dict = defaultdict(DataFrame)

		# 0. Store all records.
		states_array = np.stack([tup[1] for tup in transition_events_states], axis=0)
		dataframe = pp.DataFrame(data=states_array, var_names=self.var_names)
		final_timestamp = '{} {}'.format(transition_events_states[-1][0].date, transition_events_states[-1][0].time)
		first_timestamp = '{} {}'.format(transition_events_states[0][0].date, transition_events_states[0][0].time)
		dframe = DataFrame(id='all', var_names=self.var_names, attr_names=self.attr_names, n_events=len(transition_events_states),\
			n_days = ((datetime.fromisoformat(final_timestamp) - datetime.fromisoformat(first_timestamp)).total_seconds())*1.0/86400)
		dframe.set_device_info(self.name_device_dict, self.index_device_dict)
		dframe.set_training_data(transition_events_states, dataframe)
		dframe.set_testing_data(transition_events_states, dataframe) # For this frame which stores all events, we did not separate training and testing data
		frame_dict[dframe.id] = dframe

		# 1. Segment all frames, and store each partition to the frame_dict.
		last_timestamp = ''; count = 0
		seg_points = []; seg_days = []; past_days = 0
		for tup in transition_events_states:
			transition_event:'AttrEvent' = tup[0]
			cur_timestamp = '{} {}'.format(transition_event.date, transition_event.time)
			last_timestamp = cur_timestamp if last_timestamp == '' else last_timestamp
			past_days = ((datetime.fromisoformat(cur_timestamp) - datetime.fromisoformat(last_timestamp)).total_seconds()) * 1.0 / 86400
			if past_days >= self.partition_days:
				seg_days.append(past_days)
				seg_points.append(count)
				last_timestamp = cur_timestamp
			count += 1
		seg_points.append(count - 1)
		seg_days.append(past_days)

		# 2. Create DataFrames for logs in each segmentation interval
		last_point = 0; frame_count = 0
		for i, seg_point in enumerate(seg_points): # Get the data frame with range [last_point, seg_point]
			testing_start_point = math.floor(last_point + self.training_ratio * (seg_point - last_point))
			training_data = states_array[last_point:testing_start_point, ]; testing_data = states_array[testing_start_point: seg_point, ]
			if testing_start_point-last_point < len(self.var_names) or seg_point - testing_start_point < len(self.var_names): # If the current frame contains too few records
				last_point = seg_point
				continue
			dataframe = pp.DataFrame(data=training_data, var_names=self.var_names); testing_dataframe = pp.DataFrame(data=testing_data, var_names=self.var_names)
			dframe = DataFrame(id=frame_count, var_names=self.var_names, attr_names=self.attr_names, n_events=seg_point-last_point, n_days=seg_days[i])
			dframe.set_device_info(self.name_device_dict, self.index_device_dict)
			dframe.set_training_data(transition_events_states[last_point:testing_start_point], dataframe)
			dframe.set_testing_data(transition_events_states[testing_start_point:seg_point], testing_dataframe)
			frame_dict[frame_count] = dframe
			frame_count += 1; last_point = seg_point

		# 3. Store the frame information into the class
		self.transition_events_states = transition_events_states
		self.frame_dict = frame_dict
		self.frame_count = frame_count

	def data_loading(self):
		transition_events_states = self.read_preprocessed_data_file()
		self.partition_data_frame(transition_events_states)

class Cprocessor(GeneralProcessor):

	def __init__(self, dataset, partition_days, training_ratio, verbosity=0):
		super().__init__(dataset, partition_days, training_ratio, verbosity)
		self.device_description_dict = self._load_device_attribute_files()
		self.int_attrs = {'binary': ['Switch', 'Smart electrical outlet', 'Infrared Movement Sensor', 'Contact Sensor'],\
						  'discrete': ['Water Meter', 'Rollershutter', 'Dimmer', 'Power Sensor'],\
						  #'continuous': []}
						  'continuous': ['Humidity Sensor', 'Brightness Sensor']}
		self.int_locations = ['Bathroom', 'Bedroom', 'Dining Room', 'First Floor', 'Hallway', 'Hallway First Floor', 'Hallway Second Floor',\
						'Kitchen', 'Living Room', 'Main Entrance', 'Stairway', 'Stove', 'Study Room']

	def _load_device_attribute_files(self):
		"""
		The contextact dataset has a variable description excel file, which records dev_name - dev_attr mappings.
		Need to first read the file, and get the mapping tables.
		Returns:
			device_description_dict (defaultdict(dict)): dev-name -> {dev-name, attr, description, val-type, val-unit, location}
		"""
		device_description_dict = defaultdict(dict)
		df = pd.read_excel('{}variable-description.xlsx'.format(self.data_path), header=0, )
		df.fillna('missing', inplace=True)
		df.columns = ['dev', 'attr', 'description', 'val-type', 'val-unit', 'location']
		for row_index in range(len(df)):
			dev_info = [df.loc[row_index, col_index] for col_index in df.columns]
			for i in range(len(dev_info)):
				device_description_dict[dev_info[0]][df.columns[i]] = dev_info[i]
		return device_description_dict 

	def _parse_raw_events(self, raw_event: "str"):
		"""
		Transform raw events into well-formed tuples
		Returns:
			event_tuple (AttrEvent): AttrEvent(date, time, dev, attr, value)
			Note that dev segment should uniquely identify a device.
		"""
		raw_event = raw_event.replace("\"", "")
		try:
			inp = raw_event.strip().split(';')
			assert(len(inp)==3) # (Date-time, dev, dev-value) pair
		except:
			return None
		date = inp[0].split(' ')[0]; time = inp[0].split(' ')[1].replace(",", ".")
		dev = inp[1]; dev_val = inp[2]
		return AttrEvent(date, time, dev,\
								self.device_description_dict[dev]['attr'], dev_val)

	def sanitize_raw_events(self):
		"""
		This function aims to filter unnecessary attributes and imperfect devices.
		Returns:
			qualified_events: list[AttrEvent]: The list of qualified parsed events
		"""
		fin = open(self.origin_data, 'r', encoding='utf-8', errors='ignore')
		lines = fin.readlines()

		device_count_dict = defaultdict(int)
		for line in lines:
			parsed_event:'AttrEvent' = self._parse_raw_events(line) # (date, time, dev_name, dev_attr, value)
			if not parsed_event:
				continue
			device_count_dict[parsed_event.dev] += 1
		less_frequent_devices = [dev for dev in device_count_dict.keys()\
				if device_count_dict[dev] < 200]

		qualified_events: list[AttrEvent] = []
		device_state_dict = defaultdict(str)
		attr_occurrence_dict = defaultdict(dict)
		missed_attr_dicts = defaultdict(int)
		for line in lines:
			'''
			0. Filter noisy events.
				Some datasets contain noisy events including typos and setup events.
				As a result, the preprocessor should remove them.
			'''
			parsed_event:'AttrEvent' = self._parse_raw_events(line) # (date, time, dev_name, dev_attr, value)
			if not parsed_event:
				continue
			'''
			1. Event filtering.
				* Remove events of uninterested attributes and specific devices.
				* Remove events which indicate redundant state reports.
			'''
			# 1.0 Filter events based on device frequencies.
			if parsed_event.dev in less_frequent_devices:
				continue
			# 1.1 Filter events based on attributes and locations.
			if all(parsed_event.attr not in sublist for sublist in self.int_attrs.values()):
				missed_attr_dicts["{}".format(parsed_event.attr)] += 1
				continue
			if self.device_description_dict[parsed_event.dev]['location'] not in self.int_locations:
				continue
			# 1.2 For some attributes, filter some unnecessary devices
			if parsed_event.attr == 'Water Meter' and "Total" in parsed_event.dev:
				continue
			if parsed_event.attr == 'Switch' and parsed_event.dev.startswith('I'):
				inp = parsed_event.dev.split("_")
				if len(inp) <= 1:
					continue
				#parsed_event.dev = inp[0]
			# 1.3 Remove redundant events (which implies no state transitions)
			if device_state_dict[parsed_event.dev] == parsed_event.value:
				continue
			# 1.4 Count legitimate events.
			attr_occurrence_dict[parsed_event.attr][parsed_event.dev] = 1 if parsed_event.dev not in attr_occurrence_dict[parsed_event.attr].keys()\
					else attr_occurrence_dict[parsed_event.attr][parsed_event.dev] + 1
			device_state_dict[parsed_event.dev] = parsed_event.value
			qualified_events.append(parsed_event)
		if self.verbosity:
			#print("[Data Sanitization] Missed attr dict during data preprocessing:")
			#pprint(missed_attr_dicts)
			attrs = list(attr_occurrence_dict.keys())
			attr_n_devs = [len(attr_occurrence_dict[attr].keys()) for attr in attrs]
			attr_n_events = [sum(list(attr_occurrence_dict[attr].values())) for attr in attrs]
			print("[Data Sanitization] Candidate attrs, n_devices, and n_events: {}".format(list(zip(attrs, attr_n_devs, attr_n_events))))
		return qualified_events

	def unify_value_type(self, parsed_events: "list[AttrEvent]") -> "list[AttrEvent]":
		"""
		This function unifies the attribute values by the following two steps.
		1. Initiate the numeric-enum conversion.
		2. Unify the enum variables and map their ranges to {0, 1}
		Returns:
			unified_parsed_events (list[list[str]]): The unified events
		"""
		# Return variables
		unified_parsed_events: "list[AttrEvent]" = []

		# 1. Preprocess discrete variables, and get summary of continuous variables.
		continuous_dev_dict = defaultdict(list);  continuous_threshold_dict = defaultdict(float)
		for parsed_event in parsed_events:
			if parsed_event.attr in self.int_attrs['discrete']:
				float_val = float(parsed_event.value)
				parsed_event.value = 'ON' if float_val > 0 else 'OFF'
			elif parsed_event.attr in self.int_attrs['continuous']:
				float_val = float(parsed_event.value)
				continuous_dev_dict[parsed_event.dev].append(float_val)
		
		# 2. For continuous variables, filter extreme values using 3-sigma rules
		continuous_stat_dict = defaultdict(dict)
		for k, v in continuous_dev_dict.items():
			mean = statistics.mean(v); continuous_stat_dict[k]['mean'] = mean
			std_dev = statistics.stdev(v); continuous_stat_dict[k]['std-dev'] = std_dev
			continuous_dev_dict[k] = [x for x in v if mean-3.*std_dev <=x<= mean+3.*std_dev]

		# 3. For continuous variables, use natural breaks algorithm to get the threshold.
		for k, v in continuous_dev_dict.items():
			continuous_threshold_dict[k] = jenkspy.jenks_breaks(v, nb_class=2)[1]
			self.discretization_dict[k] = (v, continuous_threshold_dict[k])
		
		# 4. Collect events which passes 3-sigma rule tests, and unify their values to HIGH/LOW
		filtered_parsed_event = []
		for parsed_event in parsed_events:
			if parsed_event.dev not in continuous_threshold_dict.keys():
				filtered_parsed_event.append(parsed_event)
			if parsed_event.dev in continuous_threshold_dict.keys() and \
				continuous_stat_dict[parsed_event.dev]['mean']-3.*continuous_stat_dict[parsed_event.dev]['std-dev'] <= float(parsed_event.value) <= continuous_stat_dict[parsed_event.dev]['mean']+3.*continuous_stat_dict[parsed_event.dev]['std-dev']:
				parsed_event.value = "HIGH" if float(parsed_event.value) > continuous_threshold_dict[parsed_event.dev] else "LOW"
				filtered_parsed_event.append(parsed_event)

		# 5. Finally, transform all unified attribute values to 0/1
		for parsed_event in filtered_parsed_event:
			parsed_event.value = _enum_unification(parsed_event.value)
			unified_parsed_events.append(parsed_event)
			if parsed_event.attr in ['Water Meter', 'Power Sensor']:
				# These devices do not record idle states. Therefore,\
				# after each usage, we need to add an additional events, which indicate its idle states.
				unified_parsed_events.append(AttrEvent(parsed_event.date, parsed_event.time,\
										parsed_event.dev, parsed_event.attr, 0))
		return unified_parsed_events

	def create_preprocessed_data_file(self, unified_parsed_events: "list[AttrEvent]"):
		fout = open(self.transition_data, 'w+')
		# 1. Identify all devices in the dataset
		var_names = set()
		for unified_event in unified_parsed_events:
			var_names.add(unified_event.dev)
		var_names = list(var_names); var_names.sort()
		# 2. Build the index for each device
		name_device_dict:'dict[DevAttribute]' = {}; index_device_dict:'dict[DevAttribute]' = {}
		for i in range(len(var_names)):
			device = DevAttribute(name=var_names[i], index=i, attr=self.device_description_dict[var_names[i]]['attr'],\
								location=self.device_description_dict[var_names[i]]['location'])
			name_device_dict[var_names[i]] = device; index_device_dict[i] = device
		assert(len(name_device_dict.keys()) == len(index_device_dict.keys())) # The violation indicates that there exists devices with the same name
		# 3. Filter redundant events which do not imply state changes, and get summary of qualified events
		last_states = [0] * len(var_names)
		n_events = 0
		attr_occurrence_dict = defaultdict(dict)
		for unified_event in unified_parsed_events:
			cur_states = last_states.copy()
			if cur_states[name_device_dict[unified_event.dev].index] == unified_event.value:
				continue
			# Update the dataset summary
			n_events += 1
			attr_occurrence_dict[unified_event.attr][unified_event.dev] = 1 if unified_event.dev not in attr_occurrence_dict[unified_event.attr].keys()\
					else attr_occurrence_dict[unified_event.attr][unified_event.dev] + 1
			# Write the event to file
			unified_event.dev = unified_event.dev.replace(' ', '-') # Remove the space in original records
			unified_event.attr = unified_event.attr.replace(' ', '-')
			fout.write(unified_event.__str__() + '\n')
			# Update the current state vector
			cur_states[name_device_dict[unified_event.dev].index] = unified_event.value
			last_states = cur_states
		if self.verbosity:
			attrs = list(attr_occurrence_dict.keys())
			attr_n_devs = [len(attr_occurrence_dict[attr].keys()) for attr in attrs]
			attr_n_events = [sum(list(attr_occurrence_dict[attr].values())) for attr in attrs]
			print("[Event Conversion] # qualified events, devices = {}, {}".format(n_events, len(var_names)))
			print("[Event Conversion] Candidate attrs, n_devices, and n_events: {}".format(list(zip(attrs, attr_n_devs, attr_n_events))))
		fout.close()

	def read_preprocessed_data_file(self):
		# Return variables
		transition_events_states = []
		# Debugging variables
		var_names = set(); attr_names = set()
		name_device_dict = defaultdict(DevAttribute); index_device_dict = defaultdict(DevAttribute)
		attr_count_dict = defaultdict(int); dev_count_dict = defaultdict(int)

		# 1. Read data file and create AttrEvent object for each event
		unified_parsed_events = []
		fin = open(self.transition_data, 'r')
		for line in fin.readlines():
			inp = line.strip().split(' ')
			unified_parsed_events.append(AttrEvent(inp[0], inp[1], inp[2], inp[3], int(inp[4])))

		# 2. Construct the device list and corresponding index dictionary
		for unified_event in unified_parsed_events:
			var_names.add(unified_event.dev)
			attr_names.add(unified_event.attr)
		var_names = list(var_names); var_names.sort()
		attr_names = list(attr_names); attr_names.sort()
		for i in range(len(var_names)):
			device = DevAttribute(name=var_names[i], index=i, attr=self.device_description_dict[var_names[i]]['attr'],\
								location=self.device_description_dict[var_names[i]]['location'])
			name_device_dict[device.name] = device; index_device_dict[device.index] = device
		assert(len(name_device_dict.keys()) == len(index_device_dict.keys())) # Otherwise, the violation indicates that there exists devices with the same name

		# 3. Construct the state vector for each event, and return the result
		last_states = [0] * len(var_names)
		for unified_event in unified_parsed_events:
			cur_states = last_states.copy(); cur_states[name_device_dict[unified_event.dev].index] = unified_event.value
			transition_events_states.append((unified_event, np.array(cur_states)))
			dev_count_dict[unified_event.dev] += 1; attr_count_dict[unified_event.attr] += 1
			last_states = cur_states

		# 4. Store the device information into the class
		self.var_names = var_names; self.n_vars = len(var_names)
		self.attr_names = attr_names; self.n_attrs = len(attr_names)
		self.name_device_dict = name_device_dict; self.index_device_dict = index_device_dict
		self.attr_count_dict = attr_count_dict; self.dev_count_dict = dev_count_dict
		if self.verbosity > 0:
			print("[Data Loading] # records, attrs, devices = {}, {}, {}".format(
				len(unified_parsed_events), len(self.attr_count_dict.keys()), len(self.dev_count_dict.keys())
			))

		return transition_events_states

	def initiate_data_preprocessing(self):
		parsed_events = self.sanitize_raw_events()
		unified_parsed_events = self.unify_value_type(parsed_events)
		self.create_preprocessed_data_file(unified_parsed_events)

class Hprocessor(GeneralProcessor):

	def __init__(self, dataset, partition_days, training_ratio, verbosity=0):
		super().__init__(dataset, partition_days, training_ratio, verbosity)

	def _parse_raw_events(self, raw_event: "str"):
		"""
		Transform raw events into well-formed tuples
		Returns:
			event_tuple (AttrEvent): AttrEvent(date, time, dev, attr, value)
			Note that dev segment should uniquely identify a device.
		"""
		raw_event = ' '.join(raw_event.split())
		inp = raw_event.strip().split(' ')
		return AttrEvent(inp[0], inp[1], inp[2], inp[6], inp[5])

	def sanitize_raw_events(self):
		"""
		This function aims to filter unnecessary attributes and imperfect devices.
		Returns:
			qualified_events: list[AttrEvent]: The list of qualified parsed events
		"""
		qualified_events: list[AttrEvent] = []
		device_state_dict = defaultdict(str)
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
			int_attrs = ['Control4-Motion',  'Control4-Door'] # JC NOTE: We only keep motion sensor and door events in the dataset
			if parsed_event.attr not in int_attrs:
				missed_attr_dicts["{} -- {}".format(parsed_event.dev, parsed_event.attr)] += 1
				continue
			if self.dataset == 'hh130':
				if parsed_event.attr == 'Control4-LightSensor' and parsed_event.dev not in ['LS007', 'LS008', 'LS009', 'LS010']:
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
		"""
		This function unifies the attribute values by the following two steps.
		1. Initiate the numeric-enum conversion.
		2. Unify the enum variables and map their ranges to {0, 1}

		Args:
			parsed_events (list[list[str]]): The parsed sanitized events (derived from the sanitize_data function)

		Returns:
			unified_parsed_events (list[list[str]]): The unified events
		"""

		# 1. Numeric-to-enum conversion
		continuous_attr_dict = defaultdict(list);  continuous_threshold_dict = defaultdict(float)
		for parsed_event in parsed_events: # First collect all float values for each numeric attribute
			try:
				float_val = float(parsed_event.value)
				continuous_attr_dict[parsed_event.dev].append(float_val) # In this dataset, the device name is actually the attribute name.
			except:
				continue
		for k, v in continuous_attr_dict.items(): # Then call natural breaks algorithms to get the break for each attribute
			continuous_threshold_dict[k] = jenkspy.jenks_breaks(v, nb_class=2)[1]
			self.discretization_dict[k] = (v, continuous_threshold_dict[k])
		for parsed_event in parsed_events: # Finally, transform the numeric attribute to low-high enum attribute 
			if parsed_event.dev in continuous_threshold_dict.keys():
				parsed_event.value = "HIGH" if float(parsed_event.value) > continuous_threshold_dict[parsed_event.dev] else "LOW"
		# 2. Enum unification
		for parsed_event in parsed_events: # Unify the range of all enum variables to {0, 1}
			parsed_event.value = _enum_unification(parsed_event.value)
		return parsed_events
	
	def create_preprocessed_data_file(self, unified_parsed_events: "list[AttrEvent]"):
		fout = open(self.transition_data, 'w+')
		# 1. Identify all devices in the dataset
		var_names = set()
		for unified_event in unified_parsed_events:
			var_names.add(unified_event.dev)
		var_names = list(var_names); var_names.sort()
		# 2. Build the index for each device
		for i in range(len(var_names)):
			device = DevAttribute(name=var_names[i], index=i)
			self.name_device_dict[var_names[i]] = device; self.index_device_dict[i] = device
		assert(len(self.name_device_dict.keys()) == len(self.index_device_dict.keys())) # The violation indicates that there exists devices with the same name
		# 3. Filter redundant events which do not imply state changes
		last_states = [0] * len(var_names)
		for unified_event in unified_parsed_events:
			cur_states = last_states.copy()
			if cur_states[self.name_device_dict[unified_event.dev].index] == unified_event.value:
				continue
			fout.write(unified_event.__str__() + '\n')
			cur_states[self.name_device_dict[unified_event.dev].index] = unified_event.value
			last_states = cur_states
		# 4. Write legitimate events to the data file
		fout.close()

	def initiate_data_preprocessing(self):
		parsed_events = self.sanitize_raw_events()
		unified_parsed_events = self.unify_value_type(parsed_events)
		self.create_preprocessed_data_file(unified_parsed_events)
