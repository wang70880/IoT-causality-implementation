from os import stat
from tkinter import W
import jenkspy
import numpy as np
from functools import reduce
from datetime import datetime
from src.tigramite import data_processing as pp
from src.tigramite.toymodels import structural_causal_processes as toys

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

	def identify_causal_processes(self, act_label = '', sensors_list = []):
		Path(self.mat_path_prefix + act_label).mkdir(parents=True, exist_ok=True)
		Path(self.timeorder_path_prefix + act_label).mkdir(parents=True, exist_ok=True)
		Path(self.priortruth_path_prefix + act_label).mkdir(parents=True, exist_ok=True)
		Path(self.truth_path_prefix + act_label).mkdir(parents=True, exist_ok=True)
		system("rm -rf {}/*".format(self.timeorder_path_prefix + act_label))
		system("rm -rf {}/*".format(self.priortruth_path_prefix + act_label))
		fin_act = open(self.partition_path_prefix + act_label + '/data', 'r') 
		sequence_count_dict = {} # Record all existing sequences in the dataset
		process_count_dict = {} # Record more frequent sequences in the dataset
		device_dependency_count_dict: 'dict[str, int]' = {} # Record less frequent sequences in the dataset
		complete_dependency_count_dict : 'dict[str, int]' = {}
		process_index_dict = {} 
		process_instance_dict = {} # Record the instance (the sequence of device state updates) of a process.
		process_lineno_dict = {} # Record the line no for each process instance, and write it to the txt file
		slist = []; statelist = []; linelist = []
		last_date = ''
		line_number = 0

		for line in fin_act.readlines(): # The 1st step: Identify causal processes in the act-dataset
			line_number += 1
			inp = line.strip().split(' ')
			date = inp[0]
			sensor_name = inp[2]
			sensor_state = inp[3]
			if (date != last_date) or (sensor_name in slist): # NOTE: How to partition the sequence is important!
				if len(slist) > 1: # Here we only calculate those causal processes with num(devices) > 1
					k = ' '.join(slist)
					sequence_count_dict[k] = sequence_count_dict[k] + 1 if k in sequence_count_dict.keys() else 1 # Record all the sequence
					process_instance_dict[k] = process_instance_dict[k] if k in process_instance_dict.keys() else []; process_instance_dict[k].append(statelist) # Record the process instance
					process_lineno_dict[k] = process_lineno_dict[k] if k in process_lineno_dict.keys() else []; process_lineno_dict[k].append(linelist) # Record the process line
				last_date = date
				slist = []; statelist = []; linelist = []
			slist.append(sensor_name); statelist.append(sensor_state); linelist.append(line_number)
		fin_act.close()
		system("rm -rf {}/*".format(self.mat_path_prefix + act_label))
		process_index = 1
		total_days = sum(list(sequence_count_dict.values()))

		for process, count in sequence_count_dict.items(): ## The 2nd step: Build the file for each identified causal processes, and the corresponding time order file
			if count < total_days * 0.005: # NOTE: For rare sequences (say, occurrence less than 0.01%), we regard them as device dependency and store them into the dependency_count_dict
				complete_dependency_count_dict[process] = count
				devices = list(process.split(" "))
				for i in range(len(devices) - 1):
					dependency = '{} {}'.format(devices[i], devices[i+1])
					device_dependency_count_dict[dependency] = device_dependency_count_dict[dependency] + count if dependency in device_dependency_count_dict.keys() else count
			else:
				#print("highly frequent process, count, index = {}, {}, {}".format(process, count, process_index))
				process_count_dict[process] = count; process_index_dict[process] = process_index
				fout = open(self.mat_path_prefix + act_label + '/{}.mat'.format(process_index), 'w+')
				ftxtout = open(self.mat_path_prefix + act_label + '/{}.txt'.format(process_index), 'w+')
				fout.write(" ".join(sensors_list) + '\n')
				sensors = list(process.split(" "))
				for i in range (len(process_instance_dict[process])): # Traverse each instance of the process, and generate and write the data
					process_instance = process_instance_dict[process][i]
					line_instance = process_lineno_dict[process][i]
					assert(len(sensors) == len(process_instance))
					# First, construct initial states before the process instance
					vec = ['0'] *  len(sensors_list) # The vec variable records the list of binary states
					for j in range(len(sensors)):  # (sensors[i], process_instance[i]) denotes the ith record
						sensor_index = sensors_list.index(sensors[j])
						vec[sensor_index] = '0' if process_instance[j] == 'A' else '1'
					fout.write(" ".join(vec) + "\n") # write the initial state (before the process starts) to the file
					# Second, for each record, we add the record to the dataset
					for j in range(len(sensors)):
						sensor_index = sensors_list.index(sensors[j])
						vec[sensor_index] = '1' if process_instance[j] == 'A' else '0'
						fout.write(" ".join(vec) + "\n") # write each log (before the process starts) to the file
						ftxtout.write("{}: {} {}\n".format(line_instance[0], sensors[j], process_instance[j])) if j == 0 else ftxtout.write("{} {}\n".format(sensors[j], process_instance[j]))
					ftxtout.write("\n")
				process_index += 1
				fout.close()
				ftxtout.close()
		return process_index_dict, process_count_dict, device_dependency_count_dict
	
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

	def generate_interaction_graph(self, act_label, alpha, processes_count_dict, device_dependency_count_dict):
		'''
		This function calls the Rscript to infer device interactions, and store the result into adjacency_mat
		Moreover, this function labels each discovered interactions, and store the result into label_mat
			* The label code is the following.
				0: No edge
				1: Human activity
				2: Automation rule
				3: physical channel
		'''
		Path(self.adj_path_prefix).mkdir(parents=True, exist_ok=True)
		Path(self.plot_path_prefix).mkdir(parents=True, exist_ok=True)
		Path(self.truth_path_prefix).mkdir(parents=True, exist_ok=True)
		os.system("Rscript pc.R {} {} {}".format(self.dataset, act_label, alpha))
		df = pd.read_csv("{}{}/{}".format(self.adj_path_prefix, act_label, "mixed.mat"), delim_whitespace=True)
		device_list = list(df.columns.values)
		adjacency_mat = df.to_numpy()
		label_mat = np.zeros( (len(device_list), len(device_list)) )
		discovered_automation_dict:'dict[str, str]' = {}
		mu_rule, sigma_rule = 0.05, 0.03 # TODO: Here we need to estimate the execution time of the automation execution.
		for i in range(len(device_list)): # Traverse each edge 
			for j in range(len(device_list)):
				if adjacency_mat[i, j] == 1: # label the edge
					preceding_device = device_list[i]
					post_device = device_list[j]
					mu_edge, sigma_edge = (0, 0)
					try:
						connection_interval_list = self.dependency_interval_dict['{} {}'.format(preceding_device, post_device)]
						mu_edge, sigma_edge = (statistics.mean(connection_interval_list), statistics.stdev(connection_interval_list))
					except:
						print('Cannot find the interaction {} {} in the dependency_interval_dict!'.format(preceding_device, post_device))
						mu_edge, sigma_edge = 10, 1
					label_mat[device_list.index(preceding_device), device_list.index(post_device)] = 2 if mu_edge == 0 else 1 # Apply three-sigma rule here
					if mu_edge  == 0:
						discovered_automation_dict[preceding_device] = post_device
			# print("For edge ({} -> {}), the edge label is {}".format(precede_sensor, post_sensor, label_mat[device_list.index(precede_sensor), device_list.index(post_sensor)] ))
		self.interaction_graphs[act_label] = InteractionGraph(self.dataset, act_label, device_list, processes_count_dict, device_dependency_count_dict, adjacency_mat, label_mat,  discovered_automation_dict)
		return self.interaction_graphs[act_label]

	def interaction_inspection(self, devices, int_graph, label_mat):
		location_id, rules = asyncio.run(_smartthings_rule_api())
		print("{}, {}".format(location_id, rules))
		#TODO: Parse the rule string (json format) here.

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

	def __init__(self, dataset):
		super().__init__(dataset)

	def _parse_raw_events(self, raw_event: "str"):
		"""Transform raw events into well-formed tuples

		Args:
			raw_event (str): The raw event logs

		Returns:
			event_tuple (list[str]): [date, time, dev_name, dev_attr, value]
		"""
		raw_event = ' '.join(raw_event.split())
		inp = raw_event.strip().split(' ')
		return [inp[0], inp[1], inp[2], inp[6], inp[5]]

	def _enum_unification(self, val: 'str') -> 'str':
		act_list = ["ON", "OPEN", "HIGH"]
		de_list = ["OFF", "CLOSE", "LOW"]
		unified_val = ''
		if val in act_list:
			unified_val = 'A'
		elif val in de_list:
			unified_val = 'DA'
		return unified_val

	def _timestr2Seconds(timestr):
		timestamp = timestr.split('.')[0]
		hh, mm, ss = timestamp.split(':')
		sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
		return sec

	def sanitize_raw_events(self):
		"""This function aims to sanitize raw events.
		Specifically, it initiates the following two steps.
		1. Filter periodic attribute events.
		2. Filter noisy events (in particular, duplicated device events)
		Returns:
			qualified_events: list[list[str]]: The list of qualified parsed events
		"""
		fin = open(self.origin_data, 'r')
		last_parsed_event: list[str] = []
		qualified_events: list[list[str]] = []
		for line in fin.readlines():
			parsed_event: list[str] = self._parse_raw_events(line) # (date, time, dev_name, dev_attr, value)
			'''
			0. Filter noisy events.
				Some datasets contain noisy events including typos and setup events.
				As a result, the preprocessor should remove them.
			'''
			if self.dataset == 'hh101' and datetime.strptime(parsed_event[0], '%Y-%m-%d') <= datetime.strptime('2012-07-18', '%Y-%m-%d'): # The events before the date 07-18 are all setup events.
				continue
			'''
			1. Filter periodic attribute events.
				In our work, we only consider response attribute events.
				Specifically, we identify a list of attributes from SmartThings website.
				The information about the attributes can be also obtained from the dataset readme file.
			'''
			# 1. Filter periodic attribute events.
			if parsed_event[3] not in ['Control4-Motion', 'Control4-Door', 'Control4-Temperature', 'Control4-LightSensor', 'Control4-Light', 'Control4-Button']:
				continue
			# 2. Remove duplicated device events
			if len(last_parsed_event) != 0 and \
			(last_parsed_event[0], last_parsed_event[2], last_parsed_event[3], last_parsed_event[4]) == (last_parsed_event[0], parsed_event[2], parsed_event[3], parsed_event[4]) :
				continue
			qualified_events.append(parsed_event)
			last_parsed_event = parsed_event.copy()
		return qualified_events
	
	def unify_value_type(self, parsed_events: "list[list[str]]") -> "list[list[str]]":
		"""This function unifies the attribute values by the following two steps.
		1. Initiate the numeric-enum conversion.
		2. Unify the enum variables and map their ranges to ['A', 'DA']

		Args:
			parsed_events (list[list[str]]): The parsed sanitized events (derived from the sanitize_data function)

		Returns:
			unified_parsed_events (list[list[str]]): The unified events
		"""

		# 1. Numeric-to-enum conversion
		numeric_attr_dict = {}
		unified_parsed_events: "list[list[str]]" = []
		for parsed_event in parsed_events: # First collect all float values for each numeric attribute
			attr_name = parsed_event[2] # In this dataset, the device name is actually the attribute name.
			try:
				val = float(parsed_event[4])
				numeric_attr_dict[attr_name] = numeric_attr_dict[attr_name] if attr_name in numeric_attr_dict.keys() else []
				numeric_attr_dict[attr_name].append(val)
			except:
				continue
		for k, v in numeric_attr_dict.items(): # Then call natural breaks algorithms to get the break for each attribute
			numeric_attr_dict[k] = jenkspy.jenks_breaks(v, nb_class=2)[1]
		for parsed_event in parsed_events: # Finally, transform the numeric attribute to low-high enum attribute 
			attr_name = parsed_event[2]
			if attr_name in numeric_attr_dict.keys():
				val = float(parsed_event[4])
				parsed_event[4] = "HIGH" if val > numeric_attr_dict[attr_name] else "LOW"
		# 2. Enum unification
		for parsed_event in parsed_events: # Unify the range of all enum variables
			unified_val = self._enum_unification(parsed_event[4])
			if unified_val != '':
				parsed_event[4] = unified_val
				unified_parsed_events.append(parsed_event)
		return unified_parsed_events
	
	def create_data_frame(self, unified_parsed_events: "list[list[str]]"):
		"""This function takes unified events as inputs, and filters non-transition events
		Moreover, it creates the data frame for the transition events
		Finally, this function writes these transition events into data files

		Args:
			unified_parsed_events (list[list[str]]): The unified events (from the unify_value_type function)

		Returns:
			attr_names (list[str]): The list of involved attributes 
			transition_events_states (list[tuple(list[str], ndarray[int])]): The list of tuple (transition events, transited states)
		"""
		attr_names = set()
		transition_events_states = []
		for unified_event in unified_parsed_events: # Get the list of attributes
			attr_names.add(unified_event[2])
		attr_names = list(attr_names); attr_names.sort()
		last_states = [0] * len(attr_names) # The initial states are all 0
		for unified_event in unified_parsed_events: # Filering of non-transition events results in a deduction of len(unified_parsed_events) - len(transition_event_str) events
			attr_name = unified_event[2]; val = unified_event[4]
			cur_states = last_states.copy()
			cur_states[attr_names.index(attr_name)] = 1 if val == 'A' else 0
			if cur_states != last_states: # A state transition happens: Record the event and the current state
				transition_events_states.append((unified_event, np.array(cur_states)))
			last_states = cur_states
		fout = open(self.transition_data, 'w+') # Finally, write these transition events into the data file
		for tup in transition_events_states: 
			fout.write(' '.join(tup[0]) + '\n')
		fout.close()
		return attr_names, transition_events_states
	
	def partition_data_frame(self, attr_names=[], transition_events_states=[], partition_config = ()):
		"""Partition the data frame according to the set of triggered attributes

		Args:
			transition_events_states (list, optional): _description_. Defaults to [].
			partition_config (tuple(int, dict), must): The (scheme_id, parameters) tuple deciding how to partition the data
		
		Returns:
			dataframes (list[Dataframe]): The separated data frames
		"""
		assert (len(partition_config) == 2)
		dataframes = []
		states_array = np.stack([tup[1] for tup in transition_events_states], axis=0)
		if partition_config[0] == 0: # No partitioning, use the whole states_array
			dataframe = pp.DataFrame(data=states_array, var_names=attr_names)
			dataframes.append(dataframe)
		elif partition_config[0] == 1: #NOTE: Ad-hoc partitioning scheme here. Partitioning with data_interval = 2 days
			day_criteria = partition_config[1]
			last_timestamp = ''
			seg_points = []
			count = 0
			for tup in transition_events_states: # First get the semengation points
				transition_event = tup[0]
				cur_timestamp = '{} {}'.format(transition_event[0], transition_event[1])
				last_timestamp = cur_timestamp if last_timestamp == '' else last_timestamp
				past_days = ((datetime.fromisoformat(cur_timestamp) - datetime.fromisoformat(last_timestamp)).total_seconds()) / 86400
				if past_days >= day_criteria:
					seg_points.append(count)
					last_timestamp = cur_timestamp
				count += 1
			last_point = 0
			for seg_point in seg_points: # Get the data frame with range [last_point, seg_point]
				dataframe = pp.DataFrame(data=states_array[last_point:seg_point, ], var_names=attr_names)
				dataframes.append(dataframe)
				last_point = seg_point
		elif partition_config[0] == 2: # TODO: Partitioning according to eta (maximal inter-event intervals) and tau (maximum # of attribute sequences)
			eta = partition_config[1]['eta']; tau = partition_config[1]['tau']
			last_timestamp = '1990-01-01 00:00:00.200'
			transaction = []; transaction_attrs = set()
			transactions_dict = {}
			for tup in transition_events_states:
				transition_event = tup[0]
				cur_timestamp = '{} {}'.format(transition_event[0], transition_event[1])
				time_flag = (datetime.fromisoformat(cur_timestamp) - datetime.fromisoformat(last_timestamp)).total_seconds() < eta
				size_flag = len(transaction_attrs) < tau
				attr_flag = transition_event[2] in transaction_attrs
				if (time_flag) and (size_flag or attr_flag): # If the current event can be added to the current transaction
					transaction.append(tup)
					transaction_attrs.add(transition_event[2])
				else: # Record the previous transaction, initialize a new transaction
					transaction_signature = frozenset(transaction_attrs)
					if len(transaction_signature) > 1:
						transactions_dict[transaction_signature] = [] if transaction_signature not in transactions_dict.keys() else transactions_dict[transaction_signature]
						transactions_dict[transaction_signature].append(transaction)
					transaction = []; transaction_attrs = set()
					transaction.append(tup)
					transaction_attrs.add(transition_event[2])
				last_timestamp = cur_timestamp
			signatures = list(transactions_dict.keys())
		return dataframes
	
	def initiate_data_preprocessing(self, partition_config=()):
		"""The starting function for preprocessing data
		"""
		parsed_events = self.sanitize_raw_events()
		unified_parsed_events = self.unify_value_type(parsed_events)
		attr_names, transition_events_states = self.create_data_frame(unified_parsed_events)
		dataframes = self.partition_data_frame(attr_names, transition_events_states, partition_config)
		return attr_names, dataframes
	
	def deprecated_instrument_data(self):
		'''
		[Deprecated function]
		Instrument the dataset by randomly generating automation rules.
		'''
		num_rules = random.randint(1, 3); i = 0
		selected_triggers = []
		while True:
			selected_trigger = random.choice(list(self.sensor_count_dict.keys()))
			for chosen_trigger in selected_triggers: # Avoid two consecutive device event, both triggering responser event (not real-world case)
				if ('{} {}'.format(chosen_trigger, selected_trigger) in self.dependency_interval_dict.keys()) or ('{} {}'.format(selected_trigger, chosen_trigger) in self.dependency_interval_dict.keys()):
					continue
			selected_responser = random.choice(list(self.actuator_count_dict.keys()))
			potential_automaiton_interaction = '{} {}'.format(selected_trigger, selected_responser)
			if (potential_automaiton_interaction not in list(self.dependency_interval_dict.keys())) and (selected_trigger not in self.instrumented_automation_rule_dict.keys()):
				selected_triggers.append(selected_trigger)
				self.instrumented_automation_rule_dict[selected_trigger] = selected_responser
				i += 1
				if i == num_rules:
					break
		fin = open(self.datafile, 'r'); fout = open(self.datafile + '-temp', 'w+')
		last_date = ''; last_time = ''; last_device = ''; last_device_state = ''
		count = 0
		for line in fin.readlines():
			if (last_device in self.instrumented_automation_rule_dict.keys()) and (last_device_state == 'A'): # Check if the device in the last record matches the trigger name
				fout.write('{} {} {} A Control4-Automation\n'.format(last_date, last_time, self.instrumented_automation_rule_dict[last_device])) # NOTE: Later here the time should be replaced.
				count += 1
			inp = line.strip().split(' ')
			last_date = inp[0]; last_time = inp[1]; last_device = inp[2]; last_device_state = inp[3]
			fout.write(line)
		fin.close()
		fout.close()
		os.system('mv {} {}'.format(self.datafile + '-temp', self.datafile))
		self.get_device_statistics()
	
	def deprecated_partition_training_testing_data(self, testing_piece_index):
		"""
		[Deprecated function]
		"""
		fin = open(self.datafile, 'r')
		fout_training = open(self.datafile + '-training', 'w+')
		fout_testing = open(self.datafile + '-testing', 'w+')
		len_rows = len(fin.readlines())
		testing_range = [(testing_piece_index * 1.0 / CROSS_VALIDATION_PIECE ) * len_rows, ((testing_piece_index + 1) * 1.0 / CROSS_VALIDATION_PIECE ) * len_rows]
		line_no = 0
		fin.seek(0)
		for line in fin.readlines():
			if line_no >= testing_range[0] and line_no < testing_range[1]:
				fout_testing.write(line)
			else:
				fout_training.write(line)
			line_no += 1
		fin.close()
		fout_training.close()
		fout_testing.close()