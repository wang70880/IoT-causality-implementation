from fileinput import lineno
from interactiongraph import InteractionGraph

def _timestr2Seconds(timestr):
	timestamp = timestr.split('.')[0]
	hh, mm, ss = timestamp.split(':')
	sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
	return sec

class Detector():

	def __init__(self, dataset: str, act_label_list: 'list[str]', interaction_graphs: 'dict[str, InteractionGraph]'):
		self.dataset = dataset
		self.data_path = '/Users/jc/research/causalIoT/data'
		self.datafile = '{}/{}/data'.format(self.data_path, self.dataset)
		self.testing_data = self.datafile + '-testing'
		self.anomaly_path_prefix = '{}/{}/anomaly/'.format(self.data_path, self.dataset)
		self.act_label_list = act_label_list
		self.interaction_graphs: 'dict[str, InteractionGraph]' = interaction_graphs
	
	def initialize_interaction_graphs(self, act_label = ''): # Reset the location matrix for each interaction graph
		if act_label == '':
			for interaction_graph in self.interaction_graphs.values():
				interaction_graph.reset()
		else:
			self.interaction_graphs[act_label].reset()

	def anomaly_detection(self, case_id = 1, anomalous_device = '', true_anomaly_lines = []): # Main detection function.
		streamed_inputs = open(self.testing_data, 'r') if case_id == 0 else open("{}{}".format(self.anomaly_path_prefix, str(case_id)), 'r')
		lineno = 0
		last_date = ''; last_time = ''; last_dev = ''
		reported_anomaly_lines = []; promopt_lines = []
		for runtime_log in streamed_inputs.readlines():
			lineno += 1
			inp = runtime_log.strip().split(' ')
			cur_date = inp[0]
			if cur_date != last_date:
				last_date = cur_date; last_dev = ''
			cur_time = inp[1]; cur_dev = inp[2]; cur_dev_state = inp[3]
			cur_hour = int(cur_time.split(":")[0]); act_label = "{}_{}".format(str(cur_hour), str((cur_hour + 1) % 24))
			if act_label not in self.interaction_graphs.keys():
				continue
			interaction_graph: InteractionGraph = self.interaction_graphs[act_label] # Fetch the corresponding interaction graph
			cur_dev_index = interaction_graph.device_list.index(cur_dev)
			if case_id in [3, 4, 5]:
				last_dev = cur_dev if last_dev == '' else last_dev
				last_dev_index = interaction_graph.device_list.index(last_dev)
				if last_dev == cur_dev:
					continue
				else:
					if interaction_graph.transition_matrix[last_dev_index, cur_dev_index] == 0: # The event does not follow the interaction graph.
						if '{} {}'.format(last_dev, cur_dev) in interaction_graph.device_dependency_count_dict.keys(): # The event follows the dependency list: Promopt users
							## print(" *[{}] At line {}, detect a runtime prompt {} -> {}".format(lineno, lineno, last_dev, cur_dev))
							promopt_lines.append(lineno)
							last_dev = cur_dev
						else: # The event neither follows the interaction graph, nor the dependency list: Anomaly
							## print(" *[{}] At line {}, detect a runtime anomaly {} -> {}".format(lineno, lineno, last_dev, cur_dev))
							reported_anomaly_lines.append(lineno)
							last_dev = ''
					else: # This follows a benign interaction
						last_dev = cur_dev
			elif case_id in [6, 7]:
				last_dev = cur_dev if last_dev == '' else last_dev; last_dev_index = interaction_graph.device_list.index(last_dev)
				two_step_matrix = interaction_graph.adjacency_mat @ interaction_graph.adjacency_mat # The pow of the transition matrix represents two steps move forward.
				two_step_merged_matrix = interaction_graph.merged_adjacency_mat @ interaction_graph.merged_adjacency_mat # The pow of the transition matrix represents two steps move forward.
					# print("[Anomaly in Line {}]\n	Current sequence: {} -> {}\n	(adjacency_mat bit, merged_adjacency_mat bit): ({}, {})\n	(two_step_matrix bit, two_step_merged_matrix bit): ({}, {})".format(lineno-1, last_dev, cur_dev, interaction_graph.adjacency_mat[last_dev_index, cur_dev_index], interaction_graph.merged_adjacency_mat[last_dev_index, cur_dev_index], two_step_matrix[last_dev_index, cur_dev_index], two_step_merged_matrix[last_dev_index, cur_dev_index]))
				if (interaction_graph.merged_adjacency_mat[last_dev_index, cur_dev_index] == 0) and (two_step_merged_matrix[last_dev_index, cur_dev_index] != 0): # The one step transition is failed, but a two-step transition exists.
					# print(" * A missing event is detected at line {}".format(lineno-1))
					reported_anomaly_lines.append(lineno-1)
				last_dev = cur_dev
			elif case_id in [8]:
				last_dev = cur_dev if last_dev == '' else last_dev; last_dev_index = interaction_graph.device_list.index(last_dev)
				last_time = cur_time if last_time == '' else last_time
				if (last_dev in interaction_graph.discovered_automation_dict.keys()) and (cur_dev != interaction_graph.discovered_automation_dict[last_dev]):
					reported_anomaly_lines.append(lineno-1)
				last_dev = cur_dev; last_time = cur_time
		reported_anomaly_lines.sort(); promopt_lines.sort()
		return reported_anomaly_lines, promopt_lines