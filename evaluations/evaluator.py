from operator import truth
from statistics import mean
import os
from os import sep
import subprocess
from unittest import case
import pandas as pd
import statistics
import random
import sys
import numpy as np
from pathlib import Path
from pymining import itemmining, assocrules, perftesting
import matplotlib.pyplot as plt
from interactiongraph import InteractionGraph

def _analysis_statistics_report(tp_list, tn_list, fp_list, fn_list):
	sum_tp = sum(tp_list)
	sum_tn = sum(tn_list)
	sum_fp = sum(fp_list)
	sum_fn = sum(fn_list)
	accuracy = (sum_tp + sum_tn) * 1.0 / (sum_tp + sum_tn + sum_fp + sum_fn) if sum_tp + sum_tn + sum_fp + sum_fn > 0 else 0
	precision = sum_tp * 1.0 / (sum_tp + sum_fp) if sum_tp + sum_fp > 0 else 0
	recall = sum_tp * 1.0 / (sum_tp + sum_fn) if sum_tp + sum_fn > 0 else 0
	specificity = sum_tn * 1.0 / (sum_tn + sum_fp) if sum_tn + sum_fp > 0 else 0
	f1_score = 2.0 * sum_tp / (2 * sum_tp + sum_fp + sum_fn) if 2 * sum_tp + sum_fp + sum_fn > 0 else 0
	return accuracy, precision, recall, specificity, f1_score

class Plotter:

	def plot_discovery_accuracy_comparison(self, datasets, arm_precisions, arm_recalls, causal_precisions, causal_recalls): # Plot the discovery accuracy using causalIoT and ARM
		x_axis = datasets
		assert(len(x_axis) == len(arm_precisions) == len(arm_recalls) == len(causal_precisions) == len(causal_recalls))
		for x in x_axis:
			plt.axvline(x=x, color='grey')
		plt.plot(x_axis, arm_precisions, 'g*-', label='arm precision')
		plt.plot(x_axis, arm_recalls, 'g+-', label='arm recall')
		# plt.plot(x_axis, arm_f1, 'g--', label='arm fscore')
		plt.plot(x_axis, causal_precisions, 'r*-', label='causal precision')
		plt.plot(x_axis, causal_recalls, 'r+-', label='causal recall')
		# plt.plot(x_axis, causal_f1, 'r--', label='causal fscore')
		plt.title('Interaction discovery accuracy')
		plt.xlabel('Testbed')
		plt.legend()
		plt.show()

	def plot_histogram(self, name_list = [], y_label = '', method_num = 1, method_list=[], stats_list=[], fname = 'temp.pdf'):
		assert(method_num == len(stats_list))
		color_list = ['g', 'r', 'b']
		x = list(range(len(name_list)))
		total_width, n = 0.9, method_num
		width = total_width * 1.0 / method_num
		plt.figure(figsize=(26,24))
		plt.xticks([y + 1.0 * total_width / 4 for y in x], name_list)
		for i in range(method_num):
			plt.bar(x, stats_list[i], width=width, label=method_list[i], tick_label=None, edgecolor='k',  fc=color_list[i])
			for j, stat in enumerate(stats_list[i]):
				plt.text(x=x[j], y=stat+0.02, s=str("{:.1%}".format(stat)), ha='center', va='center', weight='bold', size=40)
			for j in range(len(x)):
				x[j] += width
		
		# plt.ylabel(y_label, size=40)
		plt.tick_params(labelsize=40)
		plt.legend(loc='upper left', prop={'weight':'bold','size':35})
		plt.savefig('../readings-and-drafts/Drafts/image/{}'.format(fname), dpi=600, format='pdf', bbox_inches="tight", pad_inches=0)

class Evaluator:

	def __init__(self, dataset, act_label_list, instrumented_automation_rule_dict, true_interactions_dict, interaction_graphs):
		self.data_path = '/Users/jc/research/causalIoT/data'
		self.dataset = dataset
		self.act_label_list = act_label_list
		self.adj_path_prefix = '{}/{}/adj/'.format(self.data_path, self.dataset)
		self.armadj_path_prefix = '{}/{}/armadj/'.format(self.data_path, self.dataset)
		self.stableadj_path_prefix = '{}/{}/stableadj/'.format(self.data_path, self.dataset)
		self.mat_path_prefix = '{}/{}/mat/'.format(self.data_path, self.dataset)
		self.partition_path_prefix = '{}/{}/partition/'.format(self.data_path, self.dataset)
		self.timeorder_path_prefix = '{}/{}/timeorder/'.format(self.data_path, self.dataset)
		self.priortruth_path_prefix = '{}/{}/priortruth/'.format(self.data_path, self.dataset)
		self.truth_path_prefix = '{}/{}/truth/'.format(self.data_path, self.dataset)
		self.anomaly_path_prefix = '{}/{}/anomaly/'.format(self.data_path, self.dataset)
		self.datafile = '{}/{}/data'.format(self.data_path, self.dataset) # The default name of the original data file should be "data"
		self.training_data = self.datafile + '-training'
		self.testing_data = self.datafile + '-testing'
		self.tp_list = []; self.tn_list = []; self.fp_list = []; self.fn_list = []
		self.arm_tp_list = []; self.arm_tn_list = []; self.arm_fp_list = []; self.arm_fn_list = []
		self.stable_tp_list = []; self.stable_tn_list = []; self.stable_fp_list = []; self.stable_fn_list = []
		self.s_s_count = self.s_a_count = self.a_a_count = 0
		self.case_lines_dict: 'dict[int, list[int]]' = {}
		self.case_lines_dict[0] = []
		self.instrumented_automation_rule_dict: 'dict[str, str]' = instrumented_automation_rule_dict
		self.true_interactions_dict: 'dict[str, np.ndarray]' = true_interactions_dict
		self.interaction_graphs: 'dict[str, InteractionGraph]' = interaction_graphs

	def discovery_accuracy_analysis(self):
		for act_label in self.act_label_list:
			os.system("Rscript stable-pc.R {} {} {} > /dev/null".format(self.dataset, act_label, 0.01)) # Call stable-PC for comparisons
			# Get number of devices, number of edges, and the recall and precision 
			adj_mat = self.interaction_graphs[act_label].adjacency_mat
			truth_mat = self.true_interactions_dict[act_label]
			armadj_mat = pd.read_table("{}{}/mixed.mat".format(self.armadj_path_prefix, act_label), sep= ' ')
			stableadj_mat = pd.read_table("{}{}/mixed.mat".format(self.stableadj_path_prefix, act_label), sep= ' ')
			assert(adj_mat.shape[0] == adj_mat.shape[1] == truth_mat.shape[0] == truth_mat.shape[1] == armadj_mat.shape[0] == armadj_mat.shape[1] == stableadj_mat.shape[0] == stableadj_mat.shape[1])
			col_names = self.interaction_graphs[act_label].device_list
			nvar = len(col_names)
			armadj_mat = armadj_mat.values
			stableadj_mat = stableadj_mat.values
			for i in range(nvar):
				if sum(truth_mat[i]) + sum(truth_mat[:,i]) == 0: # If the sum of the row and the sum of the column equal to 0 (which means that the node has no in/out edges, we don't count this node)
					nvar = nvar - 1
			self._count_trigger_types(truth_mat, col_names)
			self._analyze_discovery_accuracy(truth_mat= truth_mat, adj_mat= adj_mat, flag = 0) # Analyze the accuracy of Temporal-PC
			self._analyze_discovery_accuracy(truth_mat= truth_mat, adj_mat= armadj_mat, flag = 1) # Analyze the accuracy of ARM
			self._analyze_discovery_accuracy(truth_mat= truth_mat, adj_mat= stableadj_mat, flag = 2) # Analyze the accuracy of Stable-PC
		causal_accuracy, causal_precision, causal_recall, causal_specificity, causal_f1 = _analysis_statistics_report(self.tp_list, self.tn_list, self.fp_list, self.fn_list)
		arm_accuracy, arm_precision, arm_recall, arm_specificity, arm_f1 = _analysis_statistics_report(self.arm_tp_list, self.arm_tn_list, self.arm_fp_list, self.arm_fn_list)
		stable_accuracy, stable_precision, stable_recall, stable_specificity, stable_f1 = _analysis_statistics_report(self.stable_tp_list, self.stable_tn_list, self.stable_fp_list, self.stable_fn_list)
		print("Accuracy Summary for dataset: {}".format(self.dataset))
		print("	* Interactions: sum, s-s, s-a, a-a = {}, {}, {}, {}".format(self.s_s_count + self.s_a_count + self.a_a_count, self.s_s_count, self.s_a_count, self.a_a_count))
		print("	* Causal, ARM, Stable precisions: {}, {}, {}".format(causal_precision, arm_precision, stable_precision))
		print("	* Causal, ARM, Stable recalls: {}, {}, {}".format(causal_recall, arm_recall, stable_recall))
	
	def inspection_accuracy_analysis(self):
		print("instrumented automation rules: {}".format(self.instrumented_automation_rule_dict))
		automation_tps = []; automation_fps = []; automation_fns = []
		for act_label in self.act_label_list:
			device_list = self.interaction_graphs[act_label].device_list
			label_mat = self.interaction_graphs[act_label].edge_label_mat
			automation_tp = 0; automation_fp = 0; automation_fn = 0
			for true_trigger, true_responser in self.instrumented_automation_rule_dict.items(): # Calculate the automation discovery accuracy
				if label_mat[device_list.index(true_trigger), device_list.index(true_responser)] == 2:
					automation_tp += 1
				else:
					automation_fn += 1
					print("Missing detections of automation rule: {} -> {}".format(true_trigger, true_responser))
			automation_fp = np.count_nonzero(label_mat == 2) - automation_tp
			automation_tps.append(automation_tp); automation_fps.append(automation_fp); automation_fns.append(automation_fn)
			automation_tp_mean = statistics.mean(automation_tps); automation_fp_mean = statistics.mean(automation_fps); automation_fn_mean = statistics.mean(automation_fns)
			automation_precision = (1.0 * automation_tp_mean) / (automation_tp_mean + automation_fp_mean); automation_recall = (1.0 * automation_tp_mean) / (automation_tp_mean + automation_fn_mean)
		print("	* Automation Detection rate: precision, recall = {}, {}".format(automation_precision, automation_recall))
	
	def compare_arm_causal(self): # For Observation 3 in our paper.
		for act_label in self.act_label_list:
			print("	* ACT Label: {}".format(act_label))
			adj_mat = pd.read_table("{}{}/mixed.mat".format(self.adj_path_prefix, act_label), sep= ' ')
			armadj_mat = pd.read_table("{}{}/mixed.mat".format(self.armadj_path_prefix, act_label), sep= ' ')
			truth_mat = pd.read_table("{}{}/mixed.mat".format(self.priortruth_path_prefix, act_label), sep= ' ')
			assert(adj_mat.shape[0] == adj_mat.shape[1] == truth_mat.shape[0] == truth_mat.shape[1] == armadj_mat.shape[0] == armadj_mat.shape[1])
			col_names = list(adj_mat.columns)
			nvar = len(col_names)
			adj_mat = adj_mat.values
			armadj_mat = armadj_mat.values
			truth_mat = truth_mat.values
			rows = cols = len(truth_mat)
			for i in range(rows):
				for j in range(cols):
					if truth_mat[i, j] == 0 and adj_mat[i, j] == 0 and armadj_mat[i, j] == 1:
						print("A false positive for ARM: {} -> {}".format(col_names[i], col_names[j]))

	def compare_stable_causal(self):
		stable_skeleton_fps_ratios = []
		stable_orientation_fps_ratios = []
		stable_fps = []
		stable_skeleton_fps = []
		stable_orientation_fps = []
		for act_label in self.act_label_list:
			stable_fp = 0
			stable_skeleton_fp = 0
			stable_orientation_fp = 0
			adj_mat = pd.read_table("{}{}/mixed.mat".format(self.adj_path_prefix, act_label), sep= ' ')
			stableadj_mat = pd.read_table("{}{}/mixed.mat".format(self.stableadj_path_prefix, act_label), sep= ' ')
			truth_mat = pd.read_table("{}{}/mixed.mat".format(self.priortruth_path_prefix, act_label), sep= ' ')
			assert(adj_mat.shape[0] == adj_mat.shape[1] == truth_mat.shape[0] == truth_mat.shape[1] == stableadj_mat.shape[0] == stableadj_mat.shape[1])
			col_names = list(adj_mat.columns)
			nvar = len(col_names)
			adj_mat = adj_mat.values
			stableadj_mat = stableadj_mat.values
			truth_mat = truth_mat.values
			rows = cols = len(truth_mat)
			for i in range(rows):
				for j in range(cols):
					if truth_mat[i, j] == truth_mat[j, i] == adj_mat[i, j] == adj_mat[j, i] == 0 and stableadj_mat[i, j] == 1:
						# print("A false positive (wrong skeleton) in Stable: {} -> {}".format(col_names[i], col_names[j]))
						stable_fp += 1
						stable_skeleton_fp += 1
					elif truth_mat[i, j] == adj_mat[i, j] == 0 and truth_mat[j, i] == adj_mat[j, i] == 1 and stableadj_mat[i, j] == 1:
						# print("A false positive (wrong orientation) in Stable: {} -> {}".format(col_names[i], col_names[j]))
						stable_fp += 1
						stable_orientation_fp += 1
			stable_fps.append(stable_fp)
			stable_skeleton_fps.append(stable_skeleton_fp)
			stable_orientation_fps.append(stable_orientation_fp)
		assert(sum(stable_fps) == sum(stable_orientation_fps) + sum(stable_skeleton_fps))
		stable_skeleton_fps_ratios.append(sum(stable_skeleton_fps) * 1.0 / sum(stable_fps))
		stable_orientation_fps_ratios.append(sum(stable_orientation_fps) * 1.0 / sum(stable_fps))
		print("Comparison results between Stable PC and CausalIoT for dataset {}:".format(self.dataset))
		print("	* Stable FP ratios (caused by skeletons):{}".format(stable_skeleton_fps_ratios))
		print("	* Stable FP ratios (caused by orientations):{}".format(stable_orientation_fps_ratios))
			
	def _analyze_discovery_accuracy(self, truth_mat, adj_mat, flag = 0): # Flag denotes the current analysis object. 0 - causalIoT, 1 - ARM, 2 - Stable-PC
		rows = cols = len(truth_mat)
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		for i in range(rows):
			for j in range(cols):
				if truth_mat[i, j] == adj_mat[i, j] == 1:
					tp += 1
				elif truth_mat[i, j] == adj_mat[i, j] == 0:
					tn += 1
				elif truth_mat[i, j] == 1 and adj_mat[i, j] == 0:
					# print("		fn [act, location]: [{}, ({},{})]".format(act_label, col_names[i], col_names[j]))
					fn += 1
				elif truth_mat[i, j] == 0 and adj_mat[i, j] == 1:
					# print("		fp [act, location]: [{}, ({},{})]".format(act_label, col_names[i], col_names[j]))
					fp += 1
		if flag == 0:
			self.tp_list.append(tp)
			self.tn_list.append(tn)
			self.fn_list.append(fn)
			self.fp_list.append(fp)
		elif flag == 1:
			self.arm_tp_list.append(tp)
			self.arm_tn_list.append(tn)
			self.arm_fn_list.append(fn)
			self.arm_fp_list.append(fp)
		elif flag == 2:
			self.stable_tp_list.append(tp)
			self.stable_tn_list.append(tn)
			self.stable_fn_list.append(fn)
			self.stable_fp_list.append(fp)
	
	def _count_trigger_types(self, truth_mat, col_names):
		a_list = []
		if self.dataset.startswith('hh'): # Determine the feature of the actuator
			a_list = ['D', 'L']
		rows = cols = len(truth_mat)
		for i in range(rows):
			for j in range(cols):
				if truth_mat[i, j] == 1:
					device1_name = col_names[i]
					device2_name = col_names[j]
				else:
					continue
				if any(map(device1_name.startswith, a_list)) and any(map(device2_name.startswith, a_list)): # This is an actuator-actuator interaction
					print("a-a connect: {} - {}".format(device1_name, device2_name))
					self.a_a_count += 1
				elif any(map(device1_name.startswith, a_list)) or any(map(device2_name.startswith, a_list)): # This is an actuator-sensor interaction
					self.s_a_count += 1
				else:
					self.s_s_count += 1

	def anomaly_generation(self, case_id: 'int' = 1, max_anomaly = 30):
		Path(self.anomaly_path_prefix).mkdir(parents=True, exist_ok=True)
		testing_file = open(self.testing_data, 'r'); anomaly_file = open("{}{}".format(self.anomaly_path_prefix, str(case_id)), 'w+')
		num_log = len(testing_file.readlines()); num_anomaly = max(int(0.001 * num_log), max_anomaly)

		act_devlist_dict: 'dict[str, list[str]]' = {}; act_truthmat_dict = {}
		for act_label in self.act_label_list: # Read and store the golden standard
			df = pd.read_csv('{}{}/{}'.format(self.priortruth_path_prefix, act_label, 'mixed.mat'), delim_whitespace=True)
			act_devlist_dict[act_label] = list(df.columns.values)
			act_truthmat_dict[act_label] = df.to_numpy()
			interaction_graph = self.interaction_graphs[act_label]

		involved_devices = list(interaction_graph.device_list); involved_sensors = []; involved_actuators = []
		anomalous_device = ''
		self.case_lines_dict[case_id] = []
		if self.dataset.startswith('hh'): # NOTE: In hh dataset, there are only motion sensors as the sensor.
			involved_sensors = list([x for x in involved_devices if x.startswith('M')])
			involved_actuators = list(set(involved_devices) - set(involved_sensors))
		if len(involved_actuators) == 0:
			print("There are no used actuators for the act_label {}!".format(interaction_graph.act_label))
			exit(0)
		if case_id in [3, 4]: # The Case 3 and Case 4 
			anomalous_device = random.choice(involved_sensors) if case_id == 3 else random.choice(involved_actuators) # Determine the anomalous device
			suitable_insertion_locations = []; lineno = 0; last_dev = ''
			testing_file.seek(0)
			for line in testing_file.readlines(): # Find all suitable anomaly insertion locations, according to the truth matrix
				lineno += 1
				inp = line.strip().split(' ')
				date = inp[0]; time = inp[1]; dev_name = inp[2]; dev_state = inp[3]; dev_type = inp[4]
				cur_hour = int(time.split(':')[0]); cur_act_label = '{}_{}'.format(cur_hour, (cur_hour+1) % 24)
				devlist = act_devlist_dict[cur_act_label]; truthmat = act_truthmat_dict[cur_act_label]
				assert(len(devlist) == truthmat.shape[0] == truthmat.shape[1])
				if (dev_name != anomalous_device) and (truthmat[devlist.index(dev_name), devlist.index(anomalous_device)] == 0):
					suitable_insertion_locations.append(lineno)
				last_dev = dev_name
			selected_lines = random.sample(suitable_insertion_locations, min(len(suitable_insertion_locations), num_anomaly))
			lineno = 0; anomaly_lineno = 0
			testing_file.seek(0)
			for line in testing_file.readlines(): # After randomly selecting num_anomaly locations, generate the anomaly file
				lineno += 1
				anomaly_file.write(line)
				anomaly_lineno += 1
				if lineno in selected_lines:
					inp = line.strip().split(' ')
					date = inp[0]; time = inp[1]; dev_name = inp[2]
					anomaly_file.write('{} {} {} A Anomaly\n'.format(date, time, anomalous_device)) # NOTE: Here we didn't care the date, time, and dev_type. Also, we set the anomaly device status is A by default.
					anomaly_lineno += 1; self.case_lines_dict[case_id].append(anomaly_lineno)
		elif case_id == 5:
			while True: # Determine the trigger and responser of the malicious automation rule: The occurrence of the trigger should be larger than 0, and there should be suitable responser
				selected_trigger = involved_sensors[random.randint(0, len(involved_sensors) - 1)]; selected_responser = ''
				occ_flag = 0
				testing_file.seek(0)
				for line in testing_file.readlines():
					inp = line.strip().split(' '); dev_name = inp[2]
					if dev_name == selected_trigger:
						occ_flag = 1
						break
				if occ_flag == 0: # Reselect a trigger, because the current trigger has occurrence 0
					continue
				suitable_responsers = [x for x in involved_actuators if interaction_graph.merged_adjacency_mat[involved_devices.index(selected_trigger),involved_actuators.index(x)] == 0]
				if len(suitable_responsers) > 0:
					selected_responser = suitable_responsers[0]
					break
			lineno = 0; anomaly_lineno = 0
			last_date = ''; last_time = ''; last_dev = ''; last_dev_state = ''
			testing_file.seek(0)
			for line in testing_file.readlines(): # Traverse the file, and find locations which are next to the selected triggers
				lineno += 1
				inp = line.strip().split(' ')
				date = inp[0]; time = inp[1]; dev_name = inp[2]; dev_state = inp[3]; dev_type = inp[4]
				if last_dev == selected_trigger and last_dev_state == 'A':
					anomaly_file.write('{} {} {} A Anomaly\n'.format(last_date, last_time, selected_responser))
					anomaly_lineno += 1; self.case_lines_dict[case_id].append(anomaly_lineno)
				anomaly_file.write(line); anomaly_lineno += 1
				last_dev = dev_name; last_dev_state = dev_state; last_date = date; last_time = time
		elif case_id in [6, 7, 8]:
			devices_list = involved_sensors if case == 6 else involved_actuators
			suitable_del_locations = []; lineno = 0
			last_dev = ''
			testing_file.seek(0)
			for line in testing_file.readlines(): # Get all locations of the anomalous device, and prepare to delete the record
				lineno += 1
				inp = line.strip().split(' ')
				date = inp[0]; time = inp[1]; dev_name = inp[2]; dev_state = inp[3]; dev_type = inp[4]
				if (case_id in [6, 7]) and (dev_name in devices_list) and (random.uniform(0, 1) > 0.05) and (lineno -1 not in suitable_del_locations):
					suitable_del_locations.append(lineno)
				elif (case_id in [8]) and (last_dev in self.instrumented_automation_rule_dict.keys()) and (dev_name == self.instrumented_automation_rule_dict[last_dev]):
					suitable_del_locations.append(lineno)
				last_dev = dev_name
			selected_lines = random.sample(suitable_del_locations, min(len(suitable_del_locations), num_anomaly))
			selected_lines.sort()
			print("Anomaly Lines: {}".format(selected_lines))
			lineno = 0; anomaly_lineno = 0
			testing_file.seek(0)
			for line in testing_file.readlines():
				lineno += 1
				if lineno in selected_lines:
					self.case_lines_dict[case_id].append(anomaly_lineno)
					continue
				anomaly_file.write(line)
				anomaly_lineno += 1

		testing_file.close()
		anomaly_file.close()
		self.case_lines_dict[case_id].sort()
		# print("# of inserted anomalies for case {}: {}. {}".format(case_id, len(self.case_lines_dict[case_id]), self.case_lines_dict[case_id]))
		return anomalous_device, self.case_lines_dict[case_id]

	def detection_accuracy_analysis(self, case_id, reported_anomaly_lines, promopt_lines):
		true_anomaly_lines = self.case_lines_dict[case_id]
		tp = 0; fn = 0; fp = 0; precision = 0; recall = 0
		tp_list = []; fn_list = []; fp_list = []
		for true_anomaly_line in true_anomaly_lines:
			if (true_anomaly_line in reported_anomaly_lines) or (true_anomaly_line in promopt_lines):
				tp += 1
				tp_list.append(true_anomaly_line)
			else:
				fn += 1
				fn_list.append(true_anomaly_line)
		for reported_anomaly_line in reported_anomaly_lines:
			if (reported_anomaly_line not in true_anomaly_lines) and (reported_anomaly_line not in promopt_lines):
				fp += 1
				fp_list.append(reported_anomaly_line)
		recall = (tp * 1.0) / len(true_anomaly_lines) if len(true_anomaly_lines) > 0 else 0
		print("tp, fn, fp, recall = {}, {}, {}, {}".format(tp, fn, fp, recall))
		print("The false negative list: {}".format(fn_list))
		print("The false positive list: {}".format(fp_list))