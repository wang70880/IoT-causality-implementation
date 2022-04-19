from cmath import tau
from code import interact
from collections import defaultdict
from pathlib import Path
from os import name, supports_bytes_environ, system
import os
import sys
import statistics
import datetime
from typing_extensions import runtime
import jenkspy
import random
import numpy as np
import pandas as pd
import aiohttp
import asyncio
import time
from pymining import itemmining, assocrules, perftesting
from matplotlib import pyplot as plt

import src.event_processing as evt_proc
from src.correlation_miner import CausalDiscovery, CausalInference, CorrelationMiner
from src.tigramite.causal_effects import CausalEffects
from src.tigramite.independence_tests.cmisymb import CMIsymb
from src.tigramite import plotting as tp

process_devices_dict = {}
process_index_dict = {}

CROSS_VALIDATION_PIECE = 8

if __name__ == '__main__':
	# sys.stdout = open("results/result.txt", "w+")
	dataset= 'hh101' # NOTE: For testing purposes only
	partion_config = (1, {})
	print("* Initiate data preprocessing.")
	start = time.time()
	event_preprocessor = evt_proc.Hprocessor(dataset)
	attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partion_config)

	end = time.time()
	print("* Data preprocessing finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))

	current_task = 1; total_task = len(dataframes)
	for dataframe in dataframes:
		print("************** Current task, total task = {}, {} **************".format(current_task, total_task))
		correlation_miner = CorrelationMiner(dataframe=dataframe)
		print("* Initiate causal discovery.")
		start = time.time()
		correlation_miner.initiate_causal_discovery(tau_max=1, pc_alpha=0.05)
		end = time.time()
		print("* Causal discovery finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))
		print(correlation_miner.all_parents)

		print("* Initiate causal inference.")
		start = time.time()
		effects_dict = correlation_miner.initiate_causal_inference(tau_max=1)
		end = time.time()
		print("* Causal effect estimation finished. Elapsed time: {} mins".format((end - start) * 1.0 / 60))
		print(effects_dict)
		current_task += 1
		if current_task >= 1:
			break

	# NOTE: Test codes for causal effect estimation module
	# dataframe = dataframes[0]
	# time_series_graph = CausalEffects.get_graph_from_dict({0: [(0, -1), (1, -1), (15, -1), (16, -1)], 1: [(1, -1), (31, -1), (13, -1), (4, -1), (18, -1)], 2: [(2, -1), (1, -1), (18, -1), (14, -1), (32, -1)], 3: [(3, -1), (17, -1), (4, -1)], 4: [(4, -1), (3, -1), (17, -1), (6, -1)], 5: [(5, -1), (17, -1), (6, -1), (31, -1)], 6: [(6, -1), (9, -1), (14, -1), (10, -1)], 7: [(7, -1), (23, -1), (16, -1)], 8: [(8, -1), (24, -1)], 9: [(9, -1)], 10: [(10, -1), (9, -1), (15, -1), (13, -1), (1, -1)], 11: [(11, -1), (2, -1), (27, -1), (14, -1), (16, -1)], 12: [(12, -1), (15, -1), (13, -1), (1, -1)], 13: [(13, -1), (10, -1), (14, -1)], 14: [(14, -1), (1, -1), (9, -1), (11, -1), (31, -1)], 15: [(15, -1), (10, -1), (1, -1), (16, -1)], 16: [(16, -1), (33, -1), (28, -1)], 17: [(17, -1), (4, -1)], 18: [(18, -1), (1, -1), (27, -1), (30, -1), (2, -1), (16, -1)], 19: [(19, -1), (21, -1), (25, -1), (32, -1), (27, -1), (13, -1)], 20: [(20, -1), (7, -1), (25, -1), (17, -1), (23, -1), (19, -1), (1, -1), (32, -1)], 21: [(21, -1), (19, -1), (25, -1), (32, -1), (27, -1)], 22: [(1, -1), (9, -1), (22, -1), (25, -1), (18, -1), (16, -1), (12, -1), (21, -1), (28, -1)], 23: [(23, -1), (7, -1), (20, -1)], 24: [(24, -1), (8, -1)], 25: [(25, -1), (1, -1), (20, -1), (29, -1), (27, -1), (19, -1), (18, -1), (21, -1), (26, -1), (28, -1), (32, -1), (22, -1), (7, -1), (23, -1)], 26: [(26, -1), (25, -1), (20, -1), (27, -1), (18, -1)], 27: [(27, -1), (1, -1), (18, -1), (25, -1), (0, -1), (11, -1)], 28: [(28, -1), (1, -1), (25, -1), (4, -1), (27, -1)], 29: [(25, -1), (17, -1), (1, -1), (15, -1), (27, -1), (28, -1), (26, -1), (34, -1), (29, -1)], 30: [(30, -1), (25, -1), (1, -1), (15, -1)], 31: [(31, -1)], 32: [(32, -1), (31, -1)], 33: [(33, -1), (34, -1)], 34: [(34, -1), (33, -1)]}, tau_max=1)
	# X = [(attr_names.index('M001'), -1)] # M001 at time -1
	# Y = [(attr_names.index('D002'), 0)] # D002 at time 0
	# S = []
	# causal_inference = CausalInference(dataframe=dataframe, time_series_graph=time_series_graph, X=X, Y=Y, S=S)
	# causal_inference.check__XYS_paths()
	# causal_inference.get_optimal_set()
	# causal_inference.check_optimality()

	# intervention_data = [0]*len(X)
	# effect = causal_inference.intervention_effect_prediction(intervention_data)
	# print(effect)
	#current_task = 1; total_task = len(dataframes)
	#for dataframe in dataframes:
	#	start = time.time()
	#	print("Current task, total task = {}, {}".format(current_task, total_task))
	#	causal_miner = CausalDiscovery(dataframe=dataframe, cond_ind_test=CMIsymb(
	#		significance='shuffle_test', n_symbs= None
	#	), verbosity=1
	#	)
	#	all_parents, results = causal_miner.initiate_stablePC()
	#	# results = causal_miner.initiate_PCMCI()
	#	end = time.time()
	#	print("Elapsed time: {}".format((end-start) * 1.0 / 60))

	#	current_task += 1
	#	break

	# datasets= ['hh101', 'hh102', 'hh111'] # It seems that only these three datasets contain s-a interactions.
	# for dataset in datasets:
	# 	testing_piece_index = CROSS_VALIDATION_PIECE - 1
	# 	hh_processor = Hprocessor(dataset)
	# 	hh_processor.preprocess_data()
	# 	hh_processor.instrument_data()
	# 	hh_processor.partition_training_testing_data(testing_piece_index)

	# ############################### Initiate Causal Discovery and Learn from History ###################################
	# 	print("Statistics of dataset {}:".format(dataset))
	# 	print("	* Number of involved devices: {}".format(len(hh_processor.device_count_dict.keys())))
	# 	interaction_graphs = hh_processor.start()

	# ############################### Initiate Evaluations for Causal Discoveries ###################################
	# 	hh_processor.association_rule_mining(TIME_INTERVAL_LIST, min_conf=0.5) # Call ARM for future evaluations
	# 	evaluator = Evaluator(dataset, TIME_INTERVAL_LIST, hh_processor.instrumented_automation_rule_dict, hh_processor.true_interactions_dict, interaction_graphs)
	# 	# evaluator.discovery_accuracy_analysis()
	# 	evaluator.inspection_accuracy_analysis()
	# 	# evaluator.compare_stable_causal(datasets, TIME_INTERVAL_LIST)

	# ############################### Initiate Runtime Anomaly Detection and Evaluations ###################################
	# 	# NOTE: There should be a loop here (for the anomaly case) which traverses generate each anomaly case, initiate the detection, and initiate the evaluation
	# 	case_id = 8; max_anomaly = 50
	# 	anomalous_device, true_anomaly_lines = evaluator.anomaly_generation(case_id, max_anomaly) # Use the evaluator to generate runtime anomalies in testing data
	# 	detector = Detector(dataset, TIME_INTERVAL_LIST, interaction_graphs)
	# 	reported_anomaly_lines, promopt_lines = detector.anomaly_detection(case_id, anomalous_device, true_anomaly_lines)
	# 	evaluator.detection_accuracy_analysis(case_id, reported_anomaly_lines, promopt_lines)

	############################### Initiate Plotting ###################################
	# plotter = Plotter()
	# plotter.plot_histogram(['hh101', 'hh102', 'hh111'], 'precision', 3, ['ARM', 'Temporal-PC', 'Stable-PC'], [[0.5236, 0.64, 0.451], [0.997, 0.989, 0.985], [0.883, 0.868, 0.862]], 'evaluate_precision.pdf')
	# plotter.plot_histogram(['hh101', 'hh102', 'hh111'], 'recall', 3, ['ARM', 'Temporal-PC', 'Stable-PC'], [[0.997, 0.981, 0.981], [0.995, 0.979, 0.985], [0.998, 0.983, 0.986]], 'evaluate_recall.pdf')
