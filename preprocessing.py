import os, sys
from src.event_processing import Hprocessor

dataset = sys.argv[1]
partition_config = int(sys.argv[2])
apply_bk = int(sys.argv[3])
event_preprocessor:'Hprocessor' = Hprocessor(dataset)
event_preprocessor.initiate_data_preprocessing(partition_config=partition_config, training_ratio=0.9)