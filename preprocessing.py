import os, sys
from src.event_processing import Hprocessor, Cprocessor
from time import time

def _elapsed_minutes(start):
    return (time()-start) * 1.0 / 60

dp_start = time()
dataset = sys.argv[1]; partition_days = int(sys.argv[2]); training_ratio = float(sys.argv[3]); verbosity = int(sys.argv[4])
event_preprocessor = None
if dataset.startswith('hh'):
    event_preprocessor:'Hprocessor' = Hprocessor(dataset, partition_days, training_ratio, verbosity)
elif dataset.startswith('contextact'):
    event_preprocessor:'Cprocessor' = Cprocessor(dataset, partition_days, training_ratio, verbosity)
event_preprocessor.initiate_data_preprocessing()
print("[Data sanitization] Complete ({} minutes)".format(_elapsed_minutes(dp_start)))
