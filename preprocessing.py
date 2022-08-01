import os, sys
from src.event_processing import Hprocessor
from time import time

def _elapsed_minutes(start):
    return (time()-start) * 1.0 / 60

dp_start = time()
dataset = sys.argv[1]; verbosity = int(sys.argv[2])
event_preprocessor:'Hprocessor' = Hprocessor(dataset, verbosity)
event_preprocessor.initiate_data_preprocessing()
print("[Data sanitization] Complete ({} minutes)".format(_elapsed_minutes(dp_start)))
