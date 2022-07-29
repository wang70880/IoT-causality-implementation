import os, sys
from src.event_processing import Hprocessor

dataset = sys.argv[1]; verbosity = int(sys.argv[2])
event_preprocessor:'Hprocessor' = Hprocessor(dataset, verbosity)
event_preprocessor.initiate_data_preprocessing()