import os, sys
from src.event_processing import Hprocessor

dataset = sys.argv[1]
event_preprocessor:'Hprocessor' = Hprocessor(dataset)
event_preprocessor.initiate_data_preprocessing()