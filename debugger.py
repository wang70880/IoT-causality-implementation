from src.event_processing import Hprocessor
from pprint import pprint

# PARAM SETTING
dataset = 'hh130'

class DataDebugger():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.preprocessor = Hprocessor(dataset=dataset)

    def initiate_preprocessing(self):
        self.preprocessor.initiate_data_preprocessing()

    def dataset_overview(self):
        pprint(self.preprocessor.attr_count_dict)

    def conditional_independence_testing(self):
        pass

if __name__ == '__main__':
    data_debugger = DataDebugger(dataset=dataset)
    data_debugger.initiate_preprocessing()
    data_debugger.dataset_overview()