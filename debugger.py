from src.event_processing import Hprocessor

# PARAM SETTING
dataset = 'hh130'

class DataDebugger():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.preprocessor = Hprocessor(dataset=dataset)

    def data_preprocessing(self):
        pass

    def dataset_overview(self):
        pass

    def conditional_independence_testing(self):
        pass

if __name__ == '__main__':
    data_debugger = DataDebugger(dataset=dataset)