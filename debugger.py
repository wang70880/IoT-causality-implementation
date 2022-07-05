from src.event_processing import Hprocessor
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import os

# PARAM SETTING
dataset = 'hh130'

class DataDebugger():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.preprocessor = Hprocessor(dataset=dataset)

    def initiate_preprocessing(self):
        self.preprocessor.initiate_data_preprocessing()
    
    def validate_discretization(self):
        for dev, tup in self.preprocessor.discretization_dict.items():
            val_list = tup[0]; seg_point = tup[1]
            sns.displot(val_list, kde=False, color='red', bins=1000)
            plt.axvline(seg_point, 0, 1)
            plt.title('State distributions of device {}'.format(dev))
            plt.xlabel('State')
            plt.ylabel('Frequency')
            plt.savefig("temp/image/{}_{}_states.pdf".format(self.dataset, dev))

    def conditional_independence_testing(self):
        pass

if __name__ == '__main__':
    data_debugger = DataDebugger(dataset=dataset)
    data_debugger.initiate_preprocessing()
    data_debugger.validate_discretization()
    os.remove('data/{}/data-transition'.format(dataset))