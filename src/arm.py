import numpy as np
from pymining import itemmining, assocrules, perftesting
from apyori import apriori
from src.tigramite.tigramite import data_processing as pp
from src.genetic_type import DataFrame, AttrEvent, DevAttribute
from src.event_processing import Hprocessor

class ARMer():

    def __init__(self, frame:'DataFrame', min_support, min_confidence) -> None:
        self.frame = frame
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = self._transactions_generation()
    
    def _transactions_generation(self):
        # Return variables
        transactions = []
        # Auxillary variables
        index_device_dict:'dict[DevAttribute]' = self.frame.index_device_dict
        training_array:'np.ndarray' = self.frame.training_dataframe.values
        for states in training_array:
            # JC NOTE: When applying association rule mining, a criteria should be given (Here we use the activation as the criteria)
            transaction = tuple([index_device_dict[x].name for x in range(len(states)) if states[x] == 1])
            if len(transaction) > 0:
                transactions.append(transaction)
        return transactions
    
    def association_rule_mining(self):
        # Auxillary variables
        name_device_dict:'dict[DevAttribute]' = self.frame.name_device_dict
        n_vars = len(name_device_dict.keys())
        # Return variables
        association_array = np.zeros((n_vars, n_vars))

        relim_input = itemmining.get_relim_input(self.transactions)
        item_sets = itemmining.relim(relim_input, min_support=self.min_support)
        rules = assocrules.mine_assoc_rules(item_sets, min_support=self.min_support, min_confidence=self.min_confidence)
        # The rules are in the format of list[tuple], e.g., [(frozenset(['e']), frozenset(['b', 'd']), 2, 0.6666666666666666)]
        # We here transform it to an adjacent array. The adjacency array does not contain time lagged information.
        for rule in rules:
            preceding_devices = rule[0]
            consecutive_devices = rule[1]
            interactions = [(x, y) for x in preceding_devices for y in consecutive_devices]
            print("Interaction lists: {}".format(interactions))
            for interaction in interactions:
                association_array[(name_device_dict[interaction[0]].index, name_device_dict[interaction[1]].index)] = 1
        print("Total number of interactions: {}".format(np.sum(association_array)))
        return association_array



if __name__ == '__main__':
    dataset = 'hh130'; partition_days=100; training_ratio=0.8
    int_frame_id = 0; min_support = 20; min_confidence = 0.5

    preprocessor = Hprocessor(dataset=dataset, partition_days=partition_days, training_ratio=training_ratio)
    preprocessor.data_loading()

    frame:'DataFrame' = preprocessor.frame_dict[int_frame_id]
    dataframe:pp.DataFrame = frame.training_dataframe; attr_names = frame.var_names
    armer = ARMer(frame=frame, min_support=min_support, min_confidence=min_confidence)
    association_array = armer.association_rule_mining()