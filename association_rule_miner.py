from pymining import itemmining, assocrules, perftesting
from apyori import apriori
import src.event_processing as evt_proc
import time

class ARMer():

    def __init__(self, dataset, event_processor, tau_max) -> None:
        self.dataset = dataset
        self.tau_max = tau_max
        self.event_processor = event_processor
        self.transactions_list = self._transactions_generation()
    
    def _transactions_generation(self):
        transactions_list = []
        for dataframe in self.event_processor.dataframes: # For each dataframe, construct a transaction list
            var_names = dataframe.var_names; state_array = dataframe.values
            transactions = []
            for row in state_array:
                transaction = tuple([var_names[x] for x in range(len(row)) if row[x] == 1])
                if len(transaction) > 0:
                    transactions.append(transaction)
            transactions_list.append(transactions)
        return transactions_list
    
    def association_rule_mining(self, min_support, min_confidence):
        association_rules_dict = {}
        frame_id = 0
        for transactions in self.transactions_list:
            start = time.time()
            relim_input = itemmining.get_relim_input(transactions)
            item_sets = itemmining.relim(relim_input, min_support=min_support)
            rules = assocrules.mine_assoc_rules(item_sets, min_support=min_support, min_confidence=min_confidence)
            end = time.time()
            print(rules)
            print("# of transactions: {}".format(len(transactions)))
            print("Consumed time for pymining package: {} miniutes".format((end - start)*1.0/60))
            print("# of discovered rules: {}".format(len(rules)))

            start = time.time()
            results = list(apriori(transactions, min_support=0.1, min_confidence=0.1))
            end = time.time()
            print(results)
            print("# of transactions: {}".format(len(transactions)))
            print("Consumed time for apyori package: {} miniutes".format((end - start)*1.0/60))
            print("# of discovered rules: {}".format(len(results)))
            break
            #association_rules_dict[frame_id] = {}



if __name__ == '__main__':
    dataset = 'hh101'; partition_config = 20
    tau_max = 1
    event_preprocessor = evt_proc.Hprocessor(dataset)
    attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
    armer = ARMer(dataset, event_preprocessor, tau_max)
    armer.association_rule_mining(min_support=4 * partition_config, min_confidence=0.5)
