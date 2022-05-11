from pymining import itemmining, assocrules, perftesting
import event_processing as evt_proc

class ARMer():

    def __init__(self, dataset, event_processor, tau_max) -> None:
        self.dataset = dataset
        self.tau_max = tau_max
        self.event_processor = event_processor
        self.transaction_dict = self._transactions_generation()
    
    def _transactions_generation(self):
        transactions_list = []
        for dataframe in self.event_processor.dataframes:
            var_names = dataframe.var_names; state_array = dataframe.values
            transactions = []
            for row in state_array:
                transaction = tuple([var_names[x] for x in range(len(row)) if row[x] == 1])
                print(transaction)
                transactions.append(transaction)
            transactions_list.append(transactions)


if __name__ == '__main__':
    dataset = 'hh101'; partition_config = 20
    tau_max = 1
    event_preprocessor = evt_proc.Hprocessor(dataset)
    attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config)
    armer = ARMer(dataset, event_preprocessor, tau_max)
    armer._transactions_generation()