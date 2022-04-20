from src.event_processing import Hprocessor
class Evaluator():

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.event_processor = Hprocessor(dataset)
    
    def temporal_pair_identification(self, partition_config, tau_max=1):
        """Identification of the temporal correlation pair in the following format.

            (attr_c, attr_o, E[attr_o|attr_c=0], E[attr_o|attr_c=1])

            These pairs are indexed by the time lag tau and the partitioning scheme (i.e., the date).

        Args:
            partition_config (tuple): _description_
            tau_max (int, optional): _description_. Defaults to 1.
        Returns:
            temporal_pair_dict (dict[int, [int, list[tuple]]]): The frame_id-indexed lag-indexed temporal pairs.
        """
        temporal_pair_dict = {}
        self.event_processor.initiate_data_preprocessing(partition_config=partition_config)
        print(self.event_processor.frame_dict[0][])
        return temporal_pair_dict

if __name__ == '__main__':
    evaluator = Evaluator('hh101')
    partition_config = (1, 30)
    tau_max = 1
    evaluator.temporal_pair_identification(partition_config=partition_config, tau_max=tau_max)