import src.event_processing as evt_proc

dataset='hh101'
partition_config =12
event_preprocessor = evt_proc.Hprocessor(dataset)
attr_names, dataframes = event_preprocessor.initiate_data_preprocessing(partition_config=partition_config, training_ratio=0.8)
