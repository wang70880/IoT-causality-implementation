import sys
sys.path.append('../src/')
from tigramite import data_processing as pp

class SyntheticGenerator():
    """The synthetic data generator.
    The use of synthetic data is for showing the following features.
    1. The performance comparison for discovering device correlations (causal discovery v.s. association rule mining)
        * The identifiability of confounders
        * The identifiability of indirect causes
    2. The performance evaluation for causal discovery in time series data
        * The necessity for partitioning data
        * stablePC algorithm v.s. PCMCI
    """
    def __init__(self) -> None:
        pass

if __name__ == '__main__':
    # 1. Parse input parameters
    args = sys.argv
    n_samples = int(args[1])
    config = str(args[3])

    all_configs = {config: {'results':{}, 
        "graphs":{}, 
        "val_min":{}, 
        "max_cardinality":{}, 

        "true_graph":{}, 
        "computation_time":{},}}
    print("Test")