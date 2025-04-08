import pandas as pd
import numpy as np

from typing import List, Dict, Iterable

# we'll be obtaining lots of data that looks like this:

# NPT data

# time X [performance metrics]

# particle filter
# time X [performance metrics] X [number of particles]


# so we build individual tables of time vs. performance metrics, 
# of some specific experiment configuration, and then we  
# stick them together for simultaneous plotting.

# we aim to store the data in _long_ dataframes, so that it can be
# easily plotted using seaborn.



def single_experiment_run_to_df(single_run_dict: Dict[str, Iterable]) \
    -> pd.DataFrame:
    return pd.DataFrame(data=single_run_dict)

def batch_experiment_runs_into_trials_df(exp_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    # add a trial column to each of the exp_dfs, and then
    # concatenate all of them into one big array
    for _exp_idx, _exp_df in enumerate(exp_dfs):
        _exp_df['trial'] = _exp_idx

    return pd.concat(exp_dfs)
    

def batch_multiple_trial_dfs_into_comparison_df(trial_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Batches together trial dataframes to compare different methods in seaborn.

    Args:
        trial_dfs (Dict[str, pd.DataFrame]): a mapping from a name for a certain method
        (ex: PushingNeuralProcess, Particle Filter n=100) to the associated trial dataframe.

    Returns:
        pd.DataFrame: A full method that contains data from multiple methods for comparison in seaborn.
    """
    for _name, _trial_df in trial_dfs.items():
        _trial_df['method'] = _name
    
    return pd.concat(trial_dfs.values())





