import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from learning.active.utils import ActiveExperimentLogger

EXPERIMENT_ROOT = 'learning/experiments/metadata'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    # Run fitting phase for all objects that have not yet been fitted
    # (each has a standard name in the experiment logs).
    all_means, all_stds = [], []
    fit_log_paths = logs_lookup['fitting_phase']['random'].values()
    print(f'{len(fit_log_paths)} logs found.')
    for fit_log_path in fit_log_paths:

        fit_logger = ActiveExperimentLogger(fit_log_path, use_latents=True)
        means_path = fit_logger.get_figure_path('val_means.pkl')
        stds_path = fit_logger.get_figure_path('val_covars.pkl')
        with open(means_path, 'rb') as handle:
            means = pickle.load(handle)
        with open(stds_path, 'rb') as handle:
            stds = pickle.load(handle)
        all_means.append(means[-1])
        all_stds.append(stds[-1])

    all_means = np.array(all_means).squeeze()
    all_stds = np.array(all_stds).squeeze()
    import ipdb; ipdb.set_trace()

    print('Means:', np.mean(all_means, axis=0))
    print('Stds:', np.std(all_means, axis=0))
    print('Covars:', np.mean(all_stds, axis=0))
    fig, axes = plt.subplots(all_means.shape[1])
    for dx, ax in enumerate(axes):
        ax.hist(all_means[:, dx], bins=50)
    plt.show()

