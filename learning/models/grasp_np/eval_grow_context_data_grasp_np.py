from learning.active.utils import ActiveExperimentLogger

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
import random
import os

# can pull experiment train and val set details from the train args.pkl folder
# select a random sample from train and val
# and see how it does

# they are ordered in lists of 500, and in grasps, there is a maximum of 50 grasps to use

LOG_SUPEDIR = 'learning/experiments/logs'


def grow_data_and_find_latents(geoms, midpoints, forces, labels, mesh, model):
    """
    Iterates through data, and passes data points 0...n through gnp, 0<n<=#data points
    Returns a list of means and covars from each round.
    """
    model.eval()
    model.zero_grad()

    means = []
    covars = []
    for n in range(1, len(geoms) + 1):
        print('evaluating with size ' + str(n))
        n_geoms = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(geoms[:n]),
                1,
                2
            ),
            0
        )
        n_midpoints = torch.unsqueeze(
            torch.tensor(midpoints[:n]),
            0
        )
        n_forces = torch.unsqueeze(
            torch.tensor(forces[:n]),
            0
        )
        n_labels = torch.unsqueeze(
            torch.tensor(labels[:n]),
            0
        )
        n_meshes = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(mesh),  # there is only one mesh, since we are only evaluating one object
                0,
                1
            ),
            0
        )
        q, _ = model.forward_until_latents(
            (n_geoms, n_midpoints, n_forces, n_labels),
            n_meshes
        )
        means.append(q.loc.view(1, -1))
        covars.append(q.scale.view(1, -1))

    return torch.cat(means, dim=0), torch.cat(covars, dim=0)


def choose_one_object_and_grasps(dataset):
    obj_ix = random.randint(0, len(dataset['grasp_data']['labels']))
    object_meshes = dataset['grasp_data']['object_meshes'][obj_ix]
    grasp_geometries = dataset['grasp_data']['grasp_geometries'][obj_ix]
    grasp_midpoints = dataset['grasp_data']['grasp_midpoints'][obj_ix]
    grasp_forces = dataset['grasp_data']['grasp_forces'][obj_ix]
    grasp_labels = dataset['grasp_data']['labels'][obj_ix]
    return obj_ix, grasp_forces, grasp_geometries, grasp_labels, grasp_midpoints, object_meshes


def main(args):
    # set the seed for repeatability in figure making
    random.seed(args.seed)

    # load in the models and datasets for evaluation
    train_log_dir = os.path.join(LOG_SUPEDIR, args.train_logdir)
    train_log_args_fname = os.path.join(train_log_dir, 'args.pkl')
    with open(train_log_args_fname, 'rb') as handle:
        train_args = pickle.load(handle)
    with open(train_args.train_dataset_fname, 'rb') as handle:
        train_set = pickle.load(handle)
    with open(train_args.val_dataset_fname, 'rb') as handle:
        val_set = pickle.load(handle)

    train_logger = ActiveExperimentLogger(
        exp_path=train_log_dir,
        use_latents=False
    )
    model = train_logger.get_neural_process(tx=0)

    # choose a random object in train and then do progressive posterior evaluation
    train_obj_ix, grasp_forces, grasp_geometries, grasp_labels, grasp_midpoints, object_meshes = \
        choose_one_object_and_grasps(train_set)

    # do progressive latent distribution check
    train_means, train_covars = grow_data_and_find_latents(
        grasp_geometries, grasp_midpoints, grasp_forces, grasp_labels,
        object_meshes[0],  # it's the same object, so only one mesh is needed
        model
    )

    # turn train means and train covars into manipulatable arrays and then plot w/ matplotlib
    plot_progressive_means_and_covars(train_covars, train_log_dir, train_means, train_obj_ix, 'train')

    # repeat for validation objects
    val_obj_ix, grasp_forces, grasp_geometries, grasp_labels, grasp_midpoints, object_meshes = \
        choose_one_object_and_grasps(val_set)
    val_means, val_covars = grow_data_and_find_latents(
        grasp_geometries, grasp_midpoints, grasp_forces, grasp_labels,
        object_meshes[0],
        model
    )
    plot_progressive_means_and_covars(val_covars, train_log_dir, val_means, val_obj_ix, 'val')


def plot_progressive_means_and_covars(train_covars, train_log_dir, train_means, train_obj_ix, dset):
    means_arr, covars_arr = train_means.detach().numpy(), train_covars.detach().numpy()
    xs = np.arange(1, means_arr.shape[0] + 1)
    plt.figure()
    for i in range(means_arr.shape[1]):
        mean_col = means_arr[:, i]
        covar_col = covars_arr[:, i]
        plt.plot(xs, mean_col, label='latent %i' % i)
        plt.fill_between(xs, mean_col - covar_col, mean_col + covar_col, alpha=0.2)
    plt.xlabel('size of context set')
    plt.title('latent distribution vs. context set size, set %s obj #%i' % (dset, train_obj_ix))
    plt.legend()
    output_fname = os.path.join(train_log_dir,
                                'figures',
                                dset + '_' + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    plt.savefig(output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-logdir', type=str, required=True, help='training log directory name')
    parser.add_argument('--seed', type=int, default=10, help='seed for random obj selection')
    args = parser.parse_args()
    main(args)
