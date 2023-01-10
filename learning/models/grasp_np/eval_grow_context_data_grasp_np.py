from learning.active.utils import ActiveExperimentLogger

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import pickle
import torch
import random
import os

from learning.models.grasp_np.dataset import CustomGNPGraspDataset
from learning.models.grasp_np.train_grasp_np import get_loss

# can pull experiment train and val set details from the train args.pkl folder
# select a random sample from train and val
# and see how it does

# they are ordered in lists of 500, and in grasps, there is a maximum of 50 grasps to use

NUM_LATENTS = 5
LOG_SUPEDIR = 'learning/experiments/logs'


def grow_data_and_find_latents(geoms, gpoints, curvatures, midpoints, forces, labels, mesh, model):
    """
    Chooses random order to iterate through data, and passes data points 0...n through gnp, 0<n<=#data points
    Returns a list of means and covars from each round.
    """
    model.eval()
    model.zero_grad()

    max_geoms = torch.swapaxes(torch.tensor(geoms), 2, 3)
    max_gpoints = torch.tensor(gpoints)
    max_curvatures = torch.tensor(curvatures)
    max_midpoints = torch.tensor(midpoints)
    max_forces = torch.tensor(forces)
    max_labels = torch.tensor(labels)
    max_meshes = torch.swapaxes(torch.tensor(mesh), 1, 2)
    _, q_z = model.forward(
        (max_geoms, max_gpoints, max_curvatures, max_midpoints, max_forces, max_labels),
        (max_geoms, max_gpoints, max_curvatures, max_midpoints, max_forces),
        max_meshes
    )

    means = []
    covars = []
    kdls = []
    bces = []
    n_elts = len(geoms)
    order = torch.randperm(n_elts)
    for n in range(1, n_elts + 1):
        print('evaluating with size ' + str(n))
        selected_elts = order[:n].numpy()

        n_geoms = torch.swapaxes(geoms[selected_elts], 2, 3)
        n_gpoints = torch.tensor(gpoints[selected_elts])
        n_curvatures = torch.tensor(curvatures[selected_elts])
        n_midpoints = torch.tensor(midpoints[selected_elts])
        n_forces = torch.tensor(forces[selected_elts])
        n_labels = torch.tensor(labels[selected_elts])
        n_meshes = torch.swapaxes(mesh, 1, 2)
        y_probs, q_n = model.forward(
            (n_geoms, n_gpoints, n_curvatures, n_midpoints, n_forces, n_labels),
            (max_geoms, max_gpoints, max_curvatures, max_midpoints, max_forces),
            n_meshes
        )
        y_probs = y_probs.squeeze()

        means.append(q_n.loc.view(1, -1))
        covars.append(q_n.scale.view(1, -1))

        _, bce_loss, kdl_loss = get_loss(y_probs, max_labels.squeeze(), q_z, q_n)
        bces.append(bce_loss)
        kdls.append(kdl_loss)

    # TODO: there is a way to evaluate models without tracking grads - find it
    return torch.cat(means, dim=0).detach().numpy(), \
        torch.cat(covars, dim=0).detach().numpy(), \
        torch.tensor(bces).detach().numpy(), \
        torch.tensor(kdls).detach().numpy()


def choose_one_object_and_grasps(dataset):
    # we use the loader to get data preprocessing
    loader = CustomGNPGraspDataset(data=dataset)
    obj_ix = random.randint(0, len(loader))
    _, entry = loader[obj_ix]
    # only one object, so only one mesh is needed
    object_meshes = torch.unsqueeze(torch.tensor(entry['object_mesh'][0]), 0)
    grasp_geometries = torch.unsqueeze(torch.tensor(entry['grasp_geometries']), 0)
    grasp_points = torch.unsqueeze(torch.tensor(entry['grasp_points']), 0)
    grasp_curvatures = torch.unsqueeze(torch.tensor(entry['grasp_curvatures']), 0)
    grasp_midpoints = torch.unsqueeze(torch.tensor(entry['grasp_midpoints']), 0)
    grasp_forces = torch.unsqueeze(torch.tensor(entry['grasp_forces']), 0)
    grasp_labels = torch.unsqueeze(torch.tensor(entry['grasp_labels']), 0)
    return obj_ix, grasp_geometries, grasp_points, grasp_curvatures, \
        grasp_midpoints, grasp_forces, grasp_labels, object_meshes


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
    train_obj_ix, grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, \
        grasp_forces, grasp_labels, object_meshes = \
        choose_one_object_and_grasps(train_set)

    num_rounds = args.orders_per_object
    all_rounds_train_means = np.zeros((num_rounds, len(grasp_geometries), NUM_LATENTS))
    all_rounds_train_covars = np.zeros((num_rounds, len(grasp_geometries), NUM_LATENTS))
    all_rounds_train_bces = np.zeros((num_rounds, len(grasp_geometries)))
    all_rounds_train_klds = np.zeros((num_rounds, len(grasp_geometries)))

    # do progressive latent distribution check
    for i in range(num_rounds):
        print('train order #%i' % i)
        train_means, train_covars, train_bces, train_klds = grow_data_and_find_latents(
            grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, grasp_forces, grasp_labels,
            object_meshes,  # it's the same object, so only one mesh is needed
            model
        )
        all_rounds_train_means[i, :, :] = train_means
        all_rounds_train_covars[i, :, :] = train_covars
        all_rounds_train_bces[i, :] = train_bces
        all_rounds_train_klds[i, :] = train_klds

    # TODO: adjust plotting with seaborn
    # turn train means and train covars into manipulatable arrays and then plot w/ matplotlib
    # plot_progressive_means_and_covars(train_means, train_covars, train_bces, train_klds, 'train', train_obj_ix,
    #                                   train_log_dir)

    # repeat for validation objects
    val_obj_ix, grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, \
        grasp_forces, grasp_labels, object_meshes = \
        choose_one_object_and_grasps(val_set)

    # we re-clear all data arrays for easier debugging
    all_rounds_train_means = np.zeros((num_rounds, len(grasp_geometries), NUM_LATENTS))
    all_rounds_train_covars = np.zeros((num_rounds, len(grasp_geometries), NUM_LATENTS))
    all_rounds_train_bces = np.zeros((num_rounds, len(grasp_geometries)))
    all_rounds_train_klds = np.zeros((num_rounds, len(grasp_geometries)))

    for i in range(num_rounds):
        print('val order #%i' % i)
        val_means, val_covars, val_bces, val_klds = grow_data_and_find_latents(
            grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, grasp_forces, grasp_labels,
            object_meshes[0],
            model
        )
        all_rounds_train_means[i, :, :] = val_means
        all_rounds_train_covars[i, :, :] = val_covars
        all_rounds_train_bces[i, :] = val_bces
        all_rounds_train_klds[i, :] = val_klds

    # plot_progressive_means_and_covars(val_means, val_covars, val_bces, val_klds, 'val', val_obj_ix, train_log_dir)


# TODO:
# should we plot each run as a multiple time series? can we get variance data in there? that may be per parameter
def plot_progressive_means_and_covars(means, covars, bces, klds, dset, obj_ix, log_dir):
    means_arr, covars_arr = means.detach().numpy(), covars.detach().numpy()
    xs = np.arange(1, means_arr.shape[0] + 1)
    plt.figure()
    for i in range(means_arr.shape[1]):
        mean_col = means_arr[:, i]
        covar_col = covars_arr[:, i]
        plt.plot(xs, mean_col, label='latent %i' % i)
        plt.fill_between(xs, mean_col - covar_col, mean_col + covar_col, alpha=0.2)
    plt.xlabel('size of context set')
    plt.title('latent distribution vs. context set size, set %s obj #%i' % (dset, obj_ix))
    plt.legend()
    plt.ylim((-10.0, 10.0))
    output_fname = os.path.join(log_dir,
                                'figures',
                                dset + '_prog_dist_' + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    plt.savefig(output_fname)

    plt.figure()
    plt.plot(xs, bces.detach().numpy(), label='bces')
    plt.plot(xs, klds.detach().numpy(), label='klds')
    plt.legend()
    plt.title('ELBO terms vs. context set size, set %s obj #%i' % (dset, obj_ix))
    output_fname = os.path.join(log_dir,
                                'figures',
                                dset + '_bce_kld_' + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    plt.savefig(output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-logdir', type=str, required=True, help='training log directory name')
    parser.add_argument('--seed', type=int, default=10, help='seed for random obj selection')
    parser.add_argument('--orders-per-object', type=int, default=50, help='number of data collection orders per object')
    parser.add_argument('--full-run', action='store_true', default=False,
                        help='run on full training and validation dataset objects')
    args = parser.parse_args()
    main(args)
