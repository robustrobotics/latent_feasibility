from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger

from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
import random
import os

from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn_all_grasps
from learning.models.grasp_np.train_grasp_np import get_loss

# can pull experiment train and val set details from the train args.pkl folder
# select a random sample from train and val
# and see how it does

# they are ordered in lists of 500, and in grasps, there is a maximum of 50 grasps to use

NUM_LATENTS = 5
LOG_SUPEDIR = 'learning/experiments/logs'


def grow_data_and_find_latents(geoms, gpoints, curvatures, midpoints, forces, labels, meshes, model):
    """
    Chooses random order to iterate through data, and passes data points 0...n through gnp, 0<n<=#data points
    Returns a list of means and covars from each round.
    """
    model.eval()
    with torch.no_grad():
        _, q_z = model.forward(
            (geoms, gpoints, curvatures, midpoints, forces, labels),
            (geoms, gpoints, curvatures, midpoints, forces),
            meshes
        )

        means = []
        covars = []
        kdls = []
        bces = []
        n_elts = geoms.shape[1]
        order = torch.randperm(n_elts)
        for n in range(1, n_elts + 1):
            # print('evaluating with size ' + str(n))
            selected_elts = order[:n].numpy()

            n_geoms = geoms[:, selected_elts]
            n_gpoints = gpoints[:, selected_elts]
            n_curvatures = curvatures[:, selected_elts]
            n_midpoints = midpoints[:, selected_elts]
            n_forces = forces[:, selected_elts]
            n_labels = labels[:, selected_elts]
            y_probs, q_n = model.forward(
                (n_geoms, n_gpoints, n_curvatures, n_midpoints, n_forces, n_labels),
                (geoms, gpoints, curvatures, midpoints, forces),
                meshes
            )
            y_probs = y_probs.squeeze()

            means.append(torch.unsqueeze(q_n.loc, 1)) # add dimension so we can do a batched cat
            covars.append(torch.unsqueeze(q_n.scale, 1))

            _, bce_loss, kdl_loss = get_loss(y_probs, labels.squeeze(), q_z, q_n)
            bces.append(bce_loss)
            kdls.append(kdl_loss)

        return torch.cat(means, dim=1).numpy(), \
            torch.cat(covars, dim=1).numpy(), \
            torch.tensor(bces).numpy(), \
            torch.tensor(kdls).numpy()


def choose_one_object_and_grasps(dataset, obj_ix):
    # we use the loader to get data preprocessing
    loader = CustomGNPGraspDataset(data=dataset)
    _, entry = loader[obj_ix]
    # only one object, so only one mesh is needed
    object_meshes = torch.unsqueeze(
        torch.swapaxes(
            torch.tensor(entry['object_mesh'][0]),
            0, 1
        ),
        0
    )
    grasp_geometries = torch.unsqueeze(
        torch.swapaxes(
            torch.tensor(entry['grasp_geometries']),
            1, 2
        ),
        0
    )
    grasp_points = torch.unsqueeze(torch.tensor(entry['grasp_points']), 0)
    grasp_curvatures = torch.unsqueeze(torch.tensor(entry['grasp_curvatures']), 0)
    grasp_midpoints = torch.unsqueeze(torch.tensor(entry['grasp_midpoints']), 0)
    grasp_forces = torch.unsqueeze(torch.tensor(entry['grasp_forces']), 0)
    grasp_labels = torch.unsqueeze(torch.tensor(entry['grasp_labels']), 0)
    return grasp_geometries, grasp_points, grasp_curvatures, \
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

    num_rounds = args.orders_per_object
    # iterate through individual objects we must plot and then plot each chart
    for obj_ix in args.plot_train_objs:
        grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, \
            grasp_forces, grasp_labels, object_meshes = \
            choose_one_object_and_grasps(train_set, obj_ix=obj_ix)

        num_grasps = grasp_geometries.shape[1]
        all_rounds_train_means = np.zeros((num_rounds, num_grasps, NUM_LATENTS))
        all_rounds_train_covars = np.zeros((num_rounds, num_grasps, NUM_LATENTS))
        all_rounds_train_bces = np.zeros((num_rounds, num_grasps))
        all_rounds_train_klds = np.zeros((num_rounds, num_grasps))

        # do progressive latent distribution check
        for i in range(num_rounds):
            # print('train order #%i' % i)
            train_means, train_covars, train_bces, train_klds = grow_data_and_find_latents(
                grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, grasp_forces, grasp_labels,
                object_meshes,  # it's the same object, so only one mesh is needed
                model
            )
            all_rounds_train_means[i, :, :] = train_means[0] # unsqueeze since we get everything batched
            all_rounds_train_covars[i, :, :] = train_covars[0]
            all_rounds_train_bces[i, :] = train_bces
            all_rounds_train_klds[i, :] = train_klds

        # turn train means and train covars into manipulatable arrays and then plot w/ matplotlib
        plot_progressive_means_and_covars(all_rounds_train_means,
                                          all_rounds_train_covars,
                                          all_rounds_train_bces,
                                          all_rounds_train_klds, 'train', obj_ix, train_log_dir)

        # repeat for validation objects
    for obj_ix in args.plot_val_objs:
        grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, \
            grasp_forces, grasp_labels, object_meshes = \
            choose_one_object_and_grasps(val_set, obj_ix=obj_ix)

        # we re-clear all data arrays for easier debugging
        num_grasps = grasp_geometries.shape[1]
        all_rounds_val_means = np.zeros((num_rounds, num_grasps, NUM_LATENTS))
        all_rounds_val_covars = np.zeros((num_rounds, num_grasps, NUM_LATENTS))
        all_rounds_val_bces = np.zeros((num_rounds, num_grasps))
        all_rounds_val_klds = np.zeros((num_rounds, num_grasps))

        for i in range(num_rounds):
            # print('val order #%i' % i)
            val_means, val_covars, val_bces, val_klds = grow_data_and_find_latents(
                grasp_geometries, grasp_points, grasp_curvatures, grasp_midpoints, grasp_forces, grasp_labels,
                object_meshes,
                model
            )
            all_rounds_val_means[i, :, :] = val_means[0] # unsqueeze since we get everything batched
            all_rounds_val_covars[i, :, :] = val_covars[0]
            all_rounds_val_bces[i, :] = val_bces
            all_rounds_val_klds[i, :] = val_klds

        plot_progressive_means_and_covars(all_rounds_val_means,
                                          all_rounds_val_covars,
                                          all_rounds_val_bces,
                                          all_rounds_val_klds, 'val', obj_ix, train_log_dir)

    # perform training and validation-set wide performance evaluation
    if args.full_run:
        # construct dataloader
        train_dataset = CustomGNPGraspDataset(data=train_set)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=64,
            collate_fn=custom_collate_fn_all_grasps
        )

        for i, (context_data, target_data, meshes) in enumerate(train_dataloader):
            grasp_geoms, grasp_points, curvatures, midpoints, forces, labels = context_data

            train_means, train_covars, train_bces, train_klds = grow_data_and_find_latents(
                grasp_geoms, grasp_points, curvatures, midpoints, forces, labels,
                meshes,
                model
            )


def plot_progressive_means_and_covars(means, covars, bces, klds, dset, obj_ix, log_dir):
    # plot latent sequence as time evolves
    xs = np.arange(1, means.shape[1] + 1)
    plt.figure(figsize=(20, 4))
    for i_latent_var in range(means.shape[2]):
        ax = plt.subplot(1, 5, i_latent_var + 1)
        ax.set_title('latent %i, set %s, obj #%i' % (i_latent_var, dset, obj_ix))
        for i_round_num in range(means.shape[0]):
            mean_col = means[i_round_num, :, i_latent_var]
            covar_col = covars[i_round_num, :, i_latent_var]
            ax.plot(xs, mean_col, label='round %i' % i_round_num)
            ax.fill_between(xs, mean_col - covar_col, mean_col + covar_col, alpha=0.2)
            ax.set_xlabel('size of context set')
    output_fname = os.path.join(log_dir,
                                'figures',
                                dset + '_obj' + str(obj_ix) + '_prog_dist_'
                                + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    plt.savefig(output_fname)

    # plot average bce and kld curves
    # construct dataframe for seaborn plotting
    plt.figure()
    d = {'acquisition':
        np.concatenate([
            np.hstack(
                [np.arange(bces.shape[1]).reshape(-1, 1)] * bces.shape[0]
            ).flatten(),
            np.hstack(
                [np.arange(klds.shape[1]).reshape(-1, 1)] * klds.shape[0]
            ).flatten()
        ]),
        'loss_component': ['bce'] * bces.size + ['kld'] * klds.size,
        'round': list(range(bces.shape[0])) * (bces.shape[1] + klds.shape[1]),
        'value': np.concatenate([bces.flatten(order='C'), klds.flatten(order='C')])}
    loss_data = pd.DataFrame(data=d)
    sns.relplot(x='acquisition', y='value', col='loss_component',
                kind='line', data=loss_data, facet_kws=dict(sharey=False))
    output_fname = os.path.join(log_dir,
                                'figures',
                                dset + '_obj' + str(obj_ix) + '_bce_kld_'
                                + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    plt.savefig(output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-logdir', type=str, required=True, help='training log directory name')
    parser.add_argument('--orders-per-object', type=int, default=50, help='number of data collection orders per object')
    parser.add_argument('--plot-train-objs', type=int, nargs='*', default=[],
                        help='specific training objects to plots, indexed by object ids in train_grasps.pkl')
    parser.add_argument('--plot-val-objs', type=int, nargs='*', default=[],
                        help='specific objects objects to plot, as specified by object ids val_grasps.pkl')
    parser.add_argument('--full-run', action='store_true', default=False,
                        help='run on full training and validation dataset objects')
    parser.add_argument('--seed', type=int, default=10, help='seed for random obj selection')
    args = parser.parse_args()
    main(args)
