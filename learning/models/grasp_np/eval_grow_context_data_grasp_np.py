from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger

from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score
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

DATE_FORMAT = '%m%d%Y_%H%M%S'
NUM_LATENTS = 5
LOG_SUPEDIR = 'learning/experiments/logs'


# TODO: (1) Add in validation set to full eval (test!)


def grow_data_and_find_latents(context_points, target_points, meshes, model):
    """
    Chooses random order to iterate through data, and passes data points 0...n through gnp, 0<n<=#data points
    Returns a list of means and covars from each round.
    """
    c_geoms, c_gpoints, c_curvatures, c_midpoints, c_forces, c_labels = context_points
    t_geoms, t_gpoints, t_curvatures, t_midpoints, t_forces, t_labels = target_points
    model.eval()
    with torch.no_grad():
        _, q_z = model.forward(
            (c_geoms, c_gpoints, c_curvatures, c_midpoints, c_forces, c_labels),
            (c_geoms, c_gpoints, c_curvatures, c_midpoints, c_forces),
            meshes
        )

        means = []
        covars = []
        kdls = []
        bces = []
        pr_scores = []
        n_elts = c_geoms.shape[1]
        order = torch.randperm(n_elts)
        for n in range(1, n_elts + 1):
            # print('evaluating with size ' + str(n))
            selected_elts = order[:n].numpy()

            n_geoms = c_geoms[:, selected_elts]
            n_gpoints = c_gpoints[:, selected_elts]
            n_curvatures = c_curvatures[:, selected_elts]
            n_midpoints = c_midpoints[:, selected_elts]
            n_forces = c_forces[:, selected_elts]
            n_labels = c_labels[:, selected_elts]
            y_probs, q_n = model.forward(
                (n_geoms, n_gpoints, n_curvatures, n_midpoints, n_forces, n_labels),
                (t_geoms, t_gpoints, t_curvatures, t_midpoints, t_forces),
                meshes
            )
            y_probs = y_probs.squeeze()

            means.append(torch.unsqueeze(q_n.loc, 1))  # add dimension so we can do a batched cat
            covars.append(torch.unsqueeze(q_n.scale, 1))

            _, bce_loss, kdl_loss = get_loss(y_probs, t_labels.squeeze(), q_z, q_n)
            pr_score = average_precision_score(t_labels.squeeze(), y_probs)
            bces.append(bce_loss)
            kdls.append(kdl_loss)
            pr_scores.append(pr_score)

        return torch.cat(means, dim=1).numpy(), \
            torch.cat(covars, dim=1).numpy(), \
            torch.tensor(bces).numpy(), \
            torch.tensor(kdls).numpy(), \
            np.array(pr_scores)


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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    n_rounds = args.orders_per_object
    for obj_ix in args.plot_objs:
        train_geoms, train_gpoints, train_curvatures, train_midpoints, train_forces, train_labels, object_meshes = \
            choose_one_object_and_grasps(train_set, obj_ix=obj_ix)

        val_geoms, val_gpoints, val_curvatures, val_midpoints, val_forces, val_labels, _ = \
            choose_one_object_and_grasps(val_set, obj_ix=obj_ix)

        context_points = (train_geoms, train_gpoints, train_curvatures,
                          train_midpoints, train_forces, train_labels)
        target_points = (val_geoms, val_gpoints, val_curvatures,
                         val_midpoints, val_forces, val_labels)

        # we re-clear all data arrays for easier debugging
        n_acquisitions = train_geoms.shape[1]
        all_rounds_train_means = np.zeros((n_rounds, n_acquisitions, NUM_LATENTS))
        all_rounds_train_covars = np.zeros((n_rounds, n_acquisitions, NUM_LATENTS))
        all_rounds_bces = np.zeros((2, n_rounds, n_acquisitions))  # since these can get plot against eachother,
        all_rounds_klds = np.zeros((2, n_rounds, n_acquisitions))  # we have 0 for train and 1 for val
        all_rounds_prs = np.zeros((2, n_rounds, n_acquisitions))

        for i in range(n_rounds):
            # print('val order #%i' % i)
            train_means, train_covars, train_bces, train_klds, train_prs = grow_data_and_find_latents(
                context_points=context_points,
                target_points=context_points,
                meshes=object_meshes,  # it's the same object, so only one mesh is needed
                model=model
            )
            all_rounds_train_means[i, :, :] = train_means[0]  # unsqueeze since we get everything batched
            all_rounds_train_covars[i, :, :] = train_covars[0]
            all_rounds_bces[0, i, :] = train_bces
            all_rounds_klds[0, i, :] = train_klds
            all_rounds_prs[0, i, :] = train_prs

            _, _, val_bces, val_klds, val_prs = grow_data_and_find_latents(
                context_points=context_points,
                target_points=target_points,
                meshes=object_meshes,
                model=model
            )
            all_rounds_bces[1, i, :] = val_bces
            all_rounds_klds[1, i, :] = val_klds
            all_rounds_prs[1, i, :] = val_prs

        # mean and covar progression plots
        plot_progressive_means_and_covars(all_rounds_train_means,
                                          all_rounds_train_covars,
                                          'train_val', obj_ix, train_log_dir)

        # set up dataframes for plotting the remainder
        mi = pd.MultiIndex.from_product([['train', 'validaton'],
                                         ['bce', 'kld'],
                                         range(n_rounds),
                                         range(n_acquisitions)],
                                        names=['phase', 'loss_component', 'round', 'acquisition'])
        loss_data = pd.DataFrame(data=np.concatenate([all_rounds_bces.flatten(), all_rounds_klds.flatten()]),
                                 index=mi, columns=['value'])
        mi = pd.MultiIndex.from_product([['train', 'validation'],
                                         range(n_rounds),
                                         range(n_acquisitions)],
                                        names=['phase', 'round', 'acquisition'])
        pr_data = pd.DataFrame(data=all_rounds_prs.flatten(), index=mi, columns=['value'])

        plot_bces_and_klds(loss_data, 'train_val', train_log_dir, obj_ix=obj_ix)
        plot_prs(pr_data, 'train_val', train_log_dir, obj_ix=obj_ix)

    # perform training and validation-set wide performance evaluation
    if args.full_run:
        # construct dataloader
        train_dataset = CustomGNPGraspDataset(data=train_set)
        val_dataset = CustomGNPGraspDataset(data=val_set, context_data=train_set)

        batch_size = 64

        val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=custom_collate_fn_all_grasps,
            batch_size=batch_size,
            shuffle=False,
        )

        output_fname_template = os.path.join(train_log_dir,
                                             'datasets',
                                             'train_%s.pkl')

        loss_fname = output_fname_template % 'loss'
        pr_fname = output_fname_template % 'precision_recall'
        if os.path.exists(pr_fname) and os.path.exists(loss_fname):
            choice = input('[Warning] Processed data from %s exists. Redo progression computation? (yes/no)')
        else:
            choice = 'yes'

        if choice == 'yes':
            # we're saving all the data that comes out of the computation since it's so expensive
            n_batches = len(val_dataloader)
            n_acquisitions = len(train_set['grasp_data']['grasp_points'][0])

            # the first dimension represents 'training' for 0 and 'val' for 1
            all_rounds_means = np.zeros((2, n_batches, n_rounds, n_acquisitions, NUM_LATENTS))
            all_rounds_covars = np.zeros((2, n_batches, n_rounds, n_acquisitions, NUM_LATENTS))
            all_rounds_bces = np.zeros((2, n_batches, n_rounds, n_acquisitions))
            all_rounds_klds = np.zeros((2, n_batches, n_rounds, n_acquisitions))
            all_rounds_pr_scores = np.zeros((2, n_batches, n_rounds, n_acquisitions))

            # since we max out on number of grasps given in the context_set, and since we do not shuffle
            # the data, the context_data found below is the same as the context_data and the target_data
            # from the train set. So we can get away with evaluation without iterating through
            # because that information is redundant
            for i_batch, (context_data, target_data, meshes) in enumerate(val_dataloader):
                print('batch: ' + str(i_batch))

                for i_round in range(n_rounds):
                    print('round: ' + str(i_round))
                    all_rounds_means[0, i_batch, i_round, :, :], \
                        all_rounds_covars[0, i_batch, i_round, :, :], \
                        all_rounds_bces[0, i_batch, i_round, :], \
                        all_rounds_klds[0, i_batch, i_round, :], \
                        all_rounds_pr_scores[0, i_batch, i_round, :] = \
                        grow_data_and_find_latents(context_data, context_data, meshes, model)

                    all_rounds_means[1, i_batch, i_round, :, :], \
                        all_rounds_covars[1, i_batch, i_round, :, :], \
                        all_rounds_bces[1, i_batch, i_round, :], \
                        all_rounds_klds[1, i_batch, i_round, :], \
                        all_rounds_pr_scores[1, i_batch, i_round, :] = \
                        grow_data_and_find_latents(context_data, target_data, meshes, model)

            # construct dataframes and save them
            mi = pd.MultiIndex.from_product([['train', 'validation'],
                                             ['mean', 'covar'],
                                             range(n_batches),
                                             range(n_rounds),
                                             range(n_acquisitions),
                                             range(NUM_LATENTS)],
                                            names=['phase', 'normal_param', 'batch', 'round', 'acquisition', 'latent'])
            normal_data = pd.DataFrame(data=np.concatenate([all_rounds_means.flatten(), all_rounds_covars.flatten()]),
                                       index=mi, columns=['value'])
            normal_data.to_pickle(output_fname_template % 'normal')

            mi = pd.MultiIndex.from_product([['train', 'validaton'],
                                             ['bce', 'kld'],
                                             range(n_batches),
                                             range(n_rounds),
                                             range(n_acquisitions)],
                                            names=['phase', 'loss_component', 'batch', 'round', 'acquisition'])
            loss_data = pd.DataFrame(data=np.concatenate([all_rounds_bces.flatten(), all_rounds_klds.flatten()]),
                                     index=mi, columns=['value'])
            loss_data.to_pickle(loss_fname)

            mi = pd.MultiIndex.from_product([['train', 'validation'],
                                             range(n_batches),
                                             range(n_rounds),
                                             range(n_acquisitions)],
                                            names=['phase', 'batch', 'round', 'acquisition'])
            pr_data = pd.DataFrame(data=all_rounds_pr_scores.flatten(), index=mi, columns=['value'])
            pr_data.to_pickle(pr_fname)
        else:
            loss_data = pd.read_pickle(loss_fname)
            pr_data = pd.read_pickle(pr_fname)

        plot_bces_and_klds(loss_data, 'full_set', train_log_dir)
        plot_prs(pr_data, 'full_set', train_log_dir)


def plot_progressive_means_and_covars(means, covars, dset, obj_ix, log_dir):
    # plot latent sequence as time evolves
    # this is done in matplotlib since seaborn gets in the way of the covar-bar setting
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


def plot_bces_and_klds(loss_data, dset, log_dir, obj_ix=None):
    # plot average bce and kld curves
    # construct dataframe for seaborn plotting
    plt.figure()
    plt.title('loss components, dset %s, object#%i' % (dset, obj_ix))
    fg = sns.relplot(x='acquisition', y='value', col='loss_component', hue='phase',
                     kind='line', data=loss_data, facet_kws=dict(sharey=False))
    if obj_ix is not None:
        fg.set_titles(col_template="{col_name}, set %s, object %i" % (dset, obj_ix))
        output_fname = os.path.join(log_dir,
                                    'figures',
                                    dset + '_obj' + str(obj_ix) + '_bce_kld_'
                                    + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    else:
        fg.set_titles(col_template="{col_name}, set %s" % dset)
        output_fname = os.path.join(log_dir,
                                    'figures',
                                    dset + '_full_set_bce_kld_'
                                    + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
    plt.savefig(output_fname)


def plot_prs(pr_data, dset, log_dir, obj_ix=None):
    plt.figure()
    sns.lineplot(x='acquisition', y='value', hue='phase', data=pr_data)

    if obj_ix is not None:
        output_fname = os.path.join(log_dir,
                                    'figures',
                                    dset + '_obj' + str(obj_ix) + '_pr_'
                                    + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
        plt.title('set %s, obj %i, performance, average precision' % (dset, obj_ix))
    else:
        output_fname = os.path.join(log_dir,
                                    'figures',
                                    dset + '_pr_'
                                    + datetime.now().strftime('%m%d%Y_%H%M%S') + '.png')
        plt.title(dset + ' performance, average precision')

    plt.savefig(output_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-logdir', type=str, required=True, help='training log directory name')
    parser.add_argument('--orders-per-object', type=int, default=50, help='number of data collection orders per object')
    parser.add_argument('--plot-objs', type=int, nargs='*', default=[],
                        help='specific objects on which to run analysis, '
                             'indexed by object ids train_grasps.pkl/val_grasps.pkl')
    parser.add_argument('--full-run', action='store_true', default=False,
                        help='run on full training and validation dataset objects')
    parser.add_argument('--seed', type=int, default=10, help='seed for random obj selection')
    args = parser.parse_args()
    main(args)
