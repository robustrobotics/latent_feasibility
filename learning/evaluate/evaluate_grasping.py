import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import matplotlib.cm as cm
import pb_robot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, confusion_matrix, \
    accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from learning.domains.grasping.active_utils import get_fit_object, sample_unlabeled_data, get_labels, \
    get_train_and_fit_objects, drop_last_grasp_in_dataset, explode_dataset_into_list_of_datasets, select_gnp_dataset_firstk
from learning.active.acquire import bald

from block_utils import ParticleDistribution
from filter_utils import sample_particle_distribution
from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.grasp_data import GraspDataset, GraspParallelDataLoader
from learning.domains.grasping.explore_dataset import visualize_grasp_dataset, visualize_fitting_acquisition
from learning.experiments.active_fit_grasping_pf import compute_ig, particle_bald
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.train_grasp_np import check_to_cuda
from particle_belief import GraspingDiscreteLikelihoodParticleBelief, AmortizedGraspingDiscreteLikelihoodParticleBelief


def get_labels_predictions(logger, val_dataset_fname):
    # Load dataset
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    val_dataset = GraspDataset(val_data, grasp_encoding='per_point')
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = logger.get_ensemble(0)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Get predictions
    n_batches = 0
    predictions, labels = [], []
    for x, object_ids, y in val_dataloader:
        if torch.cuda.is_available():
            x = x.float().cuda()
        with torch.no_grad():
            probs = model.forward(x[:, :-5, :], object_ids).squeeze().cpu()

        preds = (probs > 0.5).float()

        wrong = (preds != y)
        print('Wrong Probabilities:', probs[wrong])
        predictions.append(preds)
        labels.append(y)

        n_batches += 1
        print(n_batches)
        if n_batches > 50:
            break

    predictions = torch.cat(predictions).numpy()
    labels = torch.cat(labels).numpy()

    return labels, predictions


def get_validation_metrics(logger, val_dataset_fname):
    labels, predictions = get_labels_predictions(logger, val_dataset_fname)

    # Calculate metrics
    metrics_fn = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score
    }

    metrics_val = {}
    for name, fn in metrics_fn.items():
        metrics_val[name] = fn(labels, predictions)

    with open(logger.get_figure_path('metrics.json'), 'w') as handle:
        json.dump(metrics_val, handle)
    print(metrics_val)


def visualize_predictions(logger, val_dataset_fname):
    labels, predictions = get_labels_predictions(logger, val_dataset_fname)

    figure_path = logger.get_figure_path('')
    # visualize_grasp_dataset(val_dataset_fname, labels=predictions==labels)
    visualize_grasp_dataset(val_dataset_fname, labels=predictions == labels, figpath=figure_path, prefix='correct_')
    visualize_grasp_dataset(val_dataset_fname, labels=labels, figpath=figure_path, prefix='labels_')
    visualize_grasp_dataset(val_dataset_fname, labels=predictions, figpath=figure_path, prefix='predictions_')


def combine_image_grids(logger, prefixes):
    for ix in range(0, 50):
        for angle in ['x', 'y', 'z']:
            images = []
            for p in prefixes:
                fname = logger.get_figure_path('%s_%d_%s.png' % (p, ix, angle))
                images.append(plt.imread(fname))

            fig = plt.figure(figsize=(5, 15))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 3))

            for ax, im in zip(grid, images):
                print(im.shape)
                im = im[:, 500:, :]
                im = im[:, :-500, :]
                ax.imshow(im)

            for ax, title in zip(grid, prefixes):
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.set_title(title)

            plt.savefig(logger.get_figure_path('combined_%d_%s.png' % (ix, angle)), bbox_inches='tight', dpi=500)


def truncate_grasps(tensor_list, n_grasps):
    return [tensor[:, :n_grasps, ...] for tensor in tensor_list]


def get_gnp_contextualized_gnp_predictions(gnp, context_data, val_data):
    preds, labels, means, covars, entropy = [], [], [], [], []
    dataset = CustomGNPGraspDataset(
        data=val_data,
        context_data=context_data,
        add_mesh_curvatures=False,
        add_mesh_normals=False
    )
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate_fn,
        batch_size=16,
        shuffle=False
    )
    gnp.eval()
    for (context_data, target_data, object_data) in dataloader:
        t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces, t_labels = \
            check_to_cuda(truncate_grasps(target_data, 100))
        c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels = check_to_cuda(
            context_data)
        meshes, object_properties = check_to_cuda(object_data)

        c_sizes = torch.ones_like(c_forces) * c_forces.shape[1] / 50
        t_labels = t_labels.squeeze(dim=-1)

        pred, q_z, inf_time = gnp.forward(
            (c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels, c_sizes),
            (t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces),
            (meshes, object_properties),
            return_inf_time=True,
            use_mean=True
        )
        pred = pred.squeeze(dim=-1).cpu().detach()

        preds.append(pred)
        labels.append(t_labels)
        means.append(q_z.loc.detach().cpu())
        covars.append(q_z.scale.detach().cpu())
        entropy.append(torch.distributions.Independent(q_z, 1).entropy().detach().cpu())

    return torch.cat(preds, dim=0).cpu(), torch.cat(labels, dim=0).cpu(), \
           torch.cat(means, dim=0).cpu(), torch.cat(covars, dim=0).cpu(), \
           torch.cat(entropy, dim=0).cpu(), inf_time


def get_predictions_with_particles(particles, grasp_data, ensemble, n_particle_samples=10):
    preds, labels = [], []
    dataset = GraspDataset(data=grasp_data, grasp_encoding='per_point')
    dataloader = GraspParallelDataLoader(dataset=dataset,
                                         batch_size=16,
                                         shuffle=False,
                                         n_dataloaders=1)

    latent_samples = torch.Tensor(particles)
    ensemble.eval()
    for set_of_batches in dataloader:
        grasps, object_ids, y = set_of_batches[0]

        if torch.cuda.is_available():
            grasps = grasps.cuda()
            object_ids = object_ids.cuda()

        with torch.no_grad():
            # Sample particles and ensembles models to use to speed up evaluation. Might hurt performance.
            ensemble_ix = np.random.choice(np.arange(ensemble.ensemble.n_models))
            latents_ix = np.arange(latent_samples.shape[0])
            np.random.shuffle(latents_ix)
            latents_ix = latents_ix[:n_particle_samples]

            # latents = latent_samples[ix*100:(ix+1)*100,:]
            latents = latent_samples[latents_ix, :]
            if torch.cuda.is_available():
                latents = latents.cuda()
            pred = ensemble.forward(X=grasps[:, :-5, :],
                                    object_ids=object_ids,
                                    N_samples=n_particle_samples,
                                    ensemble_idx=-1,  # ensemble_ix,
                                    collapse_latents=True,
                                    collapse_ensemble=True,
                                    pf_latent_ix=100,
                                    latent_samples=latents).squeeze()

            preds.append(pred.mean(dim=-1))
            labels.append(y)
            # if (len(preds)*16) > 200: break
    return torch.cat(preds, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


def get_gnp_predictions_with_particles(particles, grasp_data, gnp, n_particle_samples=10):
    preds, labels = [], []
    dataset = CustomGNPGraspDataset(
        data=grasp_data,
        context_data=grasp_data
    )
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=lambda items: custom_collate_fn(items, rotate=False),
        batch_size=1,
        shuffle=False
    )

    latent_samples = torch.Tensor(particles)
    gnp.eval()
    for (_, target_data, object_data) in dataloader:
        t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces, t_labels = \
            check_to_cuda(truncate_grasps(target_data, 100))
        meshes, object_properties = check_to_cuda(object_data)

        t_grasp_geoms = t_grasp_geoms.expand(n_particle_samples, -1, -1, -1)
        t_grasp_points = t_grasp_points.expand(n_particle_samples, -1, -1, -1)
        t_curvatures = t_curvatures.expand(n_particle_samples, -1, -1, -1)
        t_normals = t_normals.expand(n_particle_samples, -1, -1, -1)
        t_midpoints = t_midpoints.expand(n_particle_samples, -1, -1)
        t_forces = t_forces.expand(n_particle_samples, -1)

        # Sample particles and ensembles models to use to speed up evaluation. Might hurt performance.
        latents_ix = np.arange(latent_samples.shape[0])
        np.random.shuffle(latents_ix)
        latents_ix = latents_ix[:n_particle_samples]
        # import IPython; IPython.embed()
        latents = latent_samples[latents_ix, :]
        if torch.cuda.is_available():
            latents = latents.cuda()
        pred = gnp.conditional_forward(
            target_xs=(t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces),
            meshes=meshes,
            zs=latents
        ).squeeze().cpu().detach()

        preds.append(pred.mean(dim=0))
        labels.append(t_labels)

    return torch.cat(preds, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


def get_pf_task_performance(logger, fname, use_progressive_priors, task='min-force'):
    """ 
    Perform minimum force evaluation for a specific object.
    :param logger: Logger object for the fitted object.
    :param fname: Test dataset for the object of interest. Contains pool of samples.
    Note this only works with GNP models.
    """

    with open(fname, 'rb') as handle:
        val_grasp_data = pickle.load(handle)

    grasp_forces = val_grasp_data['grasp_data']['grasp_forces']
    grasp_forces = list(grasp_forces.values())[0]

    record = []
    eval_range = range(0, logger.args.max_acquisitions, 1)
    for tx in eval_range:
        print('Eval timestep, ', tx)

        particles = logger.load_particles(tx)

        # This is necessary if we're using a weighted particle filter.
        if not use_progressive_priors:
            sampling_dist = ParticleDistribution(
                particles.particles,
                particles.weights / np.sum(particles.weights)
            )
            resampled_parts = sample_particle_distribution(sampling_dist, num_samples=50)
            gnp = logger.get_neural_process(tx)
            if torch.cuda.is_available():
                gnp = gnp.cuda()
            probs, labels = get_gnp_predictions_with_particles(
                resampled_parts,
                val_grasp_data,
                gnp,
                n_particle_samples=32
            )
            probs = probs.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()
        else:
            context_data, sampled_unlabeled_data = logger.load_acquisition_data(tx)
            gnp = logger.get_neural_process(tx)
            if torch.cuda.is_available():
                gnp = gnp.cuda()
            # Write function to get predictions given a set of context data.
            probs, labels, _, _, _ = get_gnp_contextualized_gnp_predictions(
                gnp,
                context_data,
                val_grasp_data
            )
            probs = probs.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

        if task == 'min-grasp':
            max_force = 20  # NOTE: for the max likelihood grasp detector, we'd replace from here:
            neg_reward = 0  # -max_force        # We just choose the grasp with the highest likelihood and report success.

            max_reward = neg_reward
            max_exp_reward = neg_reward
            max_achieved_reward = neg_reward
            max_stable_force, min_stable_force = 0, 25

            print(f'# Stable: {np.sum(labels)}/{len(labels)}')
            for force, prob, label in zip(grasp_forces, probs, labels):
                pos_reward = (max_force - force)
                exp_reward = neg_reward * (1 - prob) + prob * pos_reward
                true_reward = pos_reward if label else neg_reward

                if (true_reward > max_reward) and (label == 1):
                    max_reward = true_reward

                if exp_reward > max_exp_reward:
                    max_exp_reward = exp_reward
                    max_achieved_reward = true_reward

                if label == 1:
                    if force > max_stable_force:
                        max_stable_force = force
                    if force < min_stable_force:
                        min_stable_force = force

                # if label == 1:
                #     print(f'F: {force}\tProb: {prob}\tExpR: {exp_reward}')

            if max_achieved_reward == neg_reward:
                regret = 1
            else:
                regret = (
                                     max_reward - max_achieved_reward) / 15  # TODO: magic number alert! is this just a normalization?
            record.append(regret)  # to here. (from Note above).

            valid_force_range = max_stable_force - min_stable_force
            print(
                f'Max Reward: {max_reward}\tReward: {max_achieved_reward}\tRegret: {regret}\tRange: {valid_force_range}')

            with open(logger.get_figure_path(f'regrets_{neg_reward}.pkl'), 'wb') as handle:
                pickle.dump(record, handle)

        elif task == 'likely-grasp':

            if np.any(labels > 0.05): # filter out degenerate cases where there are no valid grasps
                most_likely_ind = np.argmax(probs)
                ml_likelihood = probs[most_likely_ind]
                ml_label = labels[most_likely_ind]
                record.append(ml_label)

                print(
                    f'Chose grasp: {most_likely_ind}\tLikelihood: {ml_likelihood}\tStable: {ml_label}'
                )
            else:
                record.append(np.NaN)

            with open(logger.get_figure_path('success.pkl'), 'rb') as handle:
                pickle.dump(record, handle)


        else:
            raise NotImplementedError('Unrecognized task. Options: likely-grasp, min-force.')


def get_pf_validation_accuracy(logger, fname, amortize, use_progressive_priors, vis=False):
    accs, precisions, recalls, f1s, balanced_accs, av_precs = [], [], [], [], [], []
    thresholded_recalls = {}
    confusions = []

    means_agg, covars_agg, entropies, info_gains = [], [], [], []

    with open(fname, 'rb') as handle:
        val_grasp_data = pickle.load(handle)

    eval_range = range(0, logger.args.max_acquisitions, 1)
    times = []
    for tx in eval_range:
        print('Eval timestep, ', tx)

        # This is necessary if we're using a weighted particle filter.
        if not use_progressive_priors:
            particles = logger.load_particles(tx)
            sampling_dist = ParticleDistribution(
                particles.particles,
                particles.weights / np.sum(particles.weights)
            )
            resampled_parts = sample_particle_distribution(sampling_dist, num_samples=50)
            if amortize:
                gnp = logger.get_neural_process(tx)
                if torch.cuda.is_available():
                    gnp = gnp.cuda()
                probs, labels = get_gnp_predictions_with_particles(
                    resampled_parts,
                    val_grasp_data,
                    gnp,
                    n_particle_samples=32
                )

                if vis:
                    # recover datasets used for item selection
                    context_data, sampled_unlabeled_data = logger.load_acquisition_data(tx)
                    pre_selection_context_data = drop_last_grasp_in_dataset(context_data)

                    # compute predictions based on particles
                    pf = AmortizedGraspingDiscreteLikelihoodParticleBelief(
                        object_set=sampled_unlabeled_data['object_data'],  # NOTE: this is a dummy we do not use this
                        d_latents=gnp.d_latents,
                        n_particles=len(particles.particles),  # NOTE: for dummy
                        likelihood=gnp,  # this is the only parameter that matters
                        resample=False,
                        plot=False,
                        data_is_in_gnp_format=not ('grasp' in context_data['grasp_data'].keys())
                    )
                    all_grasps = explode_dataset_into_list_of_datasets(sampled_unlabeled_data)
                    all_preds = []
                    for grasp in all_grasps:
                        all_preds.append(pf.get_particle_likelihoods(particles.particles, grasp))

                    # compute information gain via bald objective
                    pred_vec = torch.Tensor(np.stack(all_preds))
                    info_gain = particle_bald(pred_vec, pf.particles.weights)

                    # visualize
                    max_entropy = torch.distributions.Independent(
                        torch.distributions.Normal(torch.zeros((1, gnp.d_latents)), torch.ones((1, gnp.d_latents))),
                        1).entropy()
                    visualize_fitting_acquisition(pre_selection_context_data, sampled_unlabeled_data, info_gain,
                                                  max_entropy,
                                                  figpath='')
                means_agg.append(np.zeros(gnp.d_latents) * np.NaN)
                covars_agg.append(np.zeros(gnp.d_latents) * np.NaN)
                entropies.append(np.NaN)
                info_gains.append(np.NaN)

            else:
                # TODO: integrate pybullet for comparison to ground truth IG
                ensemble = logger.get_ensemble(tx)
                if torch.cuda.is_available():
                    ensemble = ensemble.cuda()
                probs, labels = get_predictions_with_particles(
                    resampled_parts,
                    val_grasp_data,
                    ensemble,
                    n_particle_samples=50
                )
                means_agg.append(np.zeros(ensemble.models[0].d_latents) * np.NaN)
                covars_agg.append(np.zeros(ensemble.models[0].d_latents) * np.NaN)
                entropies.append(np.NaN)
                info_gains.append(np.NaN)
        else:
            context_data, sampled_unlabeled_data = logger.load_acquisition_data(tx)
            gnp = logger.get_neural_process(tx)
            if torch.cuda.is_available():
                gnp = gnp.cuda()
            # Write function to get predictions given a set of context data.
            probs, labels, means, covars, entropy, inf_time = get_gnp_contextualized_gnp_predictions(
                gnp,
                context_data,
                val_grasp_data,
            )
            times.append(inf_time)
            # we have to drop the last grasp in the context set to see what the ig comp looked like

            pre_selection_context_data = drop_last_grasp_in_dataset(context_data)
            info_gain, _, _ = compute_ig(gnp, pre_selection_context_data, sampled_unlabeled_data)
            info_gains.append(info_gain)
            if vis:
                max_entropy = torch.distributions.Independent(
                    torch.distributions.Normal(torch.zeros((1, gnp.d_latents)), torch.ones((1, gnp.d_latents))),
                    1).entropy()
                visualize_fitting_acquisition(pre_selection_context_data, sampled_unlabeled_data, info_gain,
                                              max_entropy,
                                              figpath='')

            means_agg.append(means.numpy())
            covars_agg.append(covars.numpy())
            entropies.append(entropy.numpy())

        probs, labels = probs.squeeze(), labels.squeeze()
        thresholds = np.arange(0.05, 1.0, 0.05)
        for threshold in thresholds:
            preds = (probs > threshold).float()
            rec = recall_score(labels, preds)
            str_t = f'{threshold: .2f}'
            if str_t not in thresholded_recalls:
                thresholded_recalls[str_t] = []
            thresholded_recalls[str_t].append(rec)

        preds = (probs > 0.5).float()
        print('Shapes:', probs.shape, labels.shape, preds.shape)
        av_prec = average_precision_score(labels, probs)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=np.nan)
        rec = recall_score(labels, preds, zero_division=np.nan)
        confs = confusion_matrix(labels, preds)
        f1 = f1_score(labels, preds, zero_division=np.nan)
        b_acc = balanced_accuracy_score(labels, preds)

        print(f'Acc: {acc}\tAverage Prec: {av_prec}\tPrecision: {prec}\tRecall: {rec}\tMean: {preds.mean()}')
        accs.append(acc)
        av_precs.append(av_prec)
        precisions.append(prec)
        recalls.append(rec)
        confusions.append(confs)
        f1s.append(f1)
        balanced_accs.append(b_acc)

    # print(times)
    # import ipdb; ipdb.set_trace()

    with open(logger.get_figure_path('val_accuracies.pkl'), 'wb') as handle:
        pickle.dump(accs, handle)
    with open(logger.get_figure_path('val_precisions.pkl'), 'wb') as handle:
        pickle.dump(precisions, handle)
    with open(logger.get_figure_path('val_average_precisions.pkl'), 'wb') as handle:
        pickle.dump(av_precs, handle)
    with open(logger.get_figure_path('val_recalls.pkl'), 'wb') as handle:
        pickle.dump(recalls, handle)
    with open(logger.get_figure_path('val_confusions.pkl'), 'wb') as handle:
        pickle.dump(confusions, handle)
    with open(logger.get_figure_path('val_f1s.pkl'), 'wb') as handle:
        pickle.dump(f1s, handle)
    with open(logger.get_figure_path('val_balanced_accs.pkl'), 'wb') as handle:
        pickle.dump(balanced_accs, handle)

    if use_progressive_priors:
        with open(logger.get_figure_path('val_means.pkl'), 'wb') as handle:
            pickle.dump(means_agg, handle)
        with open(logger.get_figure_path('val_covars.pkl'), 'wb') as handle:
            pickle.dump(covars_agg, handle)
        with open(logger.get_figure_path('val_entropies.pkl'), 'wb') as handle:
            pickle.dump(entropies, handle)
        with open(logger.get_figure_path('val_info_gains.pkl'), 'wb') as handle:
            pickle.dump(info_gains, handle)

    for k, v in thresholded_recalls.items():
        with open(logger.get_figure_path(f'val_recalls_{k}.pkl'), 'wb') as handle:
            pickle.dump(v, handle)

    return accs


def get_acquired_preditctions_pf(logger):
    pf_args = logger.args
    latent_ensemble = logger.get_ensemble(0)
    if torch.cuda.is_available():
        latent_ensemble.cuda()

    object_set = get_train_and_fit_objects(
        pretrained_ensemble_path=pf_args.pretrained_ensemble_exp_path,
        use_latents=True,
        fit_objects_fname=pf_args.objects_fname,
        fit_object_ix=pf_args.eval_object_ix
    )
    print('Total objects:', len(object_set['object_names']))
    pf_args.num_eval_objects = 1
    pf_args.num_train_objects = len(object_set['object_names']) - pf_args.num_eval_objects

    pf = GraspingDiscreteLikelihoodParticleBelief(
        object_set=object_set,
        D=latent_ensemble.d_latents,
        N=pf_args.n_particles,
        likelihood=latent_ensemble,
        plot=True
    )

    for tx in range(1, 11):
        particles = logger.load_particles(tx)
        pf.particles = particles

        grasp_data, _ = logger.load_acquisition_data(tx)

        preds = pf.get_particle_likelihoods(pf.particles.particles, grasp_data).reshape((1, pf_args.n_particles))
        print(preds.shape)
        score = bald(torch.Tensor(preds))
        print(score, preds.mean())

        import IPython
        IPython.embed()


def viz_latents(locs, scales, lim=4, fname=''):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    n_plot = 500
    if lim < 1:
        ax.plot([-lim, lim], [0, 0], [0, 0], c='r')
        ax.plot([0, 0], [-lim, lim], [0, 0], c='r')
        ax.plot([0, 0], [0, 0], [-lim, lim], c='r')

    ax.scatter(locs[:n_plot, 0],
               locs[:n_plot, 1],
               locs[:n_plot, 2], c=np.arange(n_plot), cmap=cm.tab10)

    if lim > 1:
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        for ix in range(n_plot):
            x = locs[ix, 0] + scales[ix, 0] * np.outer(np.cos(u), np.sin(v))
            y = locs[ix, 1] + scales[ix, 1] * np.outer(np.sin(u), np.sin(v))
            z = locs[ix, 2] + scales[ix, 2] * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color='b', alpha=0.05)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    if len(fname) > 0:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def plot_training_latents(logger):
    latent_ensemble = logger.get_ensemble(0)
    if torch.cuda.is_available():
        latent_ensemble.cuda()

    latent_locs = latent_ensemble.latent_locs.detach().numpy()
    latent_scales = np.exp(latent_ensemble.latent_logscales.detach().numpy())

    viz_latents(latent_locs, latent_scales)


def visualize_gnp_predictions():
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust/grasps/training_phase/val_grasps.pkl'
    with open('learning/experiments/metadata/grasp_np/results.pkl', 'rb') as handle:
        probs, targets = pickle.load(handle)

    predictions = (probs.flatten()) > 0.5
    labels = targets.flatten()

    figure_path = 'learning/experiments/metadata/grasp_np/figures'
    # visualize_grasp_dataset(val_dataset_fname, labels=predictions==labels)
    # visualize_grasp_dataset(val_dataset_fname, labels=predictions==labels, figpath=figure_path, prefix='correct_')
    # visualize_grasp_dataset(val_dataset_fname, labels=labels, figpath=figure_path, prefix='vhacd_labels_')
    # visualize_grasp_dataset(val_dataset_fname, labels=predictions, figpath=figure_path, prefix='predictions_')
    probs = probs.reshape(-1, 10)
    targets = targets.reshape(-1, 10)
    for ox in range(probs.shape[0]):
        acc = ((probs[ox] > 0.5) == targets[ox]).mean()
        if acc < 0.7:
            print(f'Object {ox}: {acc} True {targets[ox].mean()}')
        else:
            print(f'----- Object {ox}: {acc} True {targets[ox].mean()}')


def evaluate_trained_gnp(logger, train_fname, val_fname, exp_path):
    # Load GNP.
    gnp = logger.get_neural_process(tx=0)
    if torch.cuda.is_available():
        gnp.cuda()

    # Load datasets.
    with open(train_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    with open(val_fname, 'rb') as handle:
        val_data = pickle.load(handle)

    # Setup saving directory.
    fig_path = os.path.join(exp_path, 'figures', 'train_eval')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Get validation set predictions at different context levels (cache in experiment directory).
    context_sizes = [1, 2, 5, 10, 15, 20, 25, 50]
    results = {}
    for c_size in context_sizes:
        preds_path = os.path.join(fig_path, f'train_preds_context{c_size}.pkl')
        if os.path.exists(preds_path):
            with open(preds_path, 'rb') as handle:
                results[c_size] = pickle.load(handle)
            continue

        context_data = select_gnp_dataset_firstk(train_data, c_size)
        probs, labels, _, _, entropy = get_gnp_contextualized_gnp_predictions(
            gnp=gnp,
            context_data=context_data,
            val_data=val_data
        )
        res = {
            'preds': probs.numpy(),
            'labels': labels.numpy(),
            'entropy': entropy.numpy()
        }
        with open(preds_path, 'wb') as handle:
            pickle.dump(res, handle)
        results[c_size] = res

    # ----- Create plots/logs of useful metrics. -----

    # Get overall accuracies.
    all_object_acc = {}
    for c_size in context_sizes:
        preds = results[c_size]['preds']
        labels = results[c_size]['labels']
        acc = ((preds > 0.5) == labels).mean()
        acc_std = ((preds > 0.5) == labels).std()
        print(c_size, acc, acc_std)
        all_object_acc[c_size] = acc

    res_fname = os.path.join(fig_path, 'all_object_accs.json')
    with open(res_fname, 'w') as handle:
        json.dump(all_object_acc, handle)

    # Histograms of object-wise accuracy.
    object_wise_accuracies = {}
    for c_size in context_sizes:
        object_wise_accuracies[c_size] = []
        preds = results[c_size]['preds']
        labels = results[c_size]['labels']
        for ix in range(0, preds.shape[0]):
            obj_preds = preds[ix, :]
            obj_labels = labels[ix, :]
            acc = ((obj_preds > 0.5) == obj_labels).mean()
            object_wise_accuracies[c_size].append(acc)
        print('Mean:', np.mean(object_wise_accuracies[c_size]))
        print('Std:', np.std(object_wise_accuracies[c_size]))

        print(c_size, np.histogram(
            object_wise_accuracies[c_size],
            bins=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
            range=(0.0, 1.0)
        ))

    # Accuracy by object geometry.
    object_names = train_data['object_data']['object_names']

    geom_accs = {}
    for name, acc in zip(object_names, object_wise_accuracies[50]):
        if name not in geom_accs:
            geom_accs[name] = []
        geom_accs[name].append(acc)

    for name, acc_list in geom_accs.items():
        if np.mean(acc_list) < 0.7:
            print(name, acc_list)

    # Accuracy grouped by antipodal angle.
    points = val_data['grasp_data']['grasp_points']
    normals = val_data['grasp_data']['grasp_normal']
    ap_angles, correct = [], []
    for ox in range(0, len(points)):
        for gx in range(0, 10):
            point1 = points[ox][gx][0:1]
            point2 = points[ox][gx][1:2]
            normal1 = normals[ox][gx][0:1]
            normal2 = normals[ox][gx][1:2]
            direction = point2 - point1
            distance = np.linalg.norm(direction)
            error1 = pb_robot.geometry.batch_angle_between(normal1, -direction)
            error2 = pb_robot.geometry.batch_angle_between(normal2, direction)
            ap_angles.append(np.max([error1, error2]))
            ap_angles[-1] = distance
            correct.append((preds > 0.5)[ox, gx] == labels[ox, gx])

    from scipy.stats import binned_statistic
    stats = binned_statistic(ap_angles, correct, statistic='mean', bins=10, range=None)
    print(stats)

    # (Rejection rate vs. accuracy)@N plot.
    SN_ROOT = os.environ['SHAPENET_ROOT']
    metadata_fname = os.path.join(SN_ROOT, 'object_infos.pkl')
    with open(metadata_fname, 'rb') as handle:
        metadata = pickle.load(handle)

    plt.clf()
    for c_size in context_sizes:
        rrates, accs = [], []
        for ox, name in enumerate(object_names):
            rrates.append(metadata[name]['rejection_rate'])
            accs.append(object_wise_accuracies[c_size][ox])
        stats = binned_statistic(rrates, accs, statistic='mean', bins=50, range=(0.7, 0.9))

        plt.plot(stats.bin_edges[:-1], stats.statistic, label=f'N={c_size}')

    fname = os.path.join(fig_path, f'accs_by_rrate.png')
    plt.legend()
    plt.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, required=True)
    parser.add_argument('--train-dataset-fname', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    args = parser.parse_args()

    # logger = ActiveExperimentLogger(args.exp_path, use_latents=True)

    # get_validation_metrics(logger, args.val_dataset_fname)
    # get_pf_task_performance(logger, args.val_dataset_fname)
    # visualize_predictions(logger, args.val_dataset_fname)
    # combine_image_grids(logger, ['labels', 'predictions', 'correct'])
    visualize_gnp_predictions()
    # get_acquired_preditctions_pf(logger)
    # plot_training_latents(logger)

    # get_pf_validation_accuracy(logger, args.val_dataset_fname)
