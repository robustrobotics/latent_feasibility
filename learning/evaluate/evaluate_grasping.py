import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, confusion_matrix, accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from learning.domains.grasping.active_utils import get_fit_object, sample_unlabeled_data, get_labels, get_train_and_fit_objects
from learning.active.acquire import bald

from block_utils import ParticleDistribution
from filter_utils import sample_particle_distribution
from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.grasp_data import GraspDataset, GraspParallelDataLoader
from learning.domains.grasping.explore_dataset import visualize_grasp_dataset
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.train_grasp_np import check_to_cuda
from particle_belief import GraspingDiscreteLikelihoodParticleBelief


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
    visualize_grasp_dataset(val_dataset_fname, labels=predictions==labels, figpath=figure_path, prefix='correct_')
    visualize_grasp_dataset(val_dataset_fname, labels=labels, figpath=figure_path, prefix='labels_')
    visualize_grasp_dataset(val_dataset_fname, labels=predictions, figpath=figure_path, prefix='predictions_')

def combine_gnp_preds(image_root, object_ids, strategy):
    images = []
    for ox in object_ids:
        fname = os.path.join(
            f'{image_root}/object{ox}__{strategy}_targets_x.png'
        )
        images.append(plt.imread(fname))
        fname = os.path.join(
            f'{image_root}/object{ox}__{strategy}_errors_x.png'
        )
        images.append(plt.imread(fname))
        fname = os.path.join(
            f'{image_root}/object{ox}__{strategy}_preds_x.png'
        )
        images.append(plt.imread(fname))

    fig = plt.figure(figsize=(15,5))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(object_ids), 3))

    for ax, im in zip(grid, images):
        print(im.shape)
        im = im[200:, 500:, :]
        im = im[:-200, :-500, :]
        ax.imshow(im)
        ax.axis('off')

    plt.savefig(os.path.join(image_root, f'combined_{strategy}.png'), bbox_inches='tight', dpi=500)

def combine_image_grids(logger, prefixes):
    for ix in range(0, 50):
        for angle in ['x', 'y', 'z']:
            images = []
            for p in prefixes:
                fname = logger.get_figure_path('%s_%d_%s.png' % (p, ix, angle))
                images.append(plt.imread(fname))

            fig = plt.figure(figsize=(5,15))
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

            #latents = latent_samples[ix*100:(ix+1)*100,:]
            latents = latent_samples[latents_ix, :]
            if torch.cuda.is_available():
                latents = latents.cuda()
            pred = ensemble.forward(X=grasps[:, :-5, :],
                                    object_ids=object_ids,
                                    N_samples=n_particle_samples,
                                    ensemble_idx=-1,#ensemble_ix,
                                    collapse_latents=True, 
                                    collapse_ensemble=True,
                                    pf_latent_ix=100,
                                    latent_samples=latents).squeeze()

            preds.append(pred.mean(dim=-1))
            labels.append(y)
            # if (len(preds)*16) > 200: break
    return torch.cat(preds, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


def get_gnp_predictions(train_data, val_data, gnp):
    val_dataset_eval = CustomGNPGraspDataset(data=val_data, context_data=train_data)
    val_dataloader = DataLoader(
        dataset=val_dataset_eval,
        collate_fn=custom_collate_fn,
        batch_size=16,
        shuffle=False
    )
    gnp.eval()
    val_probs, val_targets = [], []
    with torch.no_grad():
        for _, (context_data, target_data, meshes) in enumerate(val_dataloader):
            c_grasp_geoms, c_midpoints, c_forces, c_labels = check_to_cuda(context_data)
            t_grasp_geoms, t_midpoints, t_forces, t_labels = check_to_cuda(target_data)
            if torch.cuda.is_available():
                meshes = meshes.cuda()
            y_probs, q_z = gnp.forward(
                (c_grasp_geoms, c_midpoints, c_forces, c_labels),
                (t_grasp_geoms, t_midpoints, t_forces),
                meshes,
                decoder_ix=-1
            )
            y_probs = y_probs.mean(dim=-1).squeeze()

            val_probs.append(y_probs.flatten())
            val_targets.append(t_labels.flatten())

    return torch.cat(val_probs, dim=0).cpu(), torch.cat(val_targets, dim=0).cpu()


def get_gnp_predictions_with_particles(particles, grasp_data, gnp, n_particle_samples=10):
    preds, labels = [], []
    dataset = CustomGNPGraspDataset(
        data=grasp_data,
        context_data=grasp_data
    )
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate_fn,
        batch_size=1,
        shuffle=False
    )

    latent_samples = torch.Tensor(particles)
    gnp.eval()
    for (_, target_data, meshes) in dataloader:
        t_grasp_geoms, t_midpoints, t_forces, t_labels = target_data

        # TODO: Currently there are too many grasps (500 requires too much memory).
        # TODO: Implement better way to batch a lot of evaluation grasps.
        t_grasp_geoms = t_grasp_geoms[:, :100, :, :]
        t_midpoints = t_midpoints[:, :100, :]
        t_forces = t_forces[:, :100]
        t_labels = t_labels[:, :100].squeeze()

        if torch.cuda.is_available():
            meshes = meshes.cuda()
            t_grasp_geoms = t_grasp_geoms.cuda()
            t_midpoints = t_midpoints.cuda()
            t_forces = t_forces.cuda()
        t_grasp_geoms = t_grasp_geoms.expand(n_particle_samples, -1, -1, -1)
        t_midpoints = t_midpoints.expand(n_particle_samples, -1, -1)
        t_forces = t_forces.expand(n_particle_samples, -1)
        
        # Sample particles and ensembles models to use to speed up evaluation. Might hurt performance.
        latents_ix = np.arange(latent_samples.shape[0])
        np.random.shuffle(latents_ix)
        latents_ix = latents_ix[:n_particle_samples]

        latents = latent_samples[latents_ix, :]
        if torch.cuda.is_available():
            latents = latents.cuda()
        pred = gnp.conditional_forward(
            target_xs = (t_grasp_geoms, t_midpoints, t_forces),
            meshes=meshes,
            zs=latents
        ).squeeze().cpu().detach()

        preds.append(pred.mean(dim=0))
        labels.append(t_labels)

    return torch.cat(preds, dim=0).cpu(), torch.cat(labels, dim=0).cpu()

def get_pf_task_performance(logger, fname):
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

    regrets = []
    eval_range = range(0, logger.args.max_acquisitions, 1)
    for tx in eval_range:
        print('Eval timestep, ', tx)

        particles = logger.load_particles(tx)

        # This is necessary if we're using a weighted particle filter.
        sampling_dist = ParticleDistribution(
            particles.particles,
            particles.weights/np.sum(particles.weights)
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
        probs = probs.cpu().numpy()
        labels = labels.cpu().numpy()

        max_force = 20
        neg_reward = -max_force

        max_reward = neg_reward
        max_exp_reward = neg_reward
        max_achieved_reward = neg_reward

        print(f'# Stable: {np.sum(labels)}/{len(labels)}')
        for force, prob, label in zip(grasp_forces, probs, labels):
            pos_reward = (max_force - force)
            exp_reward = neg_reward * (1-prob) + prob*pos_reward
            true_reward = pos_reward if label else neg_reward

            if (true_reward > max_reward) and (label == 1):
                max_reward = true_reward

            if exp_reward > max_exp_reward:
                max_exp_reward = exp_reward
                max_achieved_reward = true_reward

            # if label == 1:
            #     print(f'F: {force}\tProb: {prob}\tExpR: {exp_reward}')

        if max_achieved_reward == neg_reward:
            regret = 1
        else:
            regret = (max_reward - max_achieved_reward)/15
        regrets.append(regret)
        print(f'Max Reward: {max_reward}\tReward: {max_achieved_reward}\tRegret: {regret}')

        with open(logger.get_figure_path(f'regrets_{neg_reward}.pkl'), 'wb') as handle:
            pickle.dump(regrets, handle)

def get_pf_validation_accuracy(logger, fname, amortize):
    accs, precisions, recalls, f1s, balanced_accs, av_precs = [], [], [], [], [], []
    thresholded_recalls = {}
    confusions = []

    with open(fname, 'rb') as handle:
        val_grasp_data = pickle.load(handle)

    eval_range = range(0, logger.args.max_acquisitions, 1)
    for tx in eval_range:
        print('Eval timestep, ', tx)
        particles = logger.load_particles(tx)

        # This is necessary if we're using a weighted particle filter.
        sampling_dist = ParticleDistribution(
            particles.particles,
            particles.weights/np.sum(particles.weights)
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
        else:
            ensemble = logger.get_ensemble(tx)
            if torch.cuda.is_available():
                ensemble = ensemble.cuda()
            probs, labels = get_predictions_with_particles(
                resampled_parts,
                val_grasp_data,
                ensemble,
                n_particle_samples=50
            )

        thresholds = np.arange(0.05, 1.0, 0.05)
        for threshold in thresholds:
            preds = (probs > threshold).float()
            rec = recall_score(labels, preds)
            str_t = f'{threshold: .2f}'
            if str_t not in thresholded_recalls:
                thresholded_recalls[str_t] = []
            thresholded_recalls[str_t].append(rec)

        preds = (probs > 0.5).float()
        av_prec = average_precision_score(labels, probs)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        confs = confusion_matrix(labels, preds)
        f1 = f1_score(labels, preds)
        b_acc = balanced_accuracy_score(labels, preds)

        print(f'Acc: {acc}\tAverage Prec: {av_prec}\tPrecision: {prec}\tRecall: {rec}\tF1: {f1}')
        accs.append(acc)
        av_precs.append(av_prec)
        precisions.append(prec)
        recalls.append(rec)
        confusions.append(confs)
        f1s.append(f1)
        balanced_accs.append(b_acc)

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
    #visualize_grasp_dataset(val_dataset_fname, labels=predictions==labels, figpath=figure_path, prefix='correct_')
    #visualize_grasp_dataset(val_dataset_fname, labels=labels, figpath=figure_path, prefix='vhacd_labels_')
    # visualize_grasp_dataset(val_dataset_fname, labels=predictions, figpath=figure_path, prefix='predictions_')
    probs = probs.reshape(-1, 10)
    targets = targets.reshape(-1, 10)
    for ox in range(probs.shape[0]):
        acc = ((probs[ox] > 0.5) == targets[ox]).mean()
        if acc < 0.7:
            print(f'Object {ox}: {acc} True {targets[ox].mean()}')
        else:
            print(f'----- Object {ox}: {acc} True {targets[ox].mean()}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    # parser.add_argument('--val-dataset-fname', type=str, required=True)
    args = parser.parse_args()

    #logger = ActiveExperimentLogger(args.exp_path, use_latents=True)

    #get_validation_metrics(logger, args.val_dataset_fname)
    #get_pf_task_performance(logger, args.val_dataset_fname)
    #visualize_predictions(logger, args.val_dataset_fname)
    #combine_image_grids(logger, ['labels', 'predictions', 'correct'])
    visualize_gnp_predictions()
    #get_acquired_preditctions_pf(logger)
    #plot_training_latents(logger)
    #get_pf_validation_accuracy(logger, args.val_dataset_fname)