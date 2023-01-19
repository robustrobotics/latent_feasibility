import argparse
import numpy as np
import torch

from learning.active import acquire
from learning.models import latent_ensemble
from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.active_utils import (
    get_fit_object, 
    sample_unlabeled_data,
    sample_unlabeled_gnp_data,
    get_labels,
    get_labels_gnp,
    get_train_and_fit_objects,
    select_gnp_dataset_ix,
    merge_gnp_datasets
)
from learning.domains.grasping.pybullet_likelihood import PBLikelihood
from learning.active.acquire import bald
from learning.models.grasp_np.train_grasp_np import check_to_cuda
from particle_belief import GraspingDiscreteLikelihoodParticleBelief, AmortizedGraspingDiscreteLikelihoodParticleBelief


def particle_bald(predictions, weights, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    predictions = predictions.cpu().numpy()
    norm = weights.sum()

    mp_c1 = np.sum(weights * predictions, axis=1) / norm
    mp_c0 = 1 - mp_c1

    m_ent = -(mp_c1 * np.log(mp_c1 + eps) + mp_c0 * np.log(mp_c0 + eps))

    p_c1 = predictions
    p_c0 = 1 - predictions

    ent_per_model = p_c1 * np.log(p_c1 + eps) + p_c0 * np.log(p_c0 + eps)
    ent = np.sum(ent_per_model * weights, axis=1) / norm

    bald = m_ent + ent
    return bald


def find_informative_tower(pf, object_set, logger, args):
    data_sampler_fn = lambda n: sample_unlabeled_data(n_samples=n, object_set=object_set)

    all_grasps = []
    all_preds = []
    for ix in range(0, args.n_samples):
        grasp_data = data_sampler_fn(1)
        preds = pf.get_particle_likelihoods(pf.particles.particles, grasp_data)
        all_preds.append(preds)
        all_grasps.append(grasp_data)

    pred_vec = torch.Tensor(np.stack(all_preds))
    scores = particle_bald(pred_vec, pf.particles.weights)
    print('Scores:', scores)
    acquire_ix = np.argsort(scores)[::-1][0]

    return all_grasps[acquire_ix]


def find_informative_tower_progressive_prior(gnp, current_context, unlabeled_samples,
                                             n_samples_from_latent_dist=5, batching_size=64):
    """
    :param current_context: Grasp dict object with collected datapoints so far.
        Most-nested dictionaries: ox: [].
    :param unlabeled_samples: The same format ^.
        Most-nested dictionaries: ox: [].
        Labels are meaningless (-1).
    :param n_samples_from_latent_dist: Number of samples from latent distribution when computing expected
    entropy of the posterior.
    :param batching_size: batching size used to speed up computation
    :return: The grasp index of the unlabeled samples with the highest info gain score.
    """

    n_unlabeled_sampled = len(unlabeled_samples['grasp_forces'])

    gnp.eval()
    with torch.no_grad():
        # only one object, so only one mesh is needed
        context_mesh = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(current_context['object_mesh'][0]),
                0, 1
            ),
            0
        )
        context_geoms = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(current_context['grasp_geometries']),
                1, 2
            ),
            0
        )
        context_grasp_points = torch.unsqueeze(torch.tensor(current_context['grasp_points']), 0)
        context_curvatures = torch.unsqueeze(torch.tensor(current_context['grasp_curvatures']), 0)
        context_midpoints = torch.unsqueeze(torch.tensor(current_context['grasp_midpoints']), 0)
        context_forces = torch.unsqueeze(torch.tensor(current_context['grasp_forces']), 0)
        context_labels = torch.unsqueeze(torch.tensor(current_context['grasp_labels']), 0)

        context_mesh, \
            context_geoms, \
            context_grasp_points, \
            context_curvatures, \
            context_midpoints, \
            context_forces, \
            context_labels = \
            check_to_cuda([
                context_mesh,
                context_geoms,
                context_grasp_points,
                context_curvatures,
                context_midpoints,
                context_forces,
                context_labels
            ])

        # compute H(z)
        q_z, mesh_enc = gnp.forward_until_latents(
            (context_geoms,
             context_grasp_points,
             context_curvatures,
             context_midpoints,
             context_forces,
             context_labels),
            context_mesh
        )
        h_z = q_z.entropy()

        # prepare unlabeled points as batched singletons
        unlabeled_mesh = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(unlabeled_samples['object_mesh']),
                1, 2
            ),
            1
        )
        unlabeled_geoms = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(unlabeled_samples['grasp_geometries']),
                1, 2
            ),
            1
        )
        unlabeled_grasp_points = torch.unsqueeze(torch.tensor(unlabeled_samples['grasp_points']), 1)
        unlabeled_curvatures = torch.unsqueeze(torch.tensor(unlabeled_samples['grasp_curvatures']), 1)
        unlabeled_midpoints = torch.unsqueeze(torch.tensor(unlabeled_samples['grasp_midpoints']), 1)
        unlabeled_forces = torch.unsqueeze(torch.tensor(unlabeled_samples['grasp_forces']), 1)

        # compute p(y|D,x)
        q_z_batched = q_z.expand((n_unlabeled_sampled,))
        zs = q_z_batched.sample((n_samples_from_latent_dist,))

        p_y_equals_one_cond_D_x = torch.zeros(n_unlabeled_sampled)
        for z_ix in range(n_samples_from_latent_dist):
            _, _, _, n_pts = unlabeled_geoms.shape
            geoms = unlabeled_geoms.view(-1, 3, n_pts)
            geoms_enc = gnp.grasp_geom_encoder(geoms).view(n_unlabeled_sampled, 1, -1)
            # TODO: check with Mike on this computation to marginalize out z
            p_y_equals_one_cond_D_x += gnp.decoder(geoms_enc,
                                                   unlabeled_grasp_points,
                                                   unlabeled_curvatures,
                                                   unlabeled_midpoints,
                                                   unlabeled_forces,
                                                   zs[z_ix, :],
                                                   mesh_enc)

        # next, we create all the candidate context sets and then compute the entropy of the latent distribution
        # for each
        candidate_geoms, candidate_grasp_points, candidate_curvatures, candidate_midpoints, candidate_forces = \
            [torch.cat([
                c_set.squeeze().broadcast_to(n_unlabeled_sampled, *c_set.shape),
                u_set.unsqueeze(1)
            ], dim=1)
                for c_set, u_set in zip(
                [context_geoms, context_grasp_points, context_curvatures, context_midpoints, context_forces],
                [unlabeled_geoms, unlabeled_grasp_points, unlabeled_curvatures, unlabeled_midpoints,
                 unlabeled_forces])
            ]

        candidate_labels_zero = torch.cat([
            context_labels.squeeze().broadcast_to(n_unlabeled_sampled, *context_labels.shape),
            torch.zeros((n_unlabeled_sampled, 1))
        ])

        candidate_labels_one = torch.cat([
            context_labels.squeeze().broadcast_to(n_unlabeled_sampled, *context_labels.shape),
            torch.ones((n_unlabeled_sampled, 1))
        ])

        h_z_cond_x_y_equals_zero = gnp.forward_until_latents(
            (candidate_geoms, candidate_grasp_points, candidate_curvatures,
             candidate_midpoints, candidate_forces, candidate_labels_zero),
            unlabeled_mesh
        )[0].entropy()

        h_z_cond_x_y_equals_one = gnp.forward_until_latents(
            (candidate_geoms, candidate_grasp_points, candidate_curvatures,
             candidate_midpoints, candidate_forces, candidate_labels_one),
            unlabeled_mesh
        )[0].entropy()

        # compute expected entropy and information gain
        expected_h_z_cond_x_y = p_y_equals_one_cond_D_x * h_z_cond_x_y_equals_one + (
                1 - p_y_equals_one_cond_D_x) * h_z_cond_x_y_equals_zero
        info_gain = h_z - expected_h_z_cond_x_y

        # return the index with the largest information gain
        return torch.argmax(info_gain)


def dummy_info_gain(gnp, current_context, unlabeled_samples):
    """
    Return first element for testing.
    """
    return 0


def amoritized_filter_loop(gnp, object_set, logger, strategy, args):
    print('----- Running fitting phase with learned progressive priors -----')
    logger.save_neural_process(gnp, 0, symlink_tx0=False)

    # Initialize data dictionary in GNP format with a random data point.
    context_data = sample_unlabeled_gnp_data(n_samples=1, object_set=object_set, object_ix=args.eval_object_ix)
    context_data = get_labels_gnp(context_data)

    logger.save_acquisition_data(context_data, None, 0)

    # All random grasps end up getting labeled, so parallelize this.
    if strategy == 'random':
        random_pool = sample_unlabeled_gnp_data(
            n_samples=args.max_acquisitions,
            object_set=object_set,
            object_ix=args.eval_object_ix
        )
        random_pool = get_labels_gnp(random_pool)

    for tx in range(0, args.max_acquisitions):
        print('[AmortizedFilter] Interaction Number', tx)

        if strategy == 'random':
            grasp_dataset = select_gnp_dataset_ix(random_pool, tx)
        elif strategy == 'bald':
            # TODO: Sample target unlabeled dataset in parallel fashion.
            unlabeled_dataset = sample_unlabeled_gnp_data(args.n_samples, object_set, object_ix=args.eval_object_ix)
            best_idx = dummy_info_gain(gnp, context_data, unlabeled_dataset)
            # Get the observation for the chosen grasp.
            grasp_dataset = select_gnp_dataset_ix(unlabeled_dataset, best_idx)
            grasp_dataset = get_labels_gnp(grasp_dataset)
        else:
            raise NotImplementedError()

        # Add datapoint to context dictionary.
        context_data = merge_gnp_datasets(context_data, grasp_dataset)
<<<<<<< HEAD
        logger.save_neural_process(gnp, tx+1, symlink_tx0=True)
        logger.save_acquisition_data(context_data, None, tx+1)
=======

        logger.save_neural_process(gnp, tx + 1, symlink_tx0=True)
        logger.save_acquisition_data(context_data, None, tx + 1)
>>>>>>> implement a probably too memory-intensive IG selector... will need to break into batches instead of computing all data in one huge batch


def particle_filter_loop(pf, object_set, logger, strategy, args):
    if args.likelihood == 'nn':
        logger.save_ensemble(pf.likelihood, 0, symlink_tx0=False)
    elif args.likelihood == 'gnp':
        logger.save_neural_process(pf.likelihood, 0, symlink_tx0=False)
    logger.save_particles(pf.particles, 0)

    for tx in range(0, args.max_acquisitions):
        print('[ParticleFilter] Interaction Number', tx)

        # Choose a tower to build that includes the new block.
        if strategy == 'random':
            data_sampler_fn = lambda n: sample_unlabeled_data(n_samples=n, object_set=object_set)
            grasp_dataset = data_sampler_fn(1)
        elif strategy == 'bald':
            grasp_dataset = find_informative_tower(pf, object_set, logger, args)
        else:
            raise NotImplementedError()

        # Get the observation for the chosen tower.
        grasp_dataset = get_labels(grasp_dataset)

        # Update the particle belief.
        particles, means = pf.update(grasp_dataset)
        print('[ParticleFilter] Particle Statistics')
        print(f'Min Weight: {np.min(pf.particles.weights)}')
        print(f'Max Weight: {np.max(pf.particles.weights)}')
        print(f'Sum Weights: {np.sum(pf.particles.weights)}')

        # Save the model and particle distribution at each step.
        if args.likelihood == 'nn':
            logger.save_ensemble(pf.likelihood, tx + 1, symlink_tx0=True)
        elif args.likelihood == 'gnp':
            logger.save_neural_process(pf.likelihood, tx + 1, symlink_tx0=True)
        logger.save_acquisition_data(grasp_dataset, None, tx + 1)
        logger.save_particles(particles, tx + 1)


# "grasp_train-ycb-test-ycb-1_fit_random_train_geo_object0": "learning/experiments/logs/grasp_train-ycb-test-ycb-1_fit_random_train_geo_object0-20220504-134253"
def run_particle_filter_fitting(args):
    print(args)
    args.use_latents = True
    args.fit_pf = True

    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # ----- Load the block set -----
    print('Loading objects:', args.objects_fname)
    object_set = get_train_and_fit_objects(
        pretrained_ensemble_path=args.pretrained_ensemble_exp_path,
        use_latents=True,
        fit_objects_fname=args.objects_fname,
        fit_object_ix=args.eval_object_ix
    )
    print('Total objects:', len(object_set['object_names']))
    args.num_eval_objects = 1
    args.num_train_objects = len(object_set['object_names']) - args.num_eval_objects

    # ----- Likelihood Model -----
    if args.likelihood == 'nn':
        train_logger = ActiveExperimentLogger(
            exp_path=args.pretrained_ensemble_exp_path,
            use_latents=True
        )
        latent_ensemble = train_logger.get_ensemble(args.ensemble_tx)
        if torch.cuda.is_available():
            latent_ensemble.cuda()
        latent_ensemble.add_latents(1)
        likelihood_model = latent_ensemble
        d_latents = latent_ensemble.d_latents
    elif args.likelihood == 'gnp':
        train_logger = ActiveExperimentLogger(
            exp_path=args.pretrained_ensemble_exp_path,
            use_latents=False
        )
        likelihood_model = train_logger.get_neural_process(tx=0)
        if torch.cuda.is_available():
            likelihood_model.cuda()
        likelihood_model.eval()
        d_latents = likelihood_model.d_latents
    elif args.likelihood == 'pb':
        likelihood_model = PBLikelihood(
            object_name=object_set['object_names'][-1],
            n_samples=5,
            batch_size=50
        )
        d_latents = 5

    # ----- Initialize particle filter from prior -----
    if args.likelihood == 'gnp':
        pf = AmortizedGraspingDiscreteLikelihoodParticleBelief(
            object_set=object_set,
            d_latents=d_latents,
            n_particles=args.n_particles,
            likelihood=likelihood_model,
            resample=False,
            plot=False
        )
    else:
        pf = GraspingDiscreteLikelihoodParticleBelief(
            object_set=object_set,
            D=d_latents,
            N=args.n_particles,
            likelihood=likelihood_model,
            resample=False,
            plot=False)
    if args.likelihood == 'pb':
        pf.particles = likelihood_model.init_particles(args.n_particles)

    # ----- Run particle filter loop -----
    if args.use_progressive_priors:
        amoritized_filter_loop(likelihood_model, object_set, logger, args.strategy, args)
    else:
        particle_filter_loop(pf, object_set, logger, args.strategy, args)

    return logger.exp_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='',
                        help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--max-acquisitions', type=int, default=25,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--objects-fname', type=str, default='', help='File containing a list of objects to grasp.')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--pretrained-ensemble-exp-path', type=str, default='', help='Path to a trained ensemble.')
    parser.add_argument('--ensemble-tx', type=int, default=-1, help='Timestep of the trained ensemble to evaluate.')
    parser.add_argument('--eval-object-ix', type=int, default=0, help='Index of which eval object to use.')
    parser.add_argument('--strategy', type=str, choices=['bald', 'random', 'task'], default='bald')
    parser.add_argument('--n-particles', type=int, default=100)
    parser.add_argument('--likelihood', choices=['nn', 'pb', 'gnp'], default='nn')
    args = parser.parse_args()

    run_particle_filter_fitting(args)
