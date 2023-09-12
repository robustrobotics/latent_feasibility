import argparse
import copy
import math
import pickle
import time

import numpy as np
import torch

from agents.panda_agent import PandaClientAgent
from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.active_utils import (
    get_fit_object,
    sample_unlabeled_data,
    sample_unlabeled_gnp_data,
    get_labels,
    get_label_gnp,
    get_labels_gnp,
    get_train_and_fit_objects,
    get_fit_object,
    select_gnp_dataset_ix,
    merge_gnp_datasets,
    gnp_dataset_from_raw_grasps, explode_dataset_into_list_of_datasets
)
from learning.domains.grasping.pybullet_likelihood import PBLikelihood
from learning.domains.grasping.tamp_grasping import GraspingAgent
from learning.models.grasp_np.dataset import CustomGNPGraspDataset
from learning.models.grasp_np.train_grasp_np import check_to_cuda
from learning.models.grasp_np.create_gnp_data import process_geometry
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

    # NOTE: this is the computation we need for IG for visualization for particle filtering
    pred_vec = torch.Tensor(np.stack(all_preds))
    scores = particle_bald(pred_vec, pf.particles.weights)
    print('Scores:', scores)
    acquire_ix = np.argsort(scores)[::-1][0]

    # TODO: this is the choice we need to override
    # merge all the grasps into the same dataset
    merge_set = all_grasps[0]
    for grasp in all_grasps[1:]:
        merge_set = merge_gnp_datasets(merge_set, grasp)

    return all_grasps[acquire_ix], merge_set


def compute_ig(gnp, current_context, unlabeled_samples, n_samples_from_latent_dist=32, batching_size=20):
    """
    :param current_context: Grasp dict object with collected datapoints so far.
        Most-nested dictionaries: ox: [].
    :param unlabeled_samples: The same format ^.
        Most-nested dictionaries: ox: [].
        Labels are meaningless (-1).
    :param n_samples_from_latent_dist: Number of samples from latent distribution when computing expected
    entropy of the posterior.
    :param batching_size: batching size used to speed up computation
    :return: An array of IG scores with same length as unlabeled_samples, with index correspondence between the two.
    """
    if list(current_context['grasp_data']['labels'].values())[0]:
        dataset = CustomGNPGraspDataset(data=unlabeled_samples, context_data=current_context)
        context_data, unlabeled_data = dataset[0]  # apply post-processing
    else:
        dataset = CustomGNPGraspDataset(data=unlabeled_samples)
        context_data, unlabeled_data = dataset[0]

    n_unlabeled_sampled = len(unlabeled_data['grasp_forces'])
    n_batches = math.ceil(float(n_unlabeled_sampled) / batching_size)
    gnp.eval()

    if torch.cuda.is_available():
        gnp.cuda()

    latent_pred_st = time.process_time()
    with torch.no_grad():
        if context_data is not None:
            context_mesh = torch.unsqueeze(
                torch.swapaxes(
                    torch.tensor(context_data['object_mesh']),
                    0, 1
                ),
                0
            )

            context_geoms = torch.unsqueeze(
                torch.swapaxes(
                    torch.tensor(context_data['grasp_geometries']),
                    1, 2
                ),
                0
            )

            context_grasp_points = torch.unsqueeze(torch.tensor(context_data['grasp_points']), 0)
            context_curvatures = torch.unsqueeze(torch.tensor(context_data['grasp_curvatures']), 0)
            context_normals = torch.unsqueeze(torch.tensor(context_data['grasp_normals']), 0)
            context_midpoints = torch.unsqueeze(torch.tensor(context_data['grasp_midpoints']), 0)
            context_forces = torch.unsqueeze(torch.tensor(context_data['grasp_forces']), 0)
            context_labels = torch.unsqueeze(torch.tensor(context_data['grasp_labels']), 0)

            context_mesh, \
                context_geoms, \
                context_grasp_points, \
                context_curvatures, \
                context_normals, \
                context_midpoints, \
                context_forces, \
                context_labels = \
                check_to_cuda([
                    context_mesh,
                    context_geoms,
                    context_grasp_points,
                    context_curvatures,
                    context_normals,
                    context_midpoints,
                    context_forces,
                    context_labels
                ])
            
            context_sizes = torch.ones_like(context_forces)*context_forces.shape[1]/50

            # compute H(z)
            q_z, mesh_enc, _ = gnp.forward_until_latents(
                (context_geoms,
                 context_grasp_points,
                 context_curvatures,
                 context_normals,
                 context_midpoints,
                 context_forces,
                 context_labels,
                 context_sizes),
                context_mesh
            )
            # need to reinterpret as a multivariate Gaussian and then compute entropy
            h_z = torch.distributions.Independent(q_z, 1).entropy()
        else:
            # if there is no context data, we'll use an uninformed prior
            zeros = torch.zeros((1, gnp.d_latents))
            ones = torch.ones((1, gnp.d_latents))
            zeros, ones = check_to_cuda([zeros, ones])
            q_z = torch.distributions.Normal(zeros, ones)
            h_z = torch.distributions.Independent(q_z, 1).entropy()

        # ======> INFERENCE TIMING END
        latent_pred_et = time.process_time()

        # ======> INFO GAIN COMPUTATION TIME START
        ig_st = time.process_time()
        # prepare unlabeled points as batched singletons
        unlabeled_mesh = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(unlabeled_data['object_mesh']),
                0, 1
            ),
            0
        ).repeat(n_unlabeled_sampled, 1, 1)
        unlabeled_geoms = torch.unsqueeze(
            torch.swapaxes(
                torch.tensor(unlabeled_data['grasp_geometries']),
                1, 2
            ),
            1
        )
        unlabeled_grasp_points = torch.unsqueeze(
            torch.tensor(unlabeled_data['grasp_points']), 1)
        unlabeled_curvatures = torch.unsqueeze(
            torch.tensor(unlabeled_data['grasp_curvatures']), 1)
        unlabeled_normals = torch.unsqueeze(
            torch.tensor(unlabeled_data['grasp_normals']), 1)
        unlabeled_midpoints = torch.unsqueeze(
            torch.tensor(unlabeled_data['grasp_midpoints']), 1)
        unlabeled_forces = torch.unsqueeze(
            torch.tensor(unlabeled_data['grasp_forces']), 1)

        unlabeled_mesh, \
            unlabeled_geoms, \
            unlabeled_grasp_points, \
            unlabeled_curvatures, \
            unlabeled_normals, \
            unlabeled_midpoints, \
            unlabeled_forces = \
            check_to_cuda([
                unlabeled_mesh,
                unlabeled_geoms,
                unlabeled_grasp_points,
                unlabeled_curvatures,
                unlabeled_normals,
                unlabeled_midpoints,
                unlabeled_forces,
            ])

        # compute p(y|D,x)
        q_z_batched = q_z.expand((n_unlabeled_sampled, q_z.batch_shape[1]))
        zs = q_z_batched.sample((n_samples_from_latent_dist,))

        p_y_equals_one_cond_d_x = torch.zeros(n_unlabeled_sampled)
        p_y_equals_one_cond_d_x, = check_to_cuda([p_y_equals_one_cond_d_x])

        for z_ix in range(n_samples_from_latent_dist):
            _, _, _, n_pts = unlabeled_geoms.shape
            for batch_i in range(n_batches):
                start = batch_i * batching_size
                end = (batch_i + 1) * batching_size
                unlabeled_forces_batch = unlabeled_forces[start:end]
                p_y_equals_one_cond_d_x[start:end] += gnp.conditional_forward(
                    (
                        unlabeled_geoms[start:end],
                        unlabeled_grasp_points[start:end],
                        unlabeled_curvatures[start:end],
                        unlabeled_normals[start:end],
                        unlabeled_midpoints[start:end],
                        unlabeled_forces_batch,
                    ),
                    unlabeled_mesh[start:end],
                    zs[z_ix, start:end, :],
                ).squeeze()
        p_y_equals_one_cond_d_x /= n_samples_from_latent_dist

        # next, we create all the candidate context sets and then compute the entropy of the latent distribution
        # for each
        if context_data is not None:
            candidate_geoms, candidate_grasp_points, candidate_curvatures, candidate_normals, candidate_midpoints, candidate_forces = \
                [torch.cat([
                    c_set[0].broadcast_to(n_unlabeled_sampled, *c_set[0].shape),
                    u_set
                ], dim=1)
                    for c_set, u_set in zip(
                    [context_geoms, context_grasp_points, context_curvatures, context_normals, context_midpoints,
                     context_forces],
                    [unlabeled_geoms, unlabeled_grasp_points, unlabeled_curvatures, unlabeled_normals,
                     unlabeled_midpoints,
                     unlabeled_forces])
                ]
        else:
            # if there are no context, then the unlabeled sets are the actual candidate sets
            candidate_geoms, candidate_grasp_points, candidate_curvatures, candidate_normals, candidate_midpoints, candidate_forces = \
                unlabeled_geoms, unlabeled_grasp_points, unlabeled_curvatures, unlabeled_normals, unlabeled_midpoints, unlabeled_forces

        zero_labels = torch.zeros((n_unlabeled_sampled, 1))
        one_labels = torch.ones((n_unlabeled_sampled, 1))
        zero_labels, one_labels = check_to_cuda([zero_labels, one_labels])

        if context_data is not None:
            candidate_labels_zero = torch.cat([
                context_labels[0].broadcast_to(n_unlabeled_sampled, *context_labels[0].shape),
                zero_labels
            ], dim=1)

            candidate_labels_one = torch.cat([
                context_labels[0].broadcast_to(n_unlabeled_sampled, *context_labels[0].shape),
                one_labels
            ], dim=1)
        else:
            candidate_labels_zero = zero_labels
            candidate_labels_one = one_labels

        h_z_cond_x_y_equals_zero = torch.zeros(n_unlabeled_sampled)
        h_z_cond_x_y_equals_one = torch.zeros(n_unlabeled_sampled)
        h_z_cond_x_y_equals_zero, h_z_cond_x_y_equals_one = \
            check_to_cuda([h_z_cond_x_y_equals_zero, h_z_cond_x_y_equals_one])

        for batch_i in range(n_batches):
            start = batch_i * batching_size
            end = (batch_i + 1) * batching_size

            candidate_sizes = torch.ones_like(candidate_forces[start:end])*candidate_forces.shape[1]/50
            q_z_zero = gnp.forward_until_latents(
                (
                    candidate_geoms[start:end],
                    candidate_grasp_points[start:end],
                    candidate_curvatures[start:end],
                    candidate_normals[start:end],
                    candidate_midpoints[start:end],
                    candidate_forces[start:end],
                    candidate_labels_zero[start:end],
                    candidate_sizes
                ),
                unlabeled_mesh[start:end]
            )[0]
            h_z_cond_x_y_equals_zero[start:end] = torch.distributions.Independent(q_z_zero, 1).entropy()

            q_z_one = gnp.forward_until_latents(
                (
                    candidate_geoms[start:end],
                    candidate_grasp_points[start:end],
                    candidate_curvatures[start:end],
                    candidate_normals[start:end],
                    candidate_midpoints[start:end],
                    candidate_forces[start:end],
                    candidate_labels_one[start:end],
                    candidate_sizes
                ),
                unlabeled_mesh[start:end]
            )[0]
            h_z_cond_x_y_equals_one[start:end] = torch.distributions.Independent(q_z_one, 1).entropy()
        # compute expected entropy and information gain
        expected_h_z_cond_x_y = p_y_equals_one_cond_d_x * h_z_cond_x_y_equals_one + (
                1 - p_y_equals_one_cond_d_x) * h_z_cond_x_y_equals_zero
        info_gain = h_z - expected_h_z_cond_x_y
        # ======> INFO GAIN TIMING COMPUTATION END
        ig_et = time.process_time()

        latent_pred_time = latent_pred_et - latent_pred_st
        ig_time = ig_et - ig_st
        return info_gain.cpu().numpy(), latent_pred_time, ig_time


def amortized_filter_loop(gnp, object_set, logger, strategy, args, override_selection_fun=None):
    # We generate the datasets here, but evaluate_grasping is when we actually
    # play through the interaction data. So we will actually be saving the mean, covariance, and entropy data
    # over there.
    print('----- Running fitting phase with learned progressive priors -----')
    logger.save_neural_process(gnp, 0, symlink_tx0=False)

    constrained = args.constrained or args.exec_mode == 'real'
    # TODO: if we regularize with uninformed prior, we shouldn't start with a random sample.
    #  we should be using IG the uninformed prior
    # Initialize data dictionary in GNP format with a random data point.
    samples = sample_unlabeled_gnp_data(
        n_samples=args.n_samples,  # Why is this here? Shouldn't it be 1?
        object_set=object_set,
        object_ix=args.eval_object_ix
    )
    context_data = select_gnp_dataset_ix(samples, 0)
    context_data = get_labels_gnp(context_data)

    logger.save_acquisition_data(context_data, samples, 0)

    # All random grasps end up getting labeled, so parallelize this.
    if strategy == 'random' and not constrained:
        print('[Random]: Sampling')
        random_pool = sample_unlabeled_gnp_data(
            n_samples=args.max_acquisitions,
            object_set=object_set,
            object_ix=args.eval_object_ix
        )
        print('[Random]: Labeling')
        random_pool = get_labels_gnp(random_pool)

    if constrained:
        # Initialize PyBullet agent used for planning.
        name, properties, _ = get_fit_object(object_set)
        agent = GraspingAgent(
            object_name=name,
            object_properties=properties,
            use_gui=True
        )

    if args.exec_mode == 'sim':
        grasp_labeler = None
    elif args.exec_mode == 'real':
        grasp_labeler = PandaClientAgent()
    else:
        raise NotImplementedError()

    amortized_pred_times = []
    ig_compute_times = []
    for tx in range(0, args.max_acquisitions):
        print('[AmortizedFilter] Interaction Number', tx)

        if strategy == 'random' and not constrained:
            grasp_dataset = select_gnp_dataset_ix(random_pool, tx)
            context_data = merge_gnp_datasets(context_data, grasp_dataset)
            logger.save_neural_process(gnp, tx + 1, symlink_tx0=True)
            logger.save_acquisition_data(context_data, random_pool, tx + 1)

            amortized_pred_times.append(np.NaN)
            ig_compute_times.append(np.NaN)
        elif strategy == 'random' and constrained:
            # Sample a plan of horizon 1. Pick/place.
            # TODO: Update pose based on perception first.
            # if args.exec_mode == 'real':
            #     agent.set_object_pose(None, find_stable_z=True)

            grasp = agent._sample_grasp_action()
            # Execute plan. Set object state based on result.
            if args.exec_mode == 'real':
                label = grasp_labeler.get_label(grasp)
            else:
                label = agent.execute_first_action((grasp, None))

            # Convert plan to GNP data format.
            grasp_dataset = gnp_dataset_from_raw_grasps([grasp], [label], object_set, args.eval_object_ix)

            context_data = merge_gnp_datasets(context_data, grasp_dataset)
            logger.save_neural_process(gnp, tx + 1, symlink_tx0=True)
            logger.save_acquisition_data(context_data, random_pool, tx + 1)

            amortized_pred_times.append(np.NaN)
            ig_compute_times.append(np.NaN)
        elif strategy == 'bald':
            print('[BALD] Sampling...')
            unlabeled_dataset = sample_unlabeled_gnp_data(args.n_samples, object_set, object_ix=args.eval_object_ix)

            print('[BALD] Selecting...')
            info_gain, amortized_pred_time, ig_compute_time = compute_ig(gnp, context_data, unlabeled_dataset,
                                                                         n_samples_from_latent_dist=32,
                                                                         batching_size=32)

            # gives an opportunity to hijack the selection computation if needed for comparison experiments
            if override_selection_fun is None:
                best_idx = np.argmax(info_gain)
            else:
                best_idx = override_selection_fun(info_gain, tx)

            # Get the observation for the chosen grasp.
            # means, and covariances here.
            print('[BALD] Labeling...')
            grasp_dataset = select_gnp_dataset_ix(unlabeled_dataset, best_idx)
            grasp_dataset, grasp_labeler = get_label_gnp(grasp_dataset, labeler=grasp_labeler)

            context_data = merge_gnp_datasets(context_data, grasp_dataset)
            logger.save_neural_process(gnp, tx + 1, symlink_tx0=True)
            logger.save_acquisition_data(context_data, unlabeled_dataset, tx + 1)

            amortized_pred_times.append(amortized_pred_time)
            ig_compute_times.append(ig_compute_time)
        else:
            raise NotImplementedError()

        # Add datapoint to context dictionary.
    if not grasp_labeler is None:
        grasp_labeler.disconnect()

    if args.constrained:
        agent.disconnect()

    # save time data from run
    with open(logger.get_figure_path('belief_update_times.pkl'), 'wb') as handle:
        pickle.dump(amortized_pred_times, handle)
    with open(logger.get_figure_path('ig_compute_times.pkl'), 'wb') as handle:
        pickle.dump(ig_compute_times, handle)


def particle_filter_loop(pf, object_set, logger, strategy, args,
                         override_selection_fun=None, used_cached_samples=False):
    # NOTE: used_cached_samples is ONLY for when we are cloning experiments,
    # where samples are in the GNP format.
    if used_cached_samples:
        grasp_labeler = None

    if args.likelihood == 'nn':
        logger.save_ensemble(pf.likelihood, 0, symlink_tx0=False)
    elif args.likelihood == 'gnp':
        logger.save_neural_process(pf.likelihood, 0, symlink_tx0=False)
    logger.save_particles(pf.particles, 0)

    particle_update_times, ig_compute_times = [], []
    for tx in range(0, args.max_acquisitions):
        print('[ParticleFilter] Interaction Number', tx)

        # Choose a tower to build that includes the new block.
        if strategy == 'random':
            data_sampler_fn = lambda n: sample_unlabeled_data(n_samples=n, object_set=object_set)
            grasp_dataset = data_sampler_fn(1)
            acquired_sampled_grasps = None
            ig_compute_time = np.NaN
        elif strategy == 'bald':

            data_sampler_fn = lambda n: sample_unlabeled_data(n_samples=n, object_set=object_set)

            # BEGIN TIMING FOR PARTICLE IG STEP ================>
            ig_compute_st = time.process_time()
            if not used_cached_samples:
                all_grasps = []
                all_preds = []
                for ix in range(0, args.n_samples):
                    grasp_data = data_sampler_fn(1)
                    preds = pf.get_particle_likelihoods(pf.particles.particles, grasp_data)
                    all_preds.append(preds)
                    all_grasps.append(grasp_data)
            else:
                _, all_grasps_in_one_dataset = logger.load_acquisition_data(tx)
                all_grasps = explode_dataset_into_list_of_datasets(all_grasps_in_one_dataset)
                all_preds = []
                for grasp_data in all_grasps:
                    preds = pf.get_particle_likelihoods(pf.particles.particles, grasp_data)
                    all_preds.append(preds)

            # NOTE: this is the computation we need for IG for visualization for particle filtering
            pred_vec = torch.Tensor(np.stack(all_preds))
            scores = particle_bald(pred_vec, pf.particles.weights)

            # override for comparison experiments
            if override_selection_fun is None:
                best_ix = np.argmax(scores)
            else:
                best_ix = override_selection_fun(scores, tx)

            # ==============================> END TIMING FOR PARTICLE IG STEP
            ig_compute_et = time.process_time()

            grasp_dataset = all_grasps[best_ix]

            # merge all grasps to save them for visualization
            # convert into gnp form since we are now using the gnp model for the paper
            acquired_sampled_grasps = process_geometry(all_grasps[0], radius=0.03,
                                                       verbose=False) if not used_cached_samples else copy.deepcopy(
                                                                                                        all_grasps[0])
            for grasp in all_grasps[1:]:
                # convert into gnp form since we are now using the gnp model for the paper
                processed_grasp = process_geometry(grasp, radius=0.03,
                                                   verbose=False) if not used_cached_samples else grasp
                acquired_sampled_grasps = merge_gnp_datasets(acquired_sampled_grasps, processed_grasp)

            ig_compute_time = ig_compute_et - ig_compute_st
        else:
            raise NotImplementedError()

        # Get the observation
        if not used_cached_samples:
            grasp_dataset = get_labels(grasp_dataset)
        else:
            # if we are using cached, they are in gnp format - so label them the gnp way
            grasp_dataset, grasp_labeler = get_label_gnp(grasp_dataset, labeler=grasp_labeler)
        # Update the particle belief.

        particles, means = pf.update(grasp_dataset)
        particle_update_time = pf.get_last_update_time()

        print('[ParticleFilter] Particle Statistics')
        print(f'Min Weight: {np.min(pf.particles.weights)}')
        print(f'Max Weight: {np.max(pf.particles.weights)}')
        print(f'Sum Weights: {np.sum(pf.particles.weights)}')

        grasp_dataset = process_geometry(grasp_dataset, radius=0.03,
                                         verbose=False) if not used_cached_samples else grasp_dataset
        if tx == 0:
            context_data = grasp_dataset
        else:
            context_data = merge_gnp_datasets(context_data, grasp_dataset)

        # Save the model and particle distribution at each step.
        if args.likelihood == 'nn':
            logger.save_ensemble(pf.likelihood, tx + 1, symlink_tx0=True)
        elif args.likelihood == 'gnp':
            logger.save_neural_process(pf.likelihood, tx + 1, symlink_tx0=True)
        logger.save_acquisition_data(context_data, acquired_sampled_grasps, tx)
        logger.save_particles(particles, tx + 1)

        particle_update_times.append(particle_update_time)
        ig_compute_times.append(ig_compute_time)

    if used_cached_samples and (not grasp_labeler is None):
        grasp_labeler.disconnect()


    # save time data from run
    with open(logger.get_figure_path('belief_update_times.pkl'), 'wb') as handle:
        pickle.dump(particle_update_times, handle)
    with open(logger.get_figure_path('ig_compute_times.pkl'), 'wb') as handle:
        pickle.dump(ig_compute_times, handle)


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

    # ----- Run particle filter loop -----
    if args.use_progressive_priors:
        amortized_filter_loop(likelihood_model, object_set, logger, args.strategy, args)
    else:
        # ----- Initialize particle filter from prior -----
        if args.likelihood == 'gnp':
            pf = AmortizedGraspingDiscreteLikelihoodParticleBelief(
                object_set=object_set,
                d_latents=d_latents,
                n_particles=args.n_particles,
                likelihood=likelihood_model,
                resample=False,
                plot=False,
                means=args.particle_prop_dist_mean,
                stds=args.particle_prop_dist_stds,
                distribution=args.particle_distribution
            )
        else:
            pf = GraspingDiscreteLikelihoodParticleBelief(
                object_set=object_set,
                d_latents=d_latents,
                n_particles=args.n_particles,
                means=args.particle_prop_dist_mean,
                stds=args.particle_prop_dist_stds,
                distribution=args.particle_distribution,
                likelihood=likelihood_model,
                resample=False,
                plot=False)
        if args.likelihood == 'pb':
            pf.particles = likelihood_model.init_particles(args.n_particles)

        particle_filter_loop(pf, object_set, logger, args.strategy, args)

    return logger.exp_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='',
                        help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--max-acquisitions', type=int, default=25,
                        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--objects-fname', type=str, default='', help='File containing a list of objects to grasp.')
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--pretrained-ensemble-exp-path', type=str, default='', help='Path to a trained ensemble.')
    parser.add_argument('--ensemble-tx', type=int, default=-1, help='Timestep of the trained ensemble to evaluate.')
    parser.add_argument('--eval-object-ix', type=int, default=0, help='Index of which eval object to use.')
    parser.add_argument('--strategy', type=str, choices=['bald', 'random', 'task'], default='bald')
    parser.add_argument('--n-particles', type=int, default=100)
    parser.add_argument('--likelihood', choices=['nn', 'pb', 'gnp'], default='nn')
    parser.add_argument('--constrained', action='store_true', default=False)
    parser.add_argument('--use-progressive-priors', action='store_true', default=False)
    parser.add_argument('--exec-mode', default='sim', choices=['sim', 'real'], help='Collect labels using PyBullet (sim) or the real Panda (real).')
    parser.add_argument('--prop-means', nargs='+', help='means of ind. normal to sample particles from', type=float,
                        default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument('--prop-stds', nargs='+', help='stdevs of ind. normal to sample particles from', type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0])
    parser.add_argument('--prop-distribution', type=str, default='gaussian')
    
    args = parser.parse_args()

    run_particle_filter_fitting(args)
