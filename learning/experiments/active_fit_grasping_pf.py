import argparse
import numpy as np
import torch

from learning.active import acquire
from learning.models import latent_ensemble
from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.active_utils import get_fit_object, sample_unlabeled_data, get_labels, get_train_and_fit_objects
from learning.domains.grasping.pybullet_likelihood import PBLikelihood
from learning.active.acquire import bald
from particle_belief import GraspingDiscreteLikelihoodParticleBelief, AmortizedGraspingDiscreteLikelihoodParticleBelief


def particle_bald(predictions, weights, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    predictions = predictions.cpu().numpy()
    norm = weights.sum()

    mp_c1 = np.sum(weights*predictions, axis=1)/norm
    mp_c0 = 1 - mp_c1

    m_ent = -(mp_c1 * np.log(mp_c1+eps) + mp_c0 * np.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions

    ent_per_model = p_c1 * np.log(p_c1+eps) + p_c0 * np.log(p_c0+eps)
    ent = np.sum(ent_per_model*weights, axis=1)/norm

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


def find_informative_tower_progressive_prior(gnp, current_context, unlabeled_samples):
    """
    :param current_context: Grasp dict object with collected datapoints so far.
        Most-nested dictionaries: ox: [].
    :param unlabeled_samples: The same format ^.
        Most-nested dictionaries: ox: [].
        Labels are meaningless (-1).
    :return: The grasp index of the unlabeled samples with the highest info gain score.
    """
    pass


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
            logger.save_ensemble(pf.likelihood, tx+1, symlink_tx0=True)
        elif args.likelihood == 'gnp':
            logger.save_neural_process(pf.likelihood, tx+1, symlink_tx0=True)
        logger.save_acquisition_data(grasp_dataset, None, tx+1)
        logger.save_particles(particles, tx+1)

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
    particle_filter_loop(pf, object_set, logger, args.strategy, args)

    return logger.exp_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--max-acquisitions', type=int, default=25, help='Number of iterations to run the main active learning loop for.')
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
