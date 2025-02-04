import argparse

from matplotlib.pyplot import get
from learning.active import acquire
import torch
import numpy as np
from learning.models import latent_ensemble
from tamp.misc import get_train_and_fit_blocks, load_blocks
from learning.active.utils import ActiveExperimentLogger
from particle_belief import DiscreteLikelihoodParticleBelief
from learning.domains.towers.active_utils import sample_unlabeled_data, get_labels
from learning.active.acquire import bald
# from learning.evaluate.planner import EnsemblePlanner
import sys


def plan_task(pf, block_set, logger, args):
    pass

def find_informative_tower(pf, block_set, logger, args):
    data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set, range_n_blocks=(2, 2), include_index=args.num_train_blocks+args.num_eval_blocks-1, ignore_rot=args.disable_rotations)
    
    all_towers = []
    all_preds = []
    for ix in range(0, args.n_samples):
        tower_dict = data_sampler_fn(1)
        preds = pf.get_particle_likelihoods(pf.particles.particles, tower_dict)
        all_preds.append(preds)
        all_towers.append(tower_dict)
    
    pred_vec = torch.Tensor(np.stack(all_preds))
    scores = bald(pred_vec).cpu().numpy()
    acquire_ix = np.argsort(scores)[::-1][0]
    return all_towers[acquire_ix]

def particle_filter_loop(pf, block_set, logger, strategy, args):
    for tx in range(0, args.max_acquisitions):
        print('[ParticleFilter] Interaction Number', tx)
        
        # Choose a tower to build that includes the new block.
        if strategy == 'random':
            data_sampler_fn = lambda n: sample_unlabeled_data(n, block_set=block_set, range_n_blocks=(2, 2), include_index=args.num_train_blocks+args.num_eval_blocks-1, ignore_rot=args.disable_rotations)
            tower_dict = data_sampler_fn(1)
        elif strategy == 'bald':
            tower_dict = find_informative_tower(pf, block_set, logger, args)
        elif strategy == 'task':
            tower_dict = plan_task(pf, block_set, logger, args)
        else:
            raise NotImplementedError()

        print(tower_dict['2block']['block_ids'])
        # Get the observation for the chosen tower.
        tower_dict = get_labels(tower_dict, 'noisy-model', None, logger, 0.003)

        # Update the particle belief.
        particles, means = pf.update(tower_dict)

        # TODO: Save the model and particle distribution at each step.
        logger.save_ensemble(pf.likelihood, tx)
        logger.save_particles(particles, tx)

def run_particle_filter_fitting(args):
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # ----- Load the block set -----
    assert(len(args.eval_block_ixs) == 1)  # Right now the code only supports doing inference for a single block on top of the training blocks.
    print(args.block_set_fname)
    block_set = get_train_and_fit_blocks(pretrained_ensemble_path=args.pretrained_ensemble_exp_path,
                                         use_latents=True,
                                         fit_blocks_fname=args.block_set_fname,
                                         fit_block_ixs=args.eval_block_ixs)
    print(len(block_set))
    args.num_eval_blocks = len(args.eval_block_ixs)
    args.num_train_blocks = len(block_set) - args.num_eval_blocks

    # ----- Likelihood Model -----
    train_logger = ActiveExperimentLogger(exp_path=args.pretrained_ensemble_exp_path, use_latents=True)
    latent_ensemble = train_logger.get_ensemble(args.ensemble_tx)
    if torch.cuda.is_available():
        latent_ensemble.cuda()
    latent_ensemble.add_latents(1)

    # ----- Initialize particle filter from prior -----
    pf = DiscreteLikelihoodParticleBelief(block=block_set[-1],
                                          D=3,
                                          N=250,
                                          likelihood=latent_ensemble,
                                          plot=True)

    # ----- Run particle filter loop -----
    particle_filter_loop(pf, block_set, logger, args.strategy, args)


def restart_particle_filter(args):
    """
    Restart the particle filter using the given logger until the new max_acquisitions is reached.

    Requires experience to be saved...
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='', help='Where results will be saved. Randon number if not specified.')
    parser.add_argument('--max-acquisitions', type=int, default=25, help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--block-set-fname', type=str, default='', help='File containing a list of AT LEAST 5 blocks (block_utils.Object) where the block.name is formatted obj_#')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--pretrained-ensemble-exp-path', type=str, default='', help='Path to a trained ensemble.')
    parser.add_argument('--ensemble-tx', type=int, default=-1, help='Timestep of the trained ensemble to evaluate.')
    parser.add_argument('--eval-block-ixs', nargs='+', type=int, default=[0], help='Indices of which eval blocks to use.')
    parser.add_argument('--strategy', type=str, choices=['bald', 'random', 'task'], default='bald')
    parser.add_argument('--task', type=str, choices=['overhang', 'tallest', 'min-contact'], default='overhang')
    parser.add_argument('--disable-rotations', action='store_true', default=False)
    parser.add_argument('--restart', action='store_true', default=False)
    args = parser.parse_args()

    if args.restart:
        restart_particle_filter(args)
        sys.exit(0)

    print(args)
    args.use_latents = True
    args.fit_pf = True

    run_particle_filter_fitting(args)
