from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.active_utils import explode_dataset_into_list_of_datasets, get_train_and_fit_objects
from learning.experiments.active_fit_grasping_pf import amortized_filter_loop, particle_filter_loop
from learning.experiments.grasping_experiment_scripts.run_grasping_experiments import EXPERIMENT_ROOT

import numpy as np
import argparse
import shutil
import json
from pathlib import Path
import pickle
import os

from particle_belief import GraspingDiscreteLikelihoodParticleBelief, AmortizedGraspingDiscreteLikelihoodParticleBelief

LOG_ROOT = 'learning/experiments/logs'


def copy_dir_and_rename(dir_to_copy, dir_new_name):
    dir_to_copy_path = Path(dir_to_copy)
    parent_dir = dir_to_copy_path.parent.absolute()
    copy_to_temp_dir = shutil.copy(dir_to_copy, os.path.join(parent_dir, 'temp'))
    copy_to_temp_dir = shutil.move(copy_to_temp_dir, os.path.join(parent_dir, 'temp', dir_new_name))
    copied_renamed_dir = shutil.move(copy_to_temp_dir, parent_dir)

    return copied_renamed_dir


# override functions must satisfy
# N-long score numpy array X acquisition # -> index to select
# here's an example:
def sample_bad_until_five_and_then_max(scores, tx):
    if tx < 5:
        return np.argmin(scores)
    else:
        return np.argmax(scores)


def main(args):
    # find metadata folder and copy it to metadata with new name
    existing_exp_dir = os.path.join(EXPERIMENT_ROOT, args.existing_exp_name)
    ported_exp_dir = copy_dir_and_rename(existing_exp_dir, args.ported_exp_name)

    # load in both jsons
    with open(os.path.join(existing_exp_dir, 'logs_lookup.json'), 'r') as handle:
        existing_logs_lookup = json.load(handle)

    with open(os.path.join(ported_exp_dir, 'logs_lookup.json'), 'r') as handle:
        ported_logs_lookup = json.load(handle)

    # delete fitting data content in ported json
    for strategy in ported_logs_lookup['fitting_phase'].keys():
        ported_logs_lookup['fitting_phase'][strategy] = {}

    # look up all the data in existing_log_json, copy files over, and store their paths in ported_logs_json
    # (for each strategy)
    for obj_ix in args.train_objs_to_port:
        for strategy in existing_logs_lookup['fitting_phase'].keys():
            existing_fit_name = f'grasp_{args.existing_exp_name}_fit_{strategy}_train_geo_object{obj_ix}'

            # if we have not fit that object yet, just pass over to the next one
            try:
                existing_fit_dir = existing_logs_lookup['fitting_phase'][strategy][existing_fit_name]
            except KeyError:
                print('[Info:] Did not find object %i in %s of exp %s' % (obj_ix, strategy, args.existing_exp_name))
                continue

            # copy the file over
            ported_fit_name = f'grasp_{args.ported_exp_name}_fit_{strategy}_train_geo_object{obj_ix}'
            ported_fit_dir = copy_dir_and_rename(existing_fit_dir, ported_fit_name)

            # modify the fitting args
            fitting_args_path = os.path.join(ported_fit_dir, 'args.pkl')
            with open(fitting_args_path, 'rb') as handle:
                fitting_args = pickle.load(handle)

            # keep note for post-processing if so
            existing_used_progressive_priors = fitting_args.use_progressive_priors

            fitting_args.exp_name = ported_fit_name
            fitting_args.use_progressive_priors = args.belief == 'progressive'

            with open(fitting_args_path, 'w') as handle:
                pickle.dump(fitting_args, handle)

            ported_logger = ActiveExperimentLogger(exp_path=ported_fit_dir, use_latents=True)
            object_set = get_train_and_fit_objects(
                    pretrained_ensemble_path=fitting_args.pretrained_ensemble_exp_path,
                    use_latents=True,
                    fit_objects_fname=args.objects_fname,
                    fit_object_ix=args.eval_object_ix
                )
            # store in ported_logs_lookup
            ported_logs_lookup['fitting_phase'][strategy][ported_fit_name] = ported_fit_dir

            if args.override_with is not None:

                # choose the override fun
                if args.override_with == 'example':
                    override_fun = sample_bad_until_five_and_then_max
                else:
                    print('[ERROR]: did not specify a valid override fun. skipping...')
                    continue

                # re-run the fitting loop with overrided fun
                if existing_used_progressive_priors:
                    amortized_filter_loop(ported_logger.get_neural_process(0), object_set, ported_logger, 'bald',
                        fitting_args, override_selection_fun=override_fun)
                else:
                    gnp = ported_logger.get_neural_process(0)
                    pf = AmortizedGraspingDiscreteLikelihoodParticleBelief(
                        object_set=object_set,
                        d_latents=gnp.d_latents,
                        n_particles=fitting_args.n_particles,
                        likelihood=gnp,
                        resample=False,
                        plot=False
                    )
                    particle_filter_loop(pf, object_set, ported_logger, 'bald', fitting_args,
                                         override_selection_fun=override_fun)
            # if this is a simple port of a progressive posterior -> particles, then regenerate the particles
            elif args.belief == 'particle' and existing_used_progressive_priors:
                # now we need to regenerate the particles.
                # since context sets are ordered, let's pull the last one and just step through it
                context_set, _ = ported_logger.load_acquisition_data(fitting_args.max_acquisitions - 1)
                context_set_individual_grasps = explode_dataset_into_list_of_datasets(context_set)

                with open(os.path.join(ported_logs_lookup['training_phase'], 'args.pkl'), 'rb') as handle:
                    training_args = pickle.load(handle)

                # think critically about what object_set looksl like
                pf = AmortizedGraspingDiscreteLikelihoodParticleBelief(
                    object_set=object_set,
                    d_latents=training_args.d_latents,
                    n_particles=fitting_args.n_particles,
                    likelihood=ported_logger.get_neural_process(0),
                    resample=False,
                    plot=False
                )

                for tx, grasp in enumerate(context_set_individual_grasps):
                    particles, _ = pf.update(grasp)
                    ported_logger.save_particles(particles, tx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-exp-name', type=str, required=True, help='name of existing experiment to port')
    parser.add_argument('--ported-exp-name', type=str, required=True, help='name of ported experiment')
    parser.add_argument('--belief', type=str, choices=['particle', 'progressive'], required=True)
    parser.add_argument('--train-objs-to-port', type=int, nargs='*', default=[])
    parser.add_argument('--test-objs-to-port', type=int, nargs='*', default=[])
    parser.add_argument('--override-with', type=str, choice=['example'],  help='override with selected fun')

    args = parser.parse_args()

    if not args.train_objs_to_port and not args.test_objs_to_port:
        print('Must specify at least one train or test object to port over with experiment.')
    else:
        main(args)
