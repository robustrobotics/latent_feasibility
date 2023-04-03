import argparse
import json
import pandas as pd
import numpy as np
import os
import pickle
import sys

from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector

from learning.active.utils import ActiveExperimentLogger
from learning.evaluate.evaluate_grasping import get_pf_validation_accuracy, get_pf_task_performance
from learning.evaluate.plot_compare_grasping_runs import plot_val_loss, plot_from_dataframe
from learning.experiments.train_grasping_single import run as training_phase_variational
from learning.domains.grasping.generate_datasets_for_experiment import parse_ignore_file
from learning.models.grasp_np.train_grasp_np import run as training_phase_amortized
from learning.experiments.active_fit_grasping_pf import run_particle_filter_fitting as fitting_phase
from learning.experiments.active_fit_constrained_grasping_pf import \
    run_particle_filter_fitting as constrained_fitting_phase

DATA_ROOT = 'learning/data/grasping'
EXPERIMENT_ROOT = 'learning/experiments/metadata'


def get_dataset_path_or_fail(args):
    if len(args.dataset_name) == 0:
        print(f'[ERROR] You must specify a dataset.')
        sys.exit()
    dataset_dir = os.path.join(DATA_ROOT, args.dataset_name)
    if not os.path.exists(dataset_dir):
        print(f'[ERROR] Dataset does not exist: {args.dataset_name}')
        sys.exit()
    return dataset_dir


def create_experiment(args):
    exp_dir = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if os.path.exists(exp_dir):
        print(f'[ERROR] Folder already exists: {exp_dir}')
        sys.exit()

    dataset_dir = get_dataset_path_or_fail(args)
    os.makedirs(exp_dir)

    args_path = os.path.join(exp_dir, 'args.pkl')
    with open(args_path, 'wb') as handle:
        pickle.dump(args, handle)

    logs_path = os.path.join(exp_dir, 'logs_lookup.json')
    logs = {
        'training_phase': '',
        'fitting_phase': {'random': {}, 'bald': {}, 'constrained_random': {}, 'constrained_bald': {}}
    }
    with open(logs_path, 'w') as handle:
        json.dump(logs, handle)

    log_groups_path = os.path.join(exp_dir, 'log_groups')
    os.makedirs(log_groups_path)


def get_training_phase_dataset_args(dataset_fname):
    data_path = os.path.join(DATA_ROOT, dataset_fname)

    train_path = os.path.join(data_path, 'grasps', 'training_phase', 'train_grasps.pkl')
    val_path = os.path.join(data_path, 'grasps', 'training_phase', 'val_grasps.pkl')

    with open(train_path, 'rb') as handle:
        train_grasps = pickle.load(handle)
    n_objects = len(train_grasps['object_data']['object_names'])

    return train_path, val_path, n_objects


def get_fitting_phase_dataset_args(dataset_fname):
    data_path = os.path.join(DATA_ROOT, dataset_fname)

    train_geo_fname = os.path.join(data_path, 'objects', 'train_geo_test_props.pkl')
    test_geo_fname = os.path.join(data_path, 'objects', 'test_geo_test_props.pkl')

    with open(train_geo_fname, 'rb') as handle:
        train_geo_objects = pickle.load(handle)
    n_train_geo = len(train_geo_objects['object_data']['object_names'])

    with open(test_geo_fname, 'rb') as handle:
        test_geo_objects = pickle.load(handle)
    n_test_geo = len(test_geo_objects['object_data']['object_names'])

    return train_geo_fname, test_geo_fname, n_train_geo, n_test_geo


def run_fitting_phase(args):
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    pretrained_model_path = logs_lookup['training_phase']
    if len(pretrained_model_path) == 0:
        print(f'[ERROR] Training phase has not yet been executed.')
        sys.exit()

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    train_geo_fname, test_geo_fname, n_train_geo, n_test_geo = \
        get_fitting_phase_dataset_args(exp_args.dataset_name)

    ignore_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'ignore.txt')
    TRAIN_IGNORE, TEST_IGNORE = parse_ignore_file(ignore_fname)

    # Run fitting phase for all objects that have not yet been fitted
    # (each has a standard name in the experiment logs).
    for geo_type, objects_fname, n_objects, ignore in zip(
            ['test_geo', 'train_geo'],
            [test_geo_fname, train_geo_fname],
            [n_test_geo, n_train_geo],
            [TEST_IGNORE, TRAIN_IGNORE]
    ):
        # Get object data.
        with open(objects_fname, 'rb') as handle:
            fit_objects = pickle.load(handle)

        min_pstable, max_pstable, min_dist = 0.0, 1.0, 0.0
        valid_fit_objects, _, _, _, _, _, _ = filter_objects(
            object_names=fit_objects['object_data']['object_names'],
            ignore_list=ignore,
            phase=geo_type.split('_')[0],
            dataset_name=exp_args.dataset_name,
            min_pstable=min_pstable,
            max_pstable=max_pstable,
            min_dist_threshold=min_dist,
            max_objects=500
        )
        print(f'Total: {len(valid_fit_objects)} to fit for {geo_type}.')

        for (ox, _, _) in valid_fit_objects:
            # Some geometries have trouble when considering IK (e.g., always close to table).
            # TODO: Make this more modular when we use constraints again.
            # if args.constrained and geo_type == 'test_geo' and ox in [15, 16, 17, 18, 19]:
            #     continue
            # if args.constrained and geo_type == 'train_geo' and (ox >= 85 and ox < 90):
            #     continue
            if args.constrained:
                mode = f'constrained_{args.strategy}'
            else:
                mode = f'{args.strategy}'

            if mode not in logs_lookup['fitting_phase']:
                logs_lookup['fitting_phase'][mode] = {}
            fitting_exp_name = f'grasp_{exp_args.exp_name}_fit_{mode}_{geo_type}_object{ox}'

            # Check if we have already fit this object.
            if fitting_exp_name in logs_lookup['fitting_phase'][mode]:
                print(f'Skipping {fitting_exp_name}...')
                continue

            fitting_args = argparse.Namespace()
            fitting_args.exp_name = fitting_exp_name
            fitting_args.max_acquisitions = 25
            fitting_args.objects_fname = objects_fname
            fitting_args.n_samples = 20
            fitting_args.pretrained_ensemble_exp_path = pretrained_model_path
            fitting_args.ensemble_tx = 0
            fitting_args.eval_object_ix = ox
            fitting_args.strategy = args.strategy
            fitting_args.n_particles = 1000
            fitting_args.use_progressive_priors = True
            fitting_args.constrained = args.constrained
            if args.amortize:
                fitting_args.likelihood = 'gnp'
            else:
                fitting_args.likelihood = 'nn'

            print(f'Running fitting phase: {fitting_exp_name}')
            fit_log_path = fitting_phase(fitting_args)

            # Save fitting path in metadata.
            with open(logs_path, 'r') as handle:
                logs_lookup = json.load(handle)

            # if len(logs_lookup['fitting_phase']) == 0:
            #     logs_lookup['fitting_phase'] = {'random': {}, 'bald': {}, 'constrained_random': {},
            #                                     'constrained_bald': {}}
            logs_lookup['fitting_phase'][mode][fitting_exp_name] = fit_log_path
            with open(logs_path, 'w') as handle:
                json.dump(logs_lookup, handle)

            # Run accuracy evaluations for this object.
            print(f'Evaluating fitting phase: {fitting_exp_name}')
            fit_logger = ActiveExperimentLogger(fit_log_path, use_latents=True)
            val_dataset_fname = f'fit_grasps_{geo_type}_object{ox}.pkl'
            val_dataset_path = os.path.join(
                DATA_ROOT, exp_args.dataset_name,
                'grasps', 'fitting_phase', val_dataset_fname
            )

            get_pf_validation_accuracy(
                fit_logger,
                val_dataset_path,
                args.amortize,
                use_progressive_priors=fitting_args.use_progressive_priors,
                vis=False
            )


def playback_fitting_phase(args):
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    ignore_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'ignore.txt')
    TRAIN_IGNORE, TEST_IGNORE = parse_ignore_file(ignore_fname)

    # Get object data.
    train_objects_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'objects', 'train_geo_test_props.pkl')
    with open(train_objects_fname, 'rb') as handle:
        train_objects = pickle.load(handle)
    test_objects_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'objects', 'test_geo_test_props.pkl')
    with open(os.path.join(DATA_ROOT, exp_args.dataset_name, 'args.pkl'), 'rb') as handle:
        dataset_args = pickle.load(handle)
    n_property_samples_train = dataset_args.n_property_samples_train
    n_property_samples_test = dataset_args.n_property_samples_test

    with open(test_objects_fname, 'rb') as handle:
        test_objects = pickle.load(handle)

    with open(os.path.join(logs_lookup['training_phase'], 'args.pkl'), 'rb') as handle:
        training_args = pickle.load(handle)
    n_latents = training_args.d_latents

    logs_lookup_by_object, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = gather_experiment_logs_file_paths(
        TEST_IGNORE, TRAIN_IGNORE, args, exp_args, logs_lookup, test_objects, train_objects)

    for obj_ix in args.vis_train_objects:
        # work out correct log file, prop sample #, and proper log path for logger
        object_name = train_objects['object_data']['object_names'][obj_ix]
        no_prop_sample = obj_ix % n_property_samples_train
        log_path = logs_lookup_by_object['train_geo'][args.strategy][object_name][no_prop_sample]
        logger = ActiveExperimentLogger(exp_path=log_path, use_latents=True)

        # get args for experiment playback
        with open(os.path.join(log_path, 'args.pkl'), 'rb') as handle:
            fitting_args = pickle.load(handle)

        object_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase',
                                           f'fit_grasps_train_geo_object{obj_ix}.pkl')
        get_pf_validation_accuracy(
            logger,
            object_dataset_path,
            fitting_args.likelihood == 'gnp',
            use_progressive_priors=fitting_args.use_progressive_priors,
            vis=True
        )

    for obj_ix in args.vis_test_objects:
        # work out correct log file, prop sample #, and proper log path for logger
        object_name = test_objects['object_data']['object_names'][obj_ix]
        no_prop_sample = obj_ix % n_property_samples_test
        log_path = logs_lookup_by_object['test_geo'][args.strategy][object_name][no_prop_sample]
        logger = ActiveExperimentLogger(exp_path=log_path, use_latents=True)

        # get args for experiment playback
        with open(os.path.join(log_path, 'args.pkl'), 'rb') as handle:
            fitting_args = pickle.load(handle)

        object_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase',
                                           f'fit_grasps_test_geo_object{obj_ix}.pkl')
        get_pf_validation_accuracy(
            logger,
            object_dataset_path,
            fitting_args.likelihood == 'gnp',
            use_progressive_priors=fitting_args.use_progressive_priors,
            vis=True
        )

    if args.playback_train_fitting:
        for obj_ix, object_name in enumerate(train_objects['object_data']['object_names']):
            # work out correct log file, prop sample #, and proper log path for logger
            no_prop_sample = obj_ix % n_property_samples_train

            try:
                print('replaying: ' + object_name)
                log_path = logs_lookup_by_object['train_geo'][args.strategy][object_name][no_prop_sample]
                logger = ActiveExperimentLogger(exp_path=log_path, use_latents=True)
            except KeyError:
                print('no log found for: ' + object_name)
                continue

            # get args for experiment playback
            with open(os.path.join(log_path, 'args.pkl'), 'rb') as handle:
                fitting_args = pickle.load(handle)

            object_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase',
                                               f'fit_grasps_train_geo_object{obj_ix}.pkl')
            get_pf_validation_accuracy(
                logger,
                object_dataset_path,
                fitting_args.likelihood == 'gnp',
                use_progressive_priors=fitting_args.use_progressive_priors,
                vis=False
            )
    if args.playback_test_fitting:
        for obj_ix, object_name in enumerate(test_objects['object_data']['object_names']):
            # work out correct log file, prop sample #, and proper log path for logger
            no_prop_sample = obj_ix % n_property_samples_test

            try:
                log_path = logs_lookup_by_object['test_geo'][args.strategy][object_name][no_prop_sample]
                logger = ActiveExperimentLogger(exp_path=log_path, use_latents=True)
            except KeyError:
                print('no log found for: ' + object_name)
                continue

            # get args for experiment playback
            with open(os.path.join(log_path, 'args.pkl'), 'rb') as handle:
                fitting_args = pickle.load(handle)

            object_dataset_path = os.path.join(DATA_ROOT, exp_args.dataset_name, 'grasps', 'fitting_phase',
                                               f'fit_grasps_test_geo_object{obj_ix}.pkl')
            get_pf_validation_accuracy(
                logger,
                object_dataset_path,
                fitting_args.likelihood == 'gnp',
                use_progressive_priors=fitting_args.use_progressive_priors,
                vis=False
            )


def run_task_eval_phase(args):
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    pretrained_model_path = logs_lookup['training_phase']
    if len(pretrained_model_path) == 0:
        print(f'[ERROR] Training phase has not yet been executed.')
        sys.exit()

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    train_geo_fname, test_geo_fname, n_train_geo, n_test_geo = \
        get_fitting_phase_dataset_args(exp_args.dataset_name)

    ignore_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'ignore.txt')
    TRAIN_IGNORE, TEST_IGNORE = parse_ignore_file(ignore_fname)

    # Run fitting phase for all objects that have not yet been fitted
    # (each has a standard name in the experiment logs).
    for geo_type, objects_fname, n_objects, ignore in zip(
            ['test_geo', 'train_geo'],
            [test_geo_fname, train_geo_fname],
            [n_test_geo, n_train_geo],
            [TEST_IGNORE, TRAIN_IGNORE]
    ):
        for ox in range(min(n_objects, 100)):
            if ox in ignore: continue

            if args.constrained:
                mode = f'constrained_{args.strategy}'
            else:
                mode = f'{args.strategy}'

            if mode not in logs_lookup['fitting_phase']:
                logs_lookup['fitting_phase'][mode] = {}
            fitting_exp_name = f'grasp_{exp_args.exp_name}_fit_{mode}_{geo_type}_object{ox}'

            # Check if we have already fit this object.
            if fitting_exp_name not in logs_lookup['fitting_phase'][mode]:
                print(f'Skipping {fitting_exp_name}... (not fitted)')
                continue

            fit_log_path = logs_lookup['fitting_phase'][mode][fitting_exp_name]

            # Run accuracy evaluations for this object.
            print(f'Evaluating min-force grasping: {fitting_exp_name}')
            fit_logger = ActiveExperimentLogger(fit_log_path, use_latents=True)
            val_dataset_fname = f'fit_grasps_{geo_type}_object{ox}.pkl'
            val_dataset_path = os.path.join(
                DATA_ROOT, exp_args.dataset_name,
                'grasps', 'fitting_phase', val_dataset_fname
            )

            with open(os.path.join(fit_log_path, 'args.pkl'), 'rb') as handle:
                fitting_args = pickle.load(handle)
            get_pf_task_performance(fit_logger, val_dataset_path,
                                    use_progressive_priors=fitting_args.use_progressive_priors)


def run_training_phase(args):
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)
    if len(logs_lookup['training_phase']) > 0:
        print('[ERROR] Model already trained.')
        sys.exit()

    train_data_fname, val_data_fname, n_objs = \
        get_training_phase_dataset_args(exp_args.dataset_name)

    if args.amortize:
        training_args = argparse.Namespace()
        training_args.exp_name = f'grasp_{exp_args.exp_name}_train'
        training_args.train_dataset_fname = train_data_fname
        training_args.val_dataset_fname = val_data_fname
        training_args.n_epochs = 100
        training_args.d_latents = 2  # TODO: fix latent dimension magic number elsewhere?
        training_args.batch_size = 16
        training_args.use_latents = False  # NOTE: this is a workaround for pointnet + latents,
        # GNPs DO USE LATENTS, but they are handled more
        # cleanly in the model specification and training
        training_args.informed_prior_loss = True
        training_args.use_local_grasp_geometry = True
        training_args.add_mesh_normals = False
        training_args.add_mesh_curvatures = False

        train_log_path = training_phase_amortized(training_args)

    else:
        training_args = argparse.Namespace()
        training_args.exp_name = f'grasp_{exp_args.exp_name}_train'
        training_args.train_dataset_fname = train_data_fname
        training_args.val_dataset_fname = val_data_fname
        training_args.n_objects = n_objs
        training_args.n_epochs = 20
        training_args.model = 'pn'
        training_args.n_hidden = 64
        training_args.batch_size = 32
        training_args.property_repr = 'latent'
        training_args.n_models = 5

        train_log_path = training_phase_variational(training_args)

    # Save training path in metadata.
    logs_lookup['training_phase'] = train_log_path
    with open(logs_path, 'w') as handle:
        json.dump(logs_lookup, handle)


def filter_objects(object_names, ignore_list, phase, dataset_name, min_pstable, max_pstable, min_dist_threshold,
                   max_objects):
    """
    :param object_names: All potential objects to consider.
    :param ignore_list: List of objects that are ungraspable.
    :param phase: train or test.

    Return list of (ox, object_name) for all valid objects.
    """
    valid_objects = []
    valid_pstables = []
    valid_min_dists = []
    valid_ratios = []
    valid_maxdims = []
    valid_rrates = []
    valid_eigeval_prod = []

    with open(f'{os.environ["SHAPENET_ROOT"]}/object_infos.pkl', 'rb') as handle:
        metadata = pickle.load(handle)

    for ox, object_name in enumerate(object_names):
        if ox in ignore_list:
            continue
        val_dataset_fname = f'fit_grasps_{phase}_geo_object{ox}.pkl'
        val_dataset_path = os.path.join(
            DATA_ROOT, dataset_name,
            'grasps', 'fitting_phase', val_dataset_fname
        )
        if not os.path.exists(val_dataset_path):
            print('Does not exist!', val_dataset_path)
            continue
        with open(val_dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        name = data['object_data']['object_names'][ox]
        props = data['object_data']['object_properties'][ox]
        prop_str = ''
        for p in props:
            prop_str += '%.3f' % p + '_'
        prop_str = prop_str[:-1]
        # graspable_body = graspablebody_from_vector(name, props)
        # sim_client = GraspSimulationClient(graspable_body=graspable_body,
        #     show_pybullet=False,
        #     recompute_inertia=True)
        # mesh = sim_client.mesh

        # compute the sample covariance
        # leverage the fact that these are in a continuous block:
        sample_covar = np.zeros((props.size, props.size))
        name_ixs = np.where(np.array(data['object_data']['object_names']) == name)[0]
        for name_ix in name_ixs:
            prop_sample = data['object_data']['object_properties'][name_ix]
            sample_covar += np.outer(prop_sample, prop_sample)
        sample_covar /= len(name_ixs)
        # TODO: note that this is a hack right now since we're dealing with 2d mean and covariance
        eigval_prod = np.real(np.prod(np.linalg.eigvals(sample_covar)[0:2]))  # we know sample covar PSD so eigval real

        volume = metadata[name]['volume']
        bb_volume = metadata[name]['bb_volume']
        # max_dim = np.max(mesh.bounds[1]*2)
        ratio = bb_volume / volume
        # sim_client.disconnect()
        max_dim = 0.2
        rejection_rate = metadata[name]['rejection_rate']

        # all_midpoints = np.array(list(data['grasp_data']['grasp_midpoints'].values())[0])[:50]
        # dists_to_closest = []
        # for gx, midpoint in enumerate(all_midpoints):
        #     other_points = np.concatenate(
        #         [all_midpoints[:gx, :], all_midpoints[gx + 1:, :]],
        #         axis=0
        #     )
        #     dists = np.linalg.norm(midpoint - other_points, axis=1)
        #     dists_to_closest.append(np.min(dists))

        # avg_min_dist = np.mean(dists_to_closest)
        avg_min_dist = metadata[name]['avg_min_dist50']
        p_stable = np.mean(list(data['grasp_data']['labels'].values())[0])
        if avg_min_dist < min_dist_threshold:
            continue
        if p_stable < min_pstable or p_stable > max_pstable:
            continue

        valid_objects.append((ox, object_name, prop_str))
        valid_pstables.append(p_stable)
        valid_min_dists.append(avg_min_dist)
        valid_ratios.append(ratio)
        valid_maxdims.append(max_dim)
        valid_rrates.append(rejection_rate)
        valid_eigeval_prod.append(eigval_prod)
        print(f'{object_name} in range ({min_pstable}, {max_pstable}) ({p_stable})')

    return valid_objects[:max_objects], \
        valid_pstables[:max_objects], \
        valid_min_dists[:max_objects], \
        valid_ratios[:max_objects], \
        valid_maxdims[:max_objects], \
        valid_rrates[:max_objects], \
        valid_eigeval_prod[:max_objects]


def run_testing_phase(args):
    # Create log_group files.
    exp_path = os.path.join(EXPERIMENT_ROOT, args.exp_name)
    if not os.path.exists(exp_path):
        print(f'[ERROR] Experiment does not exist: {args.exp_name}')
        sys.exit()

    logs_path = os.path.join(exp_path, 'logs_lookup.json')
    with open(logs_path, 'r') as handle:
        logs_lookup = json.load(handle)

    # Get train_geo_test_props.pkl and test_geo_test_props.pkl
    args_path = os.path.join(exp_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        exp_args = pickle.load(handle)

    ignore_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'ignore.txt')
    TRAIN_IGNORE, TEST_IGNORE = parse_ignore_file(ignore_fname)

    # Get object data.
    train_objects_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'objects', 'train_geo_test_props.pkl')
    with open(train_objects_fname, 'rb') as handle:
        train_objects = pickle.load(handle)
    test_objects_fname = os.path.join(DATA_ROOT, exp_args.dataset_name, 'objects', 'test_geo_test_props.pkl')
    with open(os.path.join(DATA_ROOT, exp_args.dataset_name, 'args.pkl'), 'rb') as handle:
        dataset_args = pickle.load(handle)
    n_property_samples_train = dataset_args.n_property_samples_train
    n_property_samples_test = dataset_args.n_property_samples_test

    with open(test_objects_fname, 'rb') as handle:
        test_objects = pickle.load(handle)

    with open(os.path.join(logs_lookup['training_phase'], 'args.pkl'), 'rb') as handle:
        training_args = pickle.load(handle)
    n_latents = training_args.d_latents

    logs_lookup_by_object, \
        valid_test_min_dists, \
        valid_test_objects, \
        valid_test_pstables, \
        valid_test_ratios, \
        valid_test_maxdims, \
        valid_test_rrates, \
        valid_test_eigval_prod, \
        valid_train_min_dists, \
        valid_train_objects, \
        valid_train_pstables, \
        valid_train_ratios, \
        valid_train_maxdims, \
        valid_train_rrates, \
        valid_train_eigval_prod = gather_experiment_logs_file_paths(
        TEST_IGNORE, TRAIN_IGNORE, args, exp_args, logs_lookup, test_objects, train_objects)

    # collect all of the stored metric data over the course of training
    # note; we don't do confusions since they are hard to store... and that data can come from the other
    # metrics below
    accuracies = {'train_geo': {}, 'test_geo': {}}
    precisions = {'train_geo': {}, 'test_geo': {}}
    average_precisions = {'train_geo': {}, 'test_geo': {}}
    recalls = {'train_geo': {}, 'test_geo': {}}
    f1s = {'train_geo': {}, 'test_geo': {}}
    balanced_accuracy_scores = {'train_geo': {}, 'test_geo': {}}
    entropies = {'train_geo': {}, 'test_geo': {}}
    means = {'train_geo': {}, 'test_geo': {}}
    covars = {'train_geo': {}, 'test_geo': {}}
    if args.amortize:
        metric_list = [accuracies, precisions, average_precisions, recalls, f1s, balanced_accuracy_scores, entropies]
        metric_file_list = ['val_accuracies.pkl', 'val_precisions.pkl', 'val_average_precisions.pkl', 'val_recalls.pkl',
                            'val_f1s.pkl', 'val_balanced_accs.pkl', 'val_entropies.pkl']
        metric_names = ['accuracy', 'precision', 'average precision', 'recall', 'f1', 'balanced accuracy', 'entropy']
    else:
        metric_list = [accuracies, precisions, average_precisions, recalls, f1s, balanced_accuracy_scores]
        metric_file_list = ['val_accuracies.pkl', 'val_precisions.pkl', 'val_average_precisions.pkl', 'val_recalls.pkl',
                            'val_f1s.pkl', 'val_balanced_accs.pkl']
        metric_names = ['accuracy', 'precision', 'average precision', 'recall', 'f1', 'balanced accuracy']
    log_paths_set = {'train_geo': [], 'test_geo': []}
    n_acquisitions = None
    for obj_set in ['train_geo', 'test_geo']:
        for strategy in logs_lookup_by_object[obj_set].keys():
            log_paths = logs_lookup_by_object[obj_set][strategy]['all']
            log_paths_set[obj_set] += log_paths
            n_objs = len(log_paths)
            if len(log_paths) <= 0:
                continue
            with open(os.path.join(log_paths[0], 'args.pkl'), 'rb') as handle:
                fit_args = pickle.load(handle)
            n_acquisitions = fit_args.max_acquisitions
            acc, prec, avg_prec, recalls, f1s, bal_acc, etrpy = \
                np.zeros((n_objs, n_acquisitions)), \
                    np.zeros((n_objs, n_acquisitions)), \
                    np.zeros((n_objs, n_acquisitions)), \
                    np.zeros((n_objs, n_acquisitions)), \
                    np.zeros((n_objs, n_acquisitions)), \
                    np.zeros((n_objs, n_acquisitions)), \
                    np.zeros((n_objs, n_acquisitions))

            if args.amortize:
                metric_per_strategy_list = [acc, prec, avg_prec, recalls, f1s, bal_acc, etrpy]
            else:
                metric_per_strategy_list = [acc, prec, avg_prec, recalls, f1s, bal_acc]

            mn, cvr = np.zeros((n_objs, n_latents, n_acquisitions)), np.zeros((n_objs, n_latents, n_acquisitions))

            for i_obj, log_path in enumerate(log_paths):
                logger = ActiveExperimentLogger(exp_path=log_path, use_latents=True)

                for data_arr, fname in zip(metric_per_strategy_list, metric_file_list):
                    with open(logger.get_figure_path(fname), 'rb') as handle:
                        data_arr[i_obj, :] = np.array(pickle.load(handle)).squeeze()

                if args.amortize:
                    for data_arr, fname in zip([mn, cvr], ['val_means.pkl', 'val_covars.pkl']):
                        with open(logger.get_figure_path(fname), 'rb') as handle:
                            data_arr[i_obj, :, :] = np.array(pickle.load(handle)).squeeze().T

            for metric, metric_per_strategy in zip(metric_list, metric_per_strategy_list):
                metric[obj_set][strategy] = metric_per_strategy

            if args.amortize:
                means[obj_set][strategy] = mn
                covars[obj_set][strategy] = cvr

    # we now have all the data we need to construct the full dataframe
    # we first construct are two dataframes: one for the time-dependent metrics
    # one to store p_stability, p_size, and also the fitting directory so we can associate grasp selection later

    # construct hierarchical indices (we'll reuse them for both dataframes, and construct one for train and test
    # separately since they may not share the same strategies used)
    unique_object_names_train = list(dict.fromkeys(
        [name for _, name, _ in valid_train_objects]
    ))
    strategies_used_for_train = metric_list[0]['train_geo'].keys()

    mi_train = pd.MultiIndex.from_product([
        strategies_used_for_train,
        unique_object_names_train,
        range(n_property_samples_train)
    ], names=['strategy', 'name', 'no_property_sample'])

    unique_object_names_test = list(dict.fromkeys(
        [name for _, name, _ in valid_test_objects]
    ))
    strategies_used_for_test = metric_list[0]['test_geo'].keys()

    mi_test = pd.MultiIndex.from_product([
        strategies_used_for_test,
        unique_object_names_test,
        range(n_property_samples_test)
    ], names=['strategy', 'name', 'no_property_sample'])

    # construct non-time series data
    d_const_train = pd.DataFrame(data=zip(
        valid_train_eigval_prod * len(strategies_used_for_train),
        valid_train_pstables * len(strategies_used_for_train),
        valid_train_min_dists * len(strategies_used_for_train),
        valid_train_ratios * len(strategies_used_for_train),
        valid_train_maxdims * len(strategies_used_for_train),
        valid_train_rrates * len(strategies_used_for_train),
        [obj[1] for obj in valid_train_objects] * len(strategies_used_for_train),
        [obj[2] for obj in valid_train_objects] * len(strategies_used_for_train),
        log_paths_set['train_geo']
    ), index=mi_train,
        columns=['eigval_prod', 'pstable', 'avg_min_dist', 'ratio', 'maxdim', 'rrate', 'name', 'props', 'log_paths'])

    d_const_test = pd.DataFrame(data=zip(
        valid_test_eigval_prod * len(strategies_used_for_test),
        valid_test_pstables * len(strategies_used_for_test),
        valid_test_min_dists * len(strategies_used_for_test),
        valid_test_ratios * len(strategies_used_for_test),
        valid_test_maxdims * len(strategies_used_for_test),
        valid_test_rrates * len(strategies_used_for_test),
        [obj[1] for obj in valid_test_objects] * len(strategies_used_for_test),
        [obj[2] for obj in valid_test_objects] * len(strategies_used_for_test),
        log_paths_set['test_geo']
    ), index=mi_test,
        columns=['eigval_prod', 'pstable', 'avg_min_dist', 'ratio', 'maxdim', 'rrate', 'name', 'props', 'log_paths'])
    # d_const_train = d_const_test

    # construct multi-index for columns in time-series data
    mc_time = pd.MultiIndex.from_product([metric_names, range(n_acquisitions)], names=['time metric', 'acquisition'])
    # formatted_metrics_train = np.hstack([
    #     np.vstack(list(metric['train_geo'].values())) for metric in metric_list
    # ])
    formatted_metrics_train = np.hstack([
        np.vstack([
            metric['train_geo'][strat] for strat in strategies_used_for_train
        ]) for metric in metric_list
    ])
    d_time_train = pd.DataFrame(data=formatted_metrics_train, index=mi_train, columns=mc_time)

    # formatted_metrics_test = np.hstack([
    #     np.vstack(list(metric['test_geo'].values())) for metric in metric_list
    # ])
    formatted_metrics_test = np.hstack([
        np.vstack([
            metric['test_geo'][strat] for strat in strategies_used_for_test
        ]) for metric in metric_list
    ])
    d_time_test = pd.DataFrame(data=formatted_metrics_test, index=mi_test, columns=mc_time)

    # concat const frames and time frames together and melt into long form so we can merge the tables
    d_const = pd.concat([d_const_train, d_const_test], keys=['train', 'test'])
    d_time = pd.concat([d_time_train, d_time_test], keys=['train', 'test']).melt(
        value_name='time metric value', ignore_index=False)
    d_all = pd.merge(d_const, d_time, left_index=True, right_index=True)

    # if we are amortizing, then we're tracking covariances + means, include in computation
    if args.amortize:
        # construct multi-index for columns in time-series data per latent property
        mc_ltime = pd.MultiIndex.from_product([['mean', 'covar'], range(n_latents), range(n_acquisitions)],
                                              names=['latent parameter', 'latent', 'acquisition'])
        formatted_latent_metrics_train = np.hstack([
            # stack strategy data on top of each other
            np.vstack([
                lmps.reshape((lmps.shape[0], lmps.shape[1] * lmps.shape[2]), order='C')
                for lmps in latent_metric['train_geo'].values()
            ])
            # stack mean and covars side by side
            for latent_metric in [means, covars]
        ])
        d_latent_time_train = pd.DataFrame(data=formatted_latent_metrics_train, index=mi_train, columns=mc_ltime)

        formatted_latent_metrics_test = np.hstack([
            # stack strategy data on top of each other
            np.vstack([
                lmps.reshape((lmps.shape[0], lmps.shape[1] * lmps.shape[2]), order='C')
                for lmps in latent_metric['test_geo'].values()
            ])
            # stack mean and covars side by side
            for latent_metric in [means, covars]
        ])
        d_latent_time_test = pd.DataFrame(data=formatted_latent_metrics_test, index=mi_test, columns=mc_ltime)
        d_ltime = pd.concat([d_latent_time_train, d_latent_time_test], keys=['train', 'test']).melt(
            value_name='latent time value', ignore_index=False)

        # merge latents into d_all
        d_all = pd.merge(d_all, d_ltime, left_index=True, right_index=True, suffixes=(None, '_temp'))
        # clear away redundant acquisition entries
        d_all = d_all.loc[d_all.acquisition == d_all.acquisition_temp].drop(columns=['acquisition_temp'])

    # next, seaborn can only plot certain values against others in column format, so we need to
    # move some of the row indicators to columns (unpivot?)
    fig_path = os.path.join(exp_path, 'figures')
    plot_from_dataframe(d_all, fig_path)


def gather_experiment_logs_file_paths(TEST_IGNORE, TRAIN_IGNORE, args, exp_args, logs_lookup, test_objects,
                                      train_objects):
    # Nested dictionary: [train_geo/test_geo][random/bald][all/object_name]
    logs_lookup_by_object = {
        'train_geo': {
            'random': {
                'all': [],
            },
            'bald': {
                'all': [],
            },
            'constrained_random': {
                'all': []
            }
        },
        'test_geo': {
            'random': {
                'all': [],
            },
            'bald': {
                'all': [],
            },
            'constrained_random': {
                'all': []
            }
        }
    }
    valid_train_objects, valid_train_pstables, valid_train_min_dists, valid_train_ratios, valid_train_maxdims, valid_train_rrates, valid_train_eigval_prod = filter_objects(
        object_names=train_objects['object_data']['object_names'],
        ignore_list=TRAIN_IGNORE,
        phase='train',
        dataset_name=exp_args.dataset_name,
        min_pstable=0.0,
        max_pstable=1.0,
        min_dist_threshold=0.0,
        max_objects=500
    )
    for ox, object_name, _ in valid_train_objects:

        if object_name not in logs_lookup_by_object['train_geo']['random']:
            logs_lookup_by_object['train_geo']['random'][object_name] = []
        if object_name not in logs_lookup_by_object['train_geo']['bald']:
            logs_lookup_by_object['train_geo']['bald'][object_name] = []
        if object_name not in logs_lookup_by_object['train_geo']['constrained_random']:
            logs_lookup_by_object['train_geo']['constrained_random'][object_name] = []

        random_log_key = f'grasp_{exp_args.exp_name}_fit_random_train_geo_object{ox}'
        if random_log_key in logs_lookup['fitting_phase']['random']:
            random_log_fname = logs_lookup['fitting_phase']['random'][random_log_key]

            logs_lookup_by_object['train_geo']['random']['all'].append(random_log_fname)
            logs_lookup_by_object['train_geo']['random'][object_name].append(random_log_fname)

        constrained_random_log_key = f'grasp_{exp_args.exp_name}_fit_constrained_random_train_geo_object{ox}'
        if constrained_random_log_key in logs_lookup['fitting_phase']['constrained_random']:
            constrained_random_log_fname = logs_lookup['fitting_phase']['constrained_random'][
                constrained_random_log_key]

            logs_lookup_by_object['train_geo']['constrained_random']['all'].append(constrained_random_log_fname)
            logs_lookup_by_object['train_geo']['constrained_random'][object_name].append(constrained_random_log_fname)

        bald_log_key = f'grasp_{args.exp_name}_fit_bald_train_geo_object{ox}'
        if 'bald' in logs_lookup['fitting_phase'] and bald_log_key in logs_lookup['fitting_phase']['bald']:
            bald_log_fname = logs_lookup['fitting_phase']['bald'][bald_log_key]

            logs_lookup_by_object['train_geo']['bald']['all'].append(bald_log_fname)
            logs_lookup_by_object['train_geo']['bald'][object_name].append(bald_log_fname)
    print(f'{len(valid_train_objects)} train geo objects included.')
    valid_test_objects, valid_test_pstables, valid_test_min_dists, valid_test_ratios, valid_test_maxdims, valid_test_rrates, valid_test_eigval_prod = filter_objects(
        object_names=test_objects['object_data']['object_names'],
        ignore_list=TEST_IGNORE,
        phase='test',
        dataset_name=exp_args.dataset_name,
        min_pstable=0.0,
        max_pstable=1.0,
        min_dist_threshold=0.0,
        max_objects=500
    )
    for ox, object_name, _ in valid_test_objects:

        if object_name not in logs_lookup_by_object['test_geo']['random']:
            logs_lookup_by_object['test_geo']['random'][object_name] = []
        if object_name not in logs_lookup_by_object['test_geo']['bald']:
            logs_lookup_by_object['test_geo']['bald'][object_name] = []
        if object_name not in logs_lookup_by_object['test_geo']['constrained_random']:
            logs_lookup_by_object['test_geo']['constrained_random'][object_name] = []

        random_log_key = f'grasp_{exp_args.exp_name}_fit_random_test_geo_object{ox}'
        if random_log_key in logs_lookup['fitting_phase']['random']:
            random_log_fname = logs_lookup['fitting_phase']['random'][random_log_key]

            logs_lookup_by_object['test_geo']['random']['all'].append(random_log_fname)
            logs_lookup_by_object['test_geo']['random'][object_name].append(random_log_fname)

        constrained_random_log_key = f'grasp_{exp_args.exp_name}_fit_constrained_random_test_geo_object{ox}'
        if constrained_random_log_key in logs_lookup['fitting_phase']['constrained_random']:
            constrained_random_log_fname = logs_lookup['fitting_phase']['constrained_random'][
                constrained_random_log_key]
            logs_lookup_by_object['test_geo']['constrained_random']['all'].append(constrained_random_log_fname)
            logs_lookup_by_object['test_geo']['constrained_random'][object_name].append(constrained_random_log_fname)

        bald_log_key = f'grasp_{args.exp_name}_fit_bald_test_geo_object{ox}'
        if 'bald' in logs_lookup['fitting_phase'] and bald_log_key in logs_lookup['fitting_phase']['bald']:
            bald_log_fname = logs_lookup['fitting_phase']['bald'][bald_log_key]
            logs_lookup_by_object['test_geo']['bald']['all'].append(bald_log_fname)
            logs_lookup_by_object['test_geo']['bald'][object_name].append(bald_log_fname)
    print(f'{len(valid_test_objects)} test geo objects included.')
    return logs_lookup_by_object, \
        valid_test_min_dists, valid_test_objects, valid_test_pstables, valid_test_ratios, valid_test_maxdims, valid_test_rrates, valid_test_eigval_prod, \
        valid_train_min_dists, valid_train_objects, valid_train_pstables, valid_train_ratios, valid_train_maxdims, valid_train_rrates, valid_train_eigval_prod


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['create', 'training', 'fitting',
                                                           'playback', 'testing', 'task-eval'])
    parser.add_argument('--dataset-name', type=str, default='')
    parser.add_argument('--exp-name', required=True, type=str)
    parser.add_argument('--strategy', type=str, choices=['bald', 'random'], default='random')
    parser.add_argument('--constrained', action='store_true', default=False)
    parser.add_argument('--amortize', action='store_true', default=False)
    parser.add_argument('--vis-train-objects', nargs='*', type=int, default=[],
                        help='train objects (by ID) to visualize data collection')
    parser.add_argument('--vis-test-objects', nargs='*', type=int, default=[],
                        help='test objects (by ID) to visualize data collection')
    parser.add_argument('--playback-train-fitting', action='store_true', default=False,
                        help='this is really to add an new metrics to fitting we did not have before')
    parser.add_argument('--playback-test-fitting', action='store_true', default=False,
                        help='this is really to add an new metrics to fitting we did not have before')

    args = parser.parse_args()

    if args.phase == 'create':
        create_experiment(args)
    elif args.phase == 'training':
        run_training_phase(args)
    elif args.phase == 'fitting':
        run_fitting_phase(args)
    elif args.phase == 'playback':
        if len(args.vis_train_objects) == 0 \
                or len(args.vis_test_objects) == 0 \
                or args.playback_train_fitting \
                or args.playback_test_fitting:
            print('No objects given to visualize. Specify objects to visualize using '
                  '--vis-train-objects or --vis-test-objects')
        playback_fitting_phase(args)
    elif args.phase == 'testing':
        run_testing_phase(args)
    elif args.phase == 'task-eval':
        run_task_eval_phase(args)
