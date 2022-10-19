"""
Each experiment involves many dataset artifacts. This involves one large training
dataset (and corresponding validation set using the same objects), and many smaller
adaptation datasets used to evaluate accuracy throughout adaptation.

Three object sets are created:
- train_geo_train_props
- train_geo_test_props
- test_geo_test_pros

Fitting phase datasets are created for each of these to cover a range of difficulty:
- fit_grasps_train_geo_trainprop (easy: test on same objects used to train)
- fit_grasps_train_geo (medium: test on same geometry but different intrinsic properties)
- fit_grasps_test_geo (hard: test on different geometry and parts)
"""

import argparse
import multiprocessing
import pickle
import os

from types import SimpleNamespace

from learning.domains.grasping.generate_grasp_datasets import generate_objects, generate_datasets


def get_object_list(fname):
    """ Parse object text file. """
    print('Opening', fname)
    with open(fname, 'r') as handle:
        objects = handle.readlines()
    return [o.strip() for o in objects if len(o) > 1]


parser = argparse.ArgumentParser()
parser.add_argument('--train-objects-fname', type=str, required=True)
parser.add_argument('--test-objects-fname', type=str, required=True)
parser.add_argument('--data-root-name', type=str, required=True)
parser.add_argument('--n-property-samples-train', type=int, required=True)
parser.add_argument('--n-property-samples-test', type=int, required=True)
parser.add_argument('--n-grasps-per-object', type=int, required=True)
parser.add_argument('--n-points-per-object', type=int, required=True)
parser.add_argument('--n-fit-grasps', type=int, required=True)
parser.add_argument('--grasp-noise', type=float, required=True)
parser.add_argument('--n-processes', type=int, default=1)
new_args = parser.parse_args()
print(new_args)


if __name__ == '__main__':

    # Directory setup.
    data_root_path = os.path.join('learning', 'data', 'grasping', new_args.data_root_name)
    args_path = os.path.join(data_root_path, 'args.pkl')
    if os.path.exists(data_root_path):
        input('[Warning] Data root directory already exists. Continue using initial args?')
        with open(args_path, 'rb') as handle:
            args = pickle.load(handle)
            args.n_processes = new_args.n_processes
    else:
        os.mkdir(data_root_path)
        with open(args_path, 'wb') as handle:
            pickle.dump(new_args, handle)
        args = new_args

    worker_pool = multiprocessing.Pool(processes=args.n_processes)

    objects_path = os.path.join(data_root_path, 'objects')
    grasps_path = os.path.join(data_root_path, 'grasps')
    if not os.path.exists(objects_path):
        os.mkdir(objects_path)
        os.mkdir(grasps_path)

    train_objects = get_object_list(args.train_objects_fname)
    test_objects = get_object_list(args.test_objects_fname)

    # Generate initial object sets.
    object_tasks = []

    print('[Objects] Generating train objects.')
    train_objects_path = os.path.join(objects_path, 'train_geo_train_props.pkl')
    if not os.path.exists(train_objects_path):
        train_objects_args = SimpleNamespace(
            fname=train_objects_path,
            object_names=train_objects,
            n_property_samples=args.n_property_samples_train)
        object_tasks.append(train_objects_args)

    print('[Objects] Generating test objects: novel geometry.')
    test_objects_path = os.path.join(objects_path, 'test_geo_test_props.pkl')
    if not os.path.exists(test_objects_path):
        test_objects_args = SimpleNamespace(
            fname=test_objects_path,
            object_names=test_objects,
            n_property_samples=args.n_property_samples_test)
        object_tasks.append(test_objects_args)

    print('[Objects] Generating test objects: train geometry.')
    test_objects_samegeo_path = os.path.join(objects_path, 'train_geo_test_props.pkl')
    if not os.path.exists(test_objects_samegeo_path):
        test_objects_samegeo_args = SimpleNamespace(
            fname=test_objects_samegeo_path,
            object_names=train_objects,
            n_property_samples=args.n_property_samples_test)
        object_tasks.append(test_objects_samegeo_args)
    
    worker_pool.map(generate_objects, object_tasks)

    # Generate training and validation sets used for the training phase.
    training_phase_path = os.path.join(grasps_path, 'training_phase')
    if not os.path.exists(training_phase_path):
        os.mkdir(training_phase_path)
    
    SKIP_TRAIN_OBJECTS = [1665, 1666, 1667, 1668, 1669]
    train_dataset_tasks = []

    print('[Grasps] Generating train grasps for training phase.')
    # train_grasps_path = os.path.join(training_phase_path, 'train_grasps.pkl')
    for ox in range(0, len(train_objects)*args.n_property_samples_train):
        if ox in SKIP_TRAIN_OBJECTS:
            continue
        train_grasps_path = os.path.join(training_phase_path, f'train_grasps_object{ox}.pkl')
        if not os.path.exists(train_grasps_path):
            train_grasps_args = SimpleNamespace(
                fname=train_grasps_path,
                objects_fname=train_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_grasps_per_object,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            train_dataset_tasks.append(train_grasps_args)

    worker_pool.map(generate_datasets, train_dataset_tasks)

    print('[Grasps] Generating validation grasps for training phase.')
    val_dataset_tasks = []
    for ox in range(0, len(train_objects)*args.n_property_samples_train):
        val_grasps_path = os.path.join(training_phase_path, f'val_grasps_object{ox}.pkl')
        if not os.path.exists(val_grasps_path):
            val_grasps_args = SimpleNamespace(
                fname=val_grasps_path,
                objects_fname=train_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=10,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            val_dataset_tasks.append(val_grasps_args)

    worker_pool.map(generate_datasets, val_dataset_tasks)

    # Generate fitting object datasets.
    fitting_phase_path = os.path.join(grasps_path, 'fitting_phase')
    if not os.path.exists(fitting_phase_path):
        os.mkdir(fitting_phase_path)

    fit_dataset_tasks = []
    for ox in range(0, min(100, len(test_objects)*args.n_property_samples_test)):
        print('[Grasps] Generating grasps for evaluating fitting phase for object %d.' % ox)
        fit_grasps_path = os.path.join(fitting_phase_path, 'fit_grasps_test_geo_object%d.pkl' % ox)
        if not os.path.exists(fit_grasps_path):
            fit_grasps_args = SimpleNamespace(
                fname=fit_grasps_path,
                objects_fname=test_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            fit_dataset_tasks.append(fit_grasps_args)
    
    worker_pool.map(generate_datasets, fit_dataset_tasks)

    fit_dataset_samegeo_tasks = []
    for ox in range(0, min(100, len(train_objects)*args.n_property_samples_test)):
        print(f'[Grasps] Generating grasps for evaluating fitting phase for samegeo object {ox}.')
        fit_grasps_samegeo_path = os.path.join(fitting_phase_path,
                                               f'fit_grasps_train_geo_object{ox}.pkl')
        if not os.path.exists(fit_grasps_samegeo_path):
            fit_grasps_samegeo_args = SimpleNamespace(
                fname=fit_grasps_samegeo_path,
                objects_fname=test_objects_samegeo_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox,
                grasp_noise=args.grasp_noise)
            fit_dataset_samegeo_tasks.append(fit_grasps_samegeo_args)
    
    worker_pool.map(generate_datasets, fit_dataset_samegeo_tasks)

    fit_dataset_samegeo_tasks = []
    for ox in range(0, min(100, len(train_objects)*args.n_property_samples_train)):
        print(f'[Grasps] Generating grasps for evaluating fitting phase for samegeo sameprop object {ox}.')
        fit_grasps_samegeo_sameprop_path = os.path.join(fitting_phase_path, 'fit_grasps_train_geo_trainprop_object%d.pkl' % ox)
        if not os.path.exists(fit_grasps_samegeo_sameprop_path):
            fit_grasps_samegeo_args = SimpleNamespace(
                fname=fit_grasps_samegeo_sameprop_path,
                objects_fname=train_objects_path,
                n_points_per_object=args.n_points_per_object,
                n_grasps_per_object=args.n_fit_grasps,
                object_ix=ox,
                grasp_noise=0.0)
            fit_dataset_samegeo_tasks.append(fit_grasps_samegeo_args)
        
    worker_pool.map(generate_datasets, fit_dataset_samegeo_tasks)
