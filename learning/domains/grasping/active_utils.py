import multiprocessing as mp
import pickle
import time

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector, sample_grasp_Xs
from pb_robot.planners.antipodalGraspPlanner import GraspSampler, GraspStabilityChecker


def get_train_and_fit_objects(pretrained_ensemble_path, use_latents, fit_objects_fname, fit_object_ix):
    train_logger = ActiveExperimentLogger(exp_path=pretrained_ensemble_path)
    with open(train_logger.args.train_dataset_fname, 'rb') as handle:
        train_dataset = pickle.load(handle)

    train_objects = train_dataset['object_data']
    with open(fit_objects_fname, 'rb') as handle:
        fit_objects = pickle.load(handle)['object_data']

    fit_object_name = fit_objects['object_names'][fit_object_ix]
    fit_object_props = fit_objects['object_properties'][fit_object_ix]

    train_objects['object_names'].append(fit_object_name)
    train_objects['object_properties'].append(fit_object_props)

    return train_objects


def get_fit_object(object_set):
    name = object_set['object_names'][-1]
    props = object_set['object_properties'][-1]
    ix = len(object_set['object_names']) - 1
    return name, props, ix


def sample_unlabeled_data_train(n_samples_per_object, object_set, object_ixs):
    """ Generate unlabeled grasps for all objects in the dataset. """
    # Parallelize data generation.
    tasks = []
    for ox in object_ixs:
        tasks.append((
            n_samples_per_object,
            object_set['object_names'][ox],
            object_set['object_properties'][ox],
            ox
        ))
    start = time.time()
    pool = mp.Pool(processes=20)
    all_grasp_data = pool.starmap(sample_unlabeled_data_object, tasks)
    pool.close()
    # print(f'Time: {time.time() - start}')

    # Merge individual grasp datasets.
    merged_grasp_data = all_grasp_data[0]
    for grasp_data in all_grasp_data[1:]:
        for k in grasp_data:
            merged_grasp_data[k].extend(grasp_data[k])

    unlabeled_dataset = {
        'grasp_data': merged_grasp_data,
        'object_data': object_set,
        'metadata': {
            'n_samples': n_samples_per_object,
        }
    }
    return unlabeled_dataset


def sample_unlabeled_data_fit(n_samples_per_object, object_set):
    """ During the fitting phase, only generate data for the last object. """
    object_name, object_properties, object_ix = get_fit_object(object_set)

    grasp_data = sample_unlabeled_data_object(
        n_samples_per_object,
        object_name,
        object_properties,
        object_ix
    )
    unlabeled_dataset = {
        'grasp_data': grasp_data,
        'object_data': object_set,
        'metadata': {
            'n_samples': n_samples_per_object,
        }
    }
    return unlabeled_dataset

def sample_unlabeled_data_object(n_samples, object_name, object_properties, object_ix):
    """ Generate unlabeled grasps for a single object. """
    graspable_body = graspablebody_from_vector(object_name, object_properties)

    object_grasp_data, object_grasp_ids, object_grasp_forces, object_grasp_labels = [], [], [], []  
    raw_grasps = []

    grasps_and_Xs = sample_grasp_Xs(graspable_body, object_properties, n_points_per_object=10000, n_grasps=n_samples)

    raw_grasps = [x[0] for x in grasps_and_Xs]
    object_grasp_data = [x[1] for x in grasps_and_Xs]
    object_grasp_ids = [object_ix]*n_samples
    object_grasp_forces = [x[0].force for x in grasps_and_Xs]
    object_grasp_labels = [0]*n_samples

    grasp_data =  {
        'raw_grasps': raw_grasps,
        'grasps': object_grasp_data,
        'forces': object_grasp_forces,
        'object_ids': object_grasp_ids,
        'labels': object_grasp_labels
    }

    return grasp_data


def get_labels(grasp_dataset):
    raw_grasps = grasp_dataset['grasp_data']['raw_grasps']
    graspable_body = raw_grasps[0].graspable_body
    labeler = GraspStabilityChecker(graspable_body, stability_direction='all', label_type='relpose', recompute_inertia=True)
    for gx in range(0, len(raw_grasps)):
        label = labeler.get_label(raw_grasps[gx])
        print('Label', label)
        grasp_dataset['grasp_data']['labels'][gx] = label
    labeler.disconnect()
    return grasp_dataset
