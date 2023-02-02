import multiprocessing as mp
import pickle

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector, sample_grasp_X
from learning.models.grasp_np.create_gnp_data import process_geometry
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


def select_gnp_dataset_ix(dataset, ix):
    new_dataset = {
        'grasp_data': {},
        'object_data': dataset['object_data'],
        'metadata': dataset['metadata']
    }

    for field_name in dataset['grasp_data']:
        new_dataset['grasp_data'][field_name] = {}
        for ox, val_list in dataset['grasp_data'][field_name].items():
            new_dataset['grasp_data'][field_name][ox] = [val_list[ix]]

    return new_dataset


def merge_gnp_datasets(dataset1, dataset2):
    for field_name in dataset1['grasp_data']:
        for ox in dataset1['grasp_data'][field_name]:
            new_data = dataset2['grasp_data'][field_name][ox]
            dataset1['grasp_data'][field_name][ox].extend(new_data)
    return dataset1


def drop_last_grasp_in_dataset(dataset):
    one_less_dataset = {'grasp_data': {}, 'object_data': dataset['object_data']}
    for field_name in dataset['grasp_data']:
        one_less_dataset['grasp_data'][field_name] = {}
        for ox in dataset['grasp_data'][field_name]:
            one_less_dataset['grasp_data'][field_name][ox] = dataset['grasp_data'][field_name][ox][:-1]

    return one_less_dataset

def explode_dataset_into_list_of_datasets(dataset):
    template = {'grasp_data': {}, 'object_data': dataset['object_data']}
    num_grasps = len(list(dataset['grasp_data'].values())[0].values())
    grasp_sets = [template.copy() for _ in range(num_grasps)]
    for field_name in dataset['grasp_data']:
        for ox in dataset['grasp_data'][field_name]:
            for entry, grasp_set in zip(dataset['grasp_data'][field_name][ox], grasp_sets):
                grasp_set['grasp_data'][field_name] = {}
                grasp_set['grasp_data'][field_name][ox] = [entry]
    return grasp_sets

def sample_unlabeled_gnp_data(n_samples, object_set, object_ix):
    grasp_data = sample_unlabeled_data(n_samples, object_set, object_ix)
    grasp_gnp_data = process_geometry(
        grasp_data,
        radius=0.03,
        skip=1,
        verbose=False
    )
    return grasp_gnp_data


def sample_unlabeled_data(n_samples, object_set, object_ix=None):
    object_name, object_properties, tmp_ox = get_fit_object(object_set)
    if object_ix is None:
        object_ix = tmp_ox
    graspable_body = graspablebody_from_vector(object_name, object_properties)

    object_grasp_data, object_grasp_ids, object_grasp_forces, object_grasp_labels = [], [], [], []
    raw_grasps = []

    worker_pool = mp.Pool(processes=20)
    # TODO: MAGIC CURVATURE NUMBERS ALERT
    fn_args = [graspable_body, object_properties, 10000, (0.005, 0.01, 0.02)]
    results = worker_pool.starmap(sample_grasp_X, [fn_args] * n_samples)
    worker_pool.close()

    for grasp, X in results:
        raw_grasps.append(grasp)
        object_grasp_data.append(X)
        object_grasp_ids.append(object_ix)
        object_grasp_forces.append(grasp.force)
        object_grasp_labels.append(-1)

    unlabeled_dataset = {
        'grasp_data': {
            'raw_grasps': raw_grasps,
            'grasps': object_grasp_data,
            'forces': object_grasp_forces,
            'object_ids': object_grasp_ids,
            'labels': object_grasp_labels
        },
        'object_data': object_set,
        'metadata': {
            'n_samples': n_samples,
        }
    }
    return unlabeled_dataset


def get_labels(grasp_dataset):
    raw_grasps = grasp_dataset['grasp_data']['raw_grasps']

    graspable_body = raw_grasps[0].graspable_body
    labeler = GraspStabilityChecker(graspable_body, stability_direction='all', label_type='relpose',
                                    recompute_inertia=True)
    for gx in range(0, len(raw_grasps)):
        label = labeler.get_label(raw_grasps[gx])
        print('Label', label)
        grasp_dataset['grasp_data']['labels'][gx] = label
    labeler.disconnect()
    return grasp_dataset


def get_single_label(graspable_body, grasp):
    labeler = GraspStabilityChecker(graspable_body, stability_direction='all', label_type='relpose',
                                    recompute_inertia=True)
    label = labeler.get_label(grasp)
    labeler.disconnect()
    return label


def get_labels_gnp(grasp_dataset):
    raw_grasps = grasp_dataset['grasp_data']['raw_grasps']
    worker_pool = mp.Pool(processes=20)
    all_tasks = []
    for ox in raw_grasps:
        graspable_body = raw_grasps[ox][0].graspable_body
        for gx in range(0, len(raw_grasps[ox])):
            all_tasks.append((graspable_body, raw_grasps[ox][gx]))
        labels = worker_pool.starmap(get_single_label, all_tasks)
        grasp_dataset['grasp_data']['labels'][ox] = labels
    worker_pool.close()
    return grasp_dataset
