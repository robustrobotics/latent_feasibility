import argparse
import copy
import os

import multiprocessing as mp
import numpy as np

from pybullet_object_models import ycb_objects

from pb_robot.planners.antipodalGraspPlanner import (
    GraspSimulationClient,
    graspablebody_from_vector
)

IGNORE_MODELS = [
    'YCB::YcbMasterChefCan',
    'YCB::YcbChipsCan',
    'YCB::YcbPear',
    'ShapeNet::ChestOfDrawers_1aada3ab6bb4295635469b95109803c',
]
# with open('learning/data/grasping/object_lists/non-watertight.txt', 'r') as handle:
#     IGNORE_MODELS += handle.read().split('\n')
# with open('learning/data/grasping/object_lists/no-valid-grasps.txt', 'r') as handle:
#     IGNORE_MODELS += handle.read().split('\n')
SHAPENET_IGNORE_CATEGORIES = [
    'Paper',
    'Room'
]



def get_primitive_models(primitive_urdf_root=''):
    """ get a list of all primitive models as a string """
    objects_names = [os.path.splitext(name)[0] for name in os.listdir(primitive_urdf_root)]
    include_objects = []
    for obj_name in objects_names:
        category, _ = obj_name.split('_')
        full_name = f'Primitive::{obj_name}'
        include_objects.append(full_name)
    return include_objects


def get_shapenet_models(shapenet_urdf_root=''):
    """ Get a list of all ShapeNet model names as a string. """
    objects_names = [os.path.splitext(name)[0] for name in os.listdir(shapenet_urdf_root)]
    include_objects = []
    for obj_name in objects_names:
        category, _ = obj_name.split('_')
        if category in SHAPENET_IGNORE_CATEGORIES:
            continue
        full_name = f'ShapeNet::{obj_name}'
        if full_name not in IGNORE_MODELS:
            include_objects.append(full_name)
    return include_objects


def get_ycb_models():
    """ Get a list of all YCB model names as a string. """
    objects_names = [name for name in os.listdir(ycb_objects.getDataPath()) if 'Ycb' in name]
    include_objects = []
    for obj_name in objects_names:
        full_name = f'YCB::{obj_name}'
        if full_name not in IGNORE_MODELS:
            include_objects.append(full_name)
    return include_objects


def select_objects(ycb_object_names, sn_object_names, pm_object_names, datasets, n_objects):
    """ Select a subset of objects. Remove from source list to avoid reuse. """
    all_objects = []
    if 'YCB' in datasets:
        all_objects += copy.deepcopy(ycb_object_names)
    if 'ShapeNet' in datasets:
        all_objects += copy.deepcopy(sn_object_names)
    if len([zet for zet in datasets if zet not in ['Shapenet', 'YCB']]) > 0: # if there are any primitives
        all_objects += copy.deepcopy(pm_object_names)

    print(n_objects, len(all_objects))
    chosen_objects = np.random.choice(all_objects, n_objects, replace=False)

    for obj in chosen_objects:
        if obj.split('::')[0] == 'YCB':
            ycb_object_names.remove(obj)
        elif obj.split('::')[0] == 'ShapeNet':
            sn_object_names.remove(obj)
        else:
            pm_object_names.remove(obj)

    return chosen_objects

def get_object_volumes(object_name):
    body = graspablebody_from_vector(
        object_name=object_name,
        vector=[0., 0., 0., 0.5, 0.5]
    )
    client = GraspSimulationClient(
        graspable_body=body,
        show_pybullet=False
    )
    volume = client.mesh.volume
    bb_volume = client.mesh.bounding_box.volume
    ch_volume = client.mesh.convex_hull.volume
    client.disconnect()
    return {
        'volume': volume,
        'convex_hull': ch_volume,
        'bounding_box': bb_volume
    }


def filter_by_volume(object_list, volumes, min_volume=0.001, max_volume=0.027):   
    new_objects = []
    for name, vols in zip(object_list, volumes):
        vol = vols['volume']
        if vol > min_volume and vol < max_volume:
            new_objects.append(name)
            # if vol < 0.0015 or vol > 0.0075:
            #     print(vol, name)
    return new_objects

def filter_by_ratio(object_list, volumes):
    """
    Check how similar the object's volume is to the bounding box. Remove those near the extreme.
    """
    new_objects = []
    for name, vols in zip(object_list, volumes):
        bb_ratio = vols['bounding_box'] / vols['volume']
        ch_ratio = vols['convex_hull'] / vols['volume']
        if bb_ratio < 1.1:
            continue
        if ch_ratio > 5.:
            continue
        new_objects.append(name)

    return new_objects

OBJECTS_LIST_DIR = 'learning/data/pushing/box_data_fixed_sim'
if __name__ == '__main__':
    # shapenet_root = os.environ['SHAPENET_ROOT']
    # primitive_parent_root = os.environ['PRIMITIVE_PARENT_ROOT']
    # Need to change these based on the machine running the code 
    shapenet_root = '../object_models/shapenet-sem-v2/'
    primitive_parent_root = '../object_models/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-objects-fname', type=str, required=True)
    parser.add_argument('--test-objects-fname', type=str, required=True)
    parser.add_argument('--train-objects-datasets', nargs='+', required=True)
    parser.add_argument('--test-objects-datasets', nargs='+', required=True)
    parser.add_argument('--n-train', type=int, required=True)
    parser.add_argument('--n-test', type=int, required=True)
    parser.add_argument('--n-processes', type=int, default=1)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(OBJECTS_LIST_DIR):
        os.makedirs(OBJECTS_LIST_DIR)

    assert len(args.train_objects_datasets) > 0
    assert len(args.test_objects_datasets) > 0

    all_ycb_objects = get_ycb_models()

    all_shapenet_objects = get_shapenet_models(os.path.join(shapenet_root, 'urdfs'))
    if 'ShapeNet' in args.train_objects_datasets + args.test_objects_datasets:
        print(f'Before length: {len(all_shapenet_objects)}')
        worker_pool = mp.Pool(processes=args.n_processes)
        volumes = worker_pool.map(get_object_volumes, all_shapenet_objects)
        worker_pool.close()
        all_shapenet_objects = filter_by_ratio(all_shapenet_objects, volumes)
        print(f'After ratio length: {len(all_shapenet_objects)}')
        all_shapenet_objects = filter_by_volume(all_shapenet_objects, volumes)
        print(f'After volume length: {len(all_shapenet_objects)}')

    # then use train_object_datasets and then lookup with try except and then just inform
    # user if the set name passed as an argument does not exist

    # removing special cases from the sets defined
    non_ycb_shapenet_items = [s for s in (args.train_objects_datasets + args.test_objects_datasets) \
                              if s not in ['ShapeNet', 'YCB', 'shapenet-sem']]
    non_ycb_shapenet_items = list(set(non_ycb_shapenet_items))

    all_primitive_objects = []
    for primitive_models in non_ycb_shapenet_items:
        full_primitive_dir = os.path.join(primitive_parent_root, primitive_models, 'urdfs')
        try:
            all_primitive_objects += get_primitive_models(full_primitive_dir)
        except FileExistsError as e:
            print('Could not find dataset ' + full_primitive_dir + '... skipping...')

    # Remove objects from lists as you go.
    train_objects = select_objects(ycb_object_names=all_ycb_objects,
                                   sn_object_names=all_shapenet_objects,
                                   pm_object_names=all_primitive_objects,
                                   datasets=args.train_objects_datasets,
                                   n_objects=args.n_train)
    test_objects = select_objects(ycb_object_names=all_ycb_objects,
                                  sn_object_names=all_shapenet_objects,
                                  pm_object_names=all_primitive_objects,
                                  datasets=args.test_objects_datasets,
                                  n_objects=args.n_test)

    # import sys; sys.exit()
    with open(os.path.join(OBJECTS_LIST_DIR, args.train_objects_fname), 'w') as handle:
        handle.write('\n'.join(train_objects))
    with open(os.path.join(OBJECTS_LIST_DIR, args.test_objects_fname), 'w') as handle:
        handle.write('\n'.join(test_objects))
