import argparse
import copy
import os

import numpy as np
import pickle


IGNORE_MODELS = [
    'YCB::YcbMasterChefCan',
    'YCB::YcbChipsCan',
    'YCB::YcbPear',
    'ShapeNet::ChestOfDrawers_1aada3ab6bb4295635469b95109803c',
]
SHAPENET_IGNORE_CATEGORIES = [
    'Paper',
    'Room'
]


def get_shapenet_models(shapenet_urdf_root='', metadata={}):
    """ Get a list of all ShapeNet model names as a string. """
    objects_names = [os.path.splitext(name)[0] for name in os.listdir(shapenet_urdf_root)]
    include_objects = []
    for obj_name in objects_names:
        category, _ = obj_name.split('_')
        if category in SHAPENET_IGNORE_CATEGORIES:
            continue
        full_name = f'ShapeNet::{obj_name}'
        if full_name not in metadata:
            continue
        if full_name not in IGNORE_MODELS:
            include_objects.append(full_name)
    return include_objects


def select_objects(sn_object_names, n_objects):
    """ Select a subset of objects. Remove from source list to avoid reuse. """
    all_objects = copy.deepcopy(sn_object_names)

    chosen_objects = np.random.choice(all_objects, n_objects, replace=False)

    for obj in chosen_objects:
        sn_object_names.remove(obj)

    return chosen_objects


def filter_by_volume(object_list, metadata, min_volume=0.001, max_volume=0.027):
    new_objects = []
    for name in object_list:
        vol = metadata[name]['bb_volume']
        if vol > min_volume and vol < max_volume:
            new_objects.append(name)
    return new_objects

def filter_by_ratio(object_list, metadata):
    """
    Check how similar the object's volume is to the bounding box. Remove those near the extreme.
    """
    new_objects = []
    for name in object_list:
        bb_ratio = metadata[name]['bb_volume'] / metadata[name]['volume']
        ch_ratio = metadata[name]['ch_volume'] / metadata[name]['volume']
        if bb_ratio < 1.1:
            continue
        if ch_ratio > 5.:
            continue
        new_objects.append(name)

    return new_objects

def filter_by_rejection_rate(object_list, metadata, rate_threshold):
    new_objects = []
    for name in object_list:
        r_rate = metadata[name]['rejection_rate']
        if r_rate >= rate_threshold:
            continue
        new_objects.append(name)
    
    return new_objects

def filter_by_avg_min_dist(object_list, metadata, min_val, max_val):
    new_objects = []
    for name in object_list:
        dist = metadata[name]['avg_min_dist50']
        if dist >= max_val or dist <= min_val:
            continue
        new_objects.append(name)
    
    return new_objects

def filter_by_maxdim(object_list, metadata, max_val):
    new_objects = []
    for name in object_list:
        bounds = metadata[name]['bounds']
        maxdim = np.max(bounds[1])*2.
        if maxdim >= max_val:
            continue
        new_objects.append(name)

    return new_objects


OBJECTS_LIST_DIR = 'learning/data/grasping/object_lists'
if __name__ == '__main__':
    shapenet_root = os.environ['SHAPENET_ROOT']

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-objects-fname', type=str, required=True)
    parser.add_argument('--test-objects-fname', type=str, required=True)
    parser.add_argument('--n-train', type=int, required=True)
    parser.add_argument('--n-test', type=int, required=True)
    args = parser.parse_args()
    print(args)

    
    metadata_fname = os.path.join(shapenet_root, 'object_infos.pkl')
    with open(metadata_fname, 'rb') as handle:
        metadata = pickle.load(handle)

    all_shapenet_objects = get_shapenet_models(os.path.join(shapenet_root, 'urdfs'), metadata)
    print(f'Before length: {len(all_shapenet_objects)}')
    all_shapenet_objects = filter_by_rejection_rate(all_shapenet_objects, metadata, rate_threshold=0.9)
    print(f'After rejection rate length: {len(all_shapenet_objects)}')
    # all_shapenet_objects = filter_by_ratio(all_shapenet_objects, metadata)
    # print(f'After ratio length: {len(all_shapenet_objects)}')
    # all_shapenet_objects = filter_by_volume(all_shapenet_objects, metadata, min_volume=0.001, max_volume=0.027)
    # print(f'After volume length: {len(all_shapenet_objects)}')
    # all_shapenet_objects = filter_by_avg_min_dist(all_shapenet_objects, metadata, min_val=0.005, max_val=0.02)
    # print(f'After avg_min_dist length: {len(all_shapenet_objects)}')
    all_shapenet_objects = filter_by_maxdim(all_shapenet_objects, metadata, max_val=0.3)
    print(f'After bounds length: {len(all_shapenet_objects)}')

    # then use train_object_datasets and then lookup with try except and then just inform
    # user if the set name passed as an argument does not exist

    # Remove objects from lists as you go.
    train_objects = select_objects(
        sn_object_names=all_shapenet_objects,
        n_objects=args.n_train
    )
    test_objects = select_objects(
        sn_object_names=all_shapenet_objects,
        n_objects=args.n_test
    )

    # import sys; sys.exit()
    with open(os.path.join(OBJECTS_LIST_DIR, args.train_objects_fname), 'w') as handle:
        handle.write('\n'.join(train_objects))
    with open(os.path.join(OBJECTS_LIST_DIR, args.test_objects_fname), 'w') as handle:
        handle.write('\n'.join(test_objects))
