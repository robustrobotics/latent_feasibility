import argparse
import copy
import numpy as np
import os
from xml.etree.ElementInclude import include

from pybullet_object_models import ycb_objects


IGNORE_MODELS = [
    'YCB::YcbMasterChefCan',
    'YCB::YcbChipsCan',
    'YCB::YcbPear',
    'ShapeNet::ChestOfDrawers_1aada3ab6bb4295635469b95109803c',
]
with open('learning/data/grasping/object_lists/non-watertight.txt', 'r') as handle:
    IGNORE_MODELS += handle.read().split('\n')
with open('learning/data/grasping/object_lists/no-valid-grasps.txt', 'r') as handle:
    IGNORE_MODELS += handle.read().split('\n')
print(len(IGNORE_MODELS))
SHAPENET_IGNORE_CATEGORIES = [
    'Paper',
    'Room'
]

def get_shapenet_models(shapenet_urdf_root=''):
    objects_names = [os.path.splitext(name)[0] for name in os.listdir(shapenet_urdf_root)]
    include_objects = []
    for obj_name in objects_names:
        category, sn_id = obj_name.split('_')
        if category in SHAPENET_IGNORE_CATEGORIES:
            continue
        full_name = f'ShapeNet::{obj_name}'
        if full_name not in IGNORE_MODELS:
            include_objects.append(full_name)
    return include_objects


def get_ycb_models():
    objects_names = [name for name in os.listdir(ycb_objects.getDataPath()) if 'Ycb' in name] 
    include_objects = []
    for obj_name in objects_names:
        full_name = f'YCB::{obj_name}'
        if full_name not in IGNORE_MODELS:
            include_objects.append(full_name)
    return include_objects

def select_objects(ycb_objects, sn_objects, datasets, n):
    all_objects = []
    if 'YCB' in datasets:
        all_objects += copy.deepcopy(ycb_objects)
    if 'ShapeNet' in datasets:
        all_objects += copy.deepcopy(sn_objects)
    
    chosen_objects = np.random.choice(all_objects, n, replace=False)

    for obj in chosen_objects:
        if obj.split('::')[0] == 'YCB':
            ycb_objects.remove(obj)
        else:
            sn_objects.remove(obj)

    return chosen_objects


OBJECTS_LIST_DIR = 'learning/data/grasping/object_lists'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-objects-fname', type=str, required=True)
    parser.add_argument('--test-objects-fname', type=str, required=True)
    parser.add_argument('--train-objects-datasets', nargs='+', required=True)
    parser.add_argument('--test-objects-datasets', nargs='+', required=True)
    parser.add_argument('--n-train', type=int, required=True)
    parser.add_argument('--n-test', type=int, required=True)
    args = parser.parse_args()
    print(args)

    assert(len(args.train_objects_datasets) > 0)
    assert(len(args.test_objects_datasets) > 0)
    for dataset in args.train_objects_datasets + args.test_objects_datasets:
        print(dataset)
        assert(dataset in ['ShapeNet', 'YCB'])

    all_ycb_objects = get_ycb_models()
    all_shapenet_objects = get_shapenet_models('/home/mnosew/workspace/object_models/shapenet-sem/urdfs')

    # Remove objects from lists as you go.
    train_objects = select_objects(all_ycb_objects, all_shapenet_objects, args.train_objects_datasets, args.n_train)
    test_objects = select_objects(all_ycb_objects, all_shapenet_objects, args.test_objects_datasets, args.n_test)
    
    with open(os.path.join(OBJECTS_LIST_DIR, args.train_objects_fname), 'w') as handle:
        handle.write('\n'.join(train_objects))
    with open(os.path.join(OBJECTS_LIST_DIR, args.test_objects_fname), 'w') as handle:
        handle.write('\n'.join(test_objects))





