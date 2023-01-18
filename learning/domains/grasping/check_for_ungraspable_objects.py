import argparse
import os
import pickle
from learning.domains.grasping.generate_datasets_for_experiment import parse_ignore_file, get_object_list
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector, sample_grasp_X

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def check_grasp_gen(objects_fname, object_ix, phase):
    """ Generate a dataset of grasps and labels for a given object file. """

    with open(objects_fname, 'rb') as handle:
        object_data = pickle.load(handle)['object_data']

    object_name = object_data['object_names'][object_ix]
    property_vector = object_data['object_properties'][object_ix]

    graspable_body = graspablebody_from_vector(object_name, property_vector)
    # print(f'Grasping {phase} object {object_ix}: {object_name}...')
    # Sample random grasps with labels.
    try:
        with time_limit(2):
            sample_grasp_X(graspable_body, property_vector, 100, (None, None, None, None))
    except TimeoutException:
        print(f'{phase},{object_ix}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root-name', type=str, required=True)
    parser.add_argument('--start-train', type=int, default=-1)
    parser.add_argument('--start-test', type=int, default=-1)
    new_args = parser.parse_args()
    
    data_root_path = os.path.join('learning/data/grasping', new_args.data_root_name)
    args_path = os.path.join(data_root_path, 'args.pkl')
    with open(args_path, 'rb') as handle:
        args = pickle.load(handle)

    objects_path = os.path.join(data_root_path, 'objects')
    train_objects_path = os.path.join(objects_path, 'train_geo_train_props.pkl')
    test_objects_path = os.path.join(objects_path, 'test_geo_test_props.pkl')

    train_objects = get_object_list(args.train_objects_fname)
    test_objects = get_object_list(args.test_objects_fname)
    TRAIN_IGNORE, TEST_IGNORE = parse_ignore_file(os.path.join(data_root_path, 'ignore.txt'))

    for ox in range(max(0, new_args.start_train), len(train_objects)*args.n_property_samples_train):
        if new_args.start_test > -1:
            break
        if ox in TRAIN_IGNORE:
            continue
        check_grasp_gen(train_objects_path, ox, 'train')

    for ox in range(max(0, new_args.start_test), len(test_objects)*args.n_property_samples_test):
        if ox in TEST_IGNORE:
            continue
        check_grasp_gen(test_objects_path, ox, 'test')
