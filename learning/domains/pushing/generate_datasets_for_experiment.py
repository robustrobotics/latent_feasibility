import tqdm 
from functools import partial
import trimesh
import multiprocessing
import argparse
import os
import pickle
from types import SimpleNamespace
import numpy as np 
# from learning.domains.pushing.find_contact_points import find_contact_point_and_check_push
from learning.domains.pushing.find_contact_points import find_contact_point_and_check_push
import pb_robot
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody, GraspableBodySampler
import pybullet as p 
import time

# Don't know if these are correct for now, mostly filler
PUSH_VELOCITY_RANGE = (0.2, 0.3)
MASS_RANGE = (0.1, 0.2)  
FRICTION_RANGE = (0.1, 0.3)
OFFSET_RANGE = (-0.02, 0.02)

def generate_push_data_for_obj(obj, config):
    # Call subroutines/helpers to run pybullet and get you your data!

    # store data here.
    obj_data = {
        # this is not definite. just giving you an idea
        'obj': obj,
        'physics': None,
        # other params you need here.
        # list of grasps, outcomes, helpful meshes, you name it.
    }

    return obj_data

def merge_into_gnp_format(list_of_object_sets):
    # merge the object datasets together into the format specified by
    # process_and_save() and process_geometry() in create_gnp_data.py
    return {}

def get_object_list(fname):
    """ Parse object text file. """
    print('Opening', fname)
    with open(fname, 'r') as handle:
        objects = handle.readlines()
    return [o.strip() for o in objects if len(o) > 1]

# From Seiji: 
# Set up data directory.
# Good practice: save the running arguments (either as a .pkl or .json)
# so you can remember your settings! A future you will thank you.
def save_args(new_args): 
    data_root_path = os.path.join('learning', 'data', 'pushing', new_args.data_root_name)
    args_path = os.path.join(data_root_path, 'args.pkl')

    if os.path.exists(data_root_path): 
        choice = input('[Warning] Data root directory already exists. Do you want to rewrite this directory? (yes/no)')
    else:
        choice = 'yes'
    assert choice == 'yes' or choice == 'no' 
    if choice == 'yes': 
        with open(args_path, 'wb') as handle: 
            pickle.dump(new_args, handle)
            args = new_args
    else: 
        with open(args_path, 'rb') as handle: 
            args = pickle.load(handle) 
            # args.n_processes = new_args.n_processes
            # args.merge = new_args.merge
    return args

def get_object_lists(args): 
    data_root_path = os.path.join('learning', 'data', 'pushing', args.data_root_name)
    objects_path = os.path.join(data_root_path, 'objects')
    grasps_path = os.path.join(data_root_path, 'grasps')
    if not os.path.exists(objects_path):
        os.mkdir(objects_path)
        os.mkdir(grasps_path)

    # The data will be stored in a text-file, and we will just collect all of the objects
    train_objects = get_object_list(os.path.join(data_root_path, args.train_objects_fname))
    test_objects = get_object_list(os.path.join(data_root_path, args.test_objects_fname))

    return (train_objects, test_objects) 

def generate_parameters(name):

    # Sampled from the same distributions as grasping, might want to change
    mass = np.random.uniform(*MASS_RANGE) 
    friction = np.random.uniform(*FRICTION_RANGE) 
    
    # This function is essentially a black-box for now
    com = GraspableBodySampler._sample_com(name) # Taking this from GNP 
    # print(name)
    # print(mass) 
    # print(friction) 
    # print(com) 
    return {'name':name, 'mass':mass, 'friction':friction, 'com':com}

"""
Supposed to generate the weight, COM, friction coefficients for the objects 
"""
def generate_object_parameters(args):
    list_of_objects = [] 
    # Might need to parallelize this later, unsure 
    for obj in args.objects: 
        for _ in range(args.number_of_trials): 
            list_of_objects.append(generate_parameters(obj))
    
    for i in range(len(list_of_objects)): 
        file_name = os.path.join(args.directory_name, f"{i}.pkl")
        with open(file_name, 'wb') as handle: 
            # print("HERE")
            pickle.dump(list_of_objects[i], handle)
    print("DONE GENERATING ", args.directory_name)


def process_single_object(obj_data, args):
    # print(obj_data)
    # print("PROCESSING OBJECT ", obj_data['name'])
    total_success = 0
    total_fail = 0
    # print(obj_data)
    body = GraspableBody(obj_data['name'], obj_data['com'], obj_data['mass'], obj_data['friction'])
    sim_client = GraspSimulationClient(body, False)
    urdf = sim_client._get_object_urdf(body)
    sim_client.disconnect()

    data = []
    for _ in range(args.n_pushes_per_object):
        angle = np.random.uniform(0, 360)
        push_velocity = np.random.uniform(*PUSH_VELOCITY_RANGE)
        offset = [np.random.uniform(*OFFSET_RANGE), np.random.uniform(*OFFSET_RANGE), 0]

        contact_point, success, logs = find_contact_point_and_check_push(urdf, angle, push_velocity, obj_data['mass'], obj_data['friction'], obj_data['com'], offset, logging=True)
        # move_on = input('Move on!')
        data.append(((angle, contact_point, body, push_velocity), (success, logs))) 
        # exit()
        # if success: 
        #     find_contact_point_and_check_push(urdf, angle, push_velocity, obj_data['mass'], obj_data['friction'], obj_data['com'], offset, gui=True)
        #     exit()
            # time.sleep(100)
        if success:
            total_success += 1
        else:
            total_fail += 1

    return data

def main(args):
    args = save_args(args) 
    train_objects, test_objects = get_object_lists(args) 
    data_root_path = os.path.join('learning', 'data', 'pushing', args.data_root_name) 
    # print(train_objects) 
    # print(test_objects)


    # Basically just setting up these functions to generate the individual parameters for the meshes. 
    # Test samegeo basically is a test set that uses the same geometry as the training data (both were used in GNP)
    train_objects_path = os.path.join(data_root_path, 'training_data')
    if not os.path.exists(train_objects_path): 
        os.mkdir(train_objects_path)
        train_args = SimpleNamespace(directory_name=train_objects_path, objects=train_objects, number_of_trials=args.n_property_samples_train)
        generate_object_parameters(train_args) 

    test_objects_path = os.path.join(data_root_path, 'test_data')
    if not os.path.exists(test_objects_path): 
        os.mkdir(test_objects_path)
        test_args = SimpleNamespace(directory_name=test_objects_path, objects=test_objects, number_of_trials=args.n_property_samples_test)
        generate_object_parameters(test_args)

    test_samegeo_objects_path = os.path.join(data_root_path, 'test_samegeo_data') 
    if not os.path.exists(test_samegeo_objects_path): 
        os.mkdir(test_samegeo_objects_path)
        test_samegeo_args = SimpleNamespace(directory_name=test_samegeo_objects_path, objects=train_objects, number_of_trials=args.n_property_samples_test)
        generate_object_parameters(test_samegeo_args) 
    

    object_data_list = []
    for i in range(args.n_property_samples_train * len(train_objects)):
        path = os.path.join(train_objects_path, f"{i}.pkl")
        with open(path, 'rb') as handle:
            object_data_list.append(pickle.load(handle))


    # load up train objects make the set

    # num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    num_processes = 1 
    print("Num Processes: ", num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    process_func = partial(process_single_object, args=args)
    # results = pool.map(process_func, object_data_list)
    with tqdm.tqdm(total=len(object_data_list), desc="Processing objects") as pbar:
        results = []
        for result in pool.imap_unordered(process_func, object_data_list):
            results.append(result)
            pbar.update()

    path = os.path.join(data_root_path, 'train_dataset.pkl')
    with open(path, 'wb') as handle:
        pickle.dump(results, handle)

    object_data_list = []
    for i in range(args.n_property_samples_test * len(test_objects)):
        path = os.path.join(test_objects_path, f"{i}.pkl")
        with open(path, 'rb') as handle:
            object_data_list.append(pickle.load(handle))

    # results = pool.map(process_func, object_data_list)
    with tqdm.tqdm(total=len(object_data_list), desc="Processing objects") as pbar:
        results = []
        for result in pool.imap_unordered(process_func, object_data_list):
            results.append(result)
            pbar.update()
    path = os.path.join(data_root_path, 'test_dataset.pkl')
    with open(path, 'wb') as handle:
        pickle.dump(results, handle)
    
    object_data_list = []
    for i in range(args.n_property_samples_test * len(train_objects)):
        path = os.path.join(test_samegeo_objects_path, f"{i}.pkl")
        with open(path, 'rb') as handle:
            object_data_list.append(pickle.load(handle))

    # results = pool.map(process_func, object_data_list)
    with tqdm.tqdm(total=len(object_data_list), desc="Processing objects") as pbar:
        results = []
        for result in pool.imap_unordered(process_func, object_data_list):
            results.append(result)
            pbar.update()
    path = os.path.join(data_root_path, 'samegeo_test_dataset.pkl')
    with open(path, 'wb') as handle:
        pickle.dump(results, handle)
    pool.close()
    pool.join()
    # store the set, likely as a huge .pkl
    # if it's __too__ large, then you can split the dataset up into grouped/individual objects.

    # do the same thing as above for evaluation data for the GNP loop.
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # define command line arguments
    # Consume object lists
    # parser.add_arguments...

    # TODO: Figure out if all of these arguments are neccesary
    parser.add_argument('--train-objects-fname', type=str, required=True)
    parser.add_argument('--test-objects-fname', type=str, required=True)
    parser.add_argument('--data-root-name', type=str, required=True)
    parser.add_argument('--n-property-samples-train', type=int, required=True)
    parser.add_argument('--n-property-samples-test', type=int, required=True)
    parser.add_argument('--n-pushes-per-object', type=int, required=True)
    # parser.add_argument('--n-points-per-object', type=int, required=True)
    # parser.add_argument('--n-fit-grasps', type=int, required=True)
    # parser.add_argument('--grasp-noise', type=float, required=True)
    # parser.add_argument('--curvature-radii', type=float, nargs=3, required=True)  # we choose 3 for feature vector ease
    # parser.add_argument('--n-processes', type=int, default=1)
    # parser.add_argument('--merge', action='store_true', default=True)
    # parse command line arguments
    args = parser.parse_args()


    # A personal practice of mine (you can refuse to go with it) is to define
    # all main script code in a function at the top.
    main(args)
