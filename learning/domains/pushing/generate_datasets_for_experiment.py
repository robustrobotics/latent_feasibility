import math
import tqdm
from functools import partial
import trimesh
import multiprocessing
import argparse
import os
import pickle
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R 

from learning.domains.pushing.find_contact_points import (
    find_contact_point_and_check_push,
    run_sim,
)
from pb_robot.planners.antipodalGraspPlanner import (
    GraspSimulationClient,
    GraspableBody,
    GraspableBodySampler,

)

import pb_robot 
import pybullet as p

# Don't know if these are correct for now, mostly filler
PUSH_VELOCITY_RANGE = (0.08, 0.12)
MASS_RANGE = (0.1, 0.3)
FRICTION_RANGE = (0.05, 0.1)
OFFSET_STD_DEV = 0.025


def get_object_list(fname):
    """Parse object text file."""
    print("Opening", fname)
    with open(fname, "r") as handle:
        objects = handle.readlines()
    return [o.strip() for o in objects if len(o) > 1]

def save_args(new_args):
    """
    Save the arguments for generating the datasets so that it doesn't have to be repeated in the future.
    """
    data_root_path = os.path.join(
        "learning", "data", "pushing", new_args.data_root_name
    )
    args_path = os.path.join(data_root_path, "args.pkl")

    if os.path.exists(data_root_path):
        choice = input(
            "[Warning] Data root directory already exists. Do you want to rewrite this directory? (yes/no)"
        )
    else:
        choice = "yes"
    assert choice == "yes" or choice == "no"
    if choice == "yes":
        with open(args_path, "wb") as handle:
            pickle.dump(new_args, handle)
            args = new_args
    else:
        with open(args_path, "rb") as handle:
            args = pickle.load(handle)
            # args.n_processes = new_args.n_processes
            # args.merge = new_args.merge
    return args


def get_object_lists(args):
    """
    Get the list of relevant objects and return a list for training and testing.
    """
    data_root_path = os.path.join("learning", "data", "pushing", args.data_root_name)
    # objects_path = os.path.join(data_root_path, 'objects')
    # grasps_path = os.path.join(data_root_path, 'grasps')
    # if not os.path.exists(objects_path):
    #     os.mkdir(objects_path)
    #     os.mkdir(grasps_path)

    # The data will be stored in a text-file, and we will just collect all of the objects
    train_objects = get_object_list(
        os.path.join(data_root_path, args.train_objects_fname)
    )
    test_objects = get_object_list(
        os.path.join(data_root_path, args.test_objects_fname)
    )

    return (train_objects, test_objects)


def generate_parameters(name):
    """
    Generate the parameters for an individual object by sampling the center of mass using the
    GraspableBodySampler.
    """
    # Sampled from the same distributions as grasping, might want to change
    # mass = np.random.uniform(*MASS_RANGE)
    # friction = np.random.uniform(*FRICTION_RANGE)

    # mass = (MASS_RANGE[0] + MASS_RANGE[1]) / 2 
    # friction = (FRICTION_RANGE[0] + FRICTION_RANGE[1]) / 2 

    mass = 0.2
    friction = 0.07

    # This function is essentially a black-box for now
    com = GraspableBodySampler._sample_com(name)  # Taking this from GNP

    # Let's only change one dimension of the COM 
    # com[1] = 0 
    com[2] = 0 

    return {"name": name, "mass": mass, "friction": friction, "com": com}


def generate_object_parameters(args):
    """
    Supposed to generate the weight, COM, friction coefficients for the list of objects.
    """
    list_of_objects = []
    # Might need to parallelize this later, unsure
    for obj in args.objects:
        for _ in range(args.number_of_trials):
            list_of_objects.append(generate_parameters(obj))

    for i in range(len(list_of_objects)):
        file_name = os.path.join(args.directory_name, f"{i}.pkl")
        with open(file_name, "wb") as handle:
            # print("HERE")
            pickle.dump(list_of_objects[i], handle)
    print("Done generating", args.directory_name)


def process_single_object(obj_data, args):
    """
    A new way of simulating the pushing action that uses the robot. 
    This is designed to create a simulation that will have more varied results, 
    as well as being more realistic. 
    Args: 
        obj_data: The data for the object that is being simulated.
        args: The arguments for the simulatio, which is inputted into the file.
    Returns:
        data: The data for the object that was simulated.
    """
    body = GraspableBody(
        obj_data["name"], obj_data["com"], obj_data["mass"], obj_data["friction"]
    )

    sim_client = GraspSimulationClient(body, False)
    urdf = sim_client._get_object_urdf(body)
    sim_client.disconnect() 

    data = [] 
    # TODO: Fix cases where the robot acts weirdly 
    # TODO: Check if offset is reasonable
    num_fallen = 0
    for _ in range(args.n_pushes_per_object):
        contact_points = None 
        n_attempts = 0
        while contact_points is None: 
            n_attempts += 1
            if n_attempts > 5:
                # transformation, contact_points, initial = run_sim(urdf, push_angle, object_angle, push_velocity, offset, gui=True) 
                break
            push_angle = np.random.uniform(0, 2 * math.pi) 
            object_angle = np.random.uniform(0, 2 * math.pi) 
            # offset = np.random.normal(loc=0, scale=OFFSET_STD_DEV)
            # push_velocity = np.random.uniform(*PUSH_VELOCITY_RANGE) 
            # print("DEBUG MODE")
            # if _ == 0:
            # push_angle = 0
            # object_angle = 0 
            offset = 0 
            push_velocity = 0.1 

            # print(push_angle, object_angle, push_velocity, offset)
            
            transformation, contact_points, initial, fallen = run_sim(urdf, push_angle, object_angle, push_velocity, offset, gui=args.gui) 
            if contact_points is not None: 
                initial_euler_angles = R.from_matrix(initial[:3,:3]).as_euler('xyz', degrees=False)
                if initial_euler_angles[0] < -0.4 or initial_euler_angles[0] > 0.4 or initial_euler_angles[1] < -0.4 or initial_euler_angles[1] > 0.4:
                    print("This case")
                    contact_points = None 
                    continue
                transformation_ = R.from_matrix(transformation[:3,:3]).as_euler('xyz', degrees=False) 
                if transformation_[0] < -0.4 or transformation_[0] > 0.4 or transformation_[1] < -0.4 or transformation_[1] > 0.4:
                    # print("Flipped over somehow.")
                    contact_points = None 
                    continue    


            # print(obj_data, transformation[:3, 3])
        # print(n_attempts)

        if contact_points is not None:
            # print(R.from_matrix(initial[:3,:3]).as_euler('xyz', degrees=False), object_angle)
            # print(fallen)
            data.append(((push_angle, contact_points[0], contact_points[1], body, push_velocity, initial, object_angle), (transformation, fallen)))
            num_fallen += fallen 
        else: 
            data.append(((None, None, None, None, None, None), (None, None))) 
            num_fallen += 1

    # if num_fallen == 0:
    #     print("No fallen objects for ", obj_data)
    # print(num_fallen, "/", args.n_pushes_per_object)
    print(f'num pybullet instances: {count_num_pybullet_instances(1000)}')
    return data 


def count_num_pybullet_instances(max_counter):

    n_connected_instances = 0

    for _c in range(max_counter):
        n_connected_instances += p.getConnectionInfo(_c)['isConnected']

    return n_connected_instances

def process_single_object_old(obj_data, args):
    """
    Generate the pushing actions for a singular object. This uses a simple pushing action and this function allows
    easy use of multiprocessing for helping the data generation process speed up.

    WARNING: This is old and should no longer really be used.
    """
    # print("????")
    total_success = 0
    total_fail = 0
    # print(obj_data)
    body = GraspableBody(
        obj_data["name"], obj_data["com"], obj_data["mass"], obj_data["friction"]
    )

    sim_client = GraspSimulationClient(body, False)
    urdf = sim_client._get_object_urdf(body)
    

    # run_sim(urdf) 

    # print(urdf)) 

    # sim_client.disconnect()
    # run_sim(urdf, np.random.uniform(2*math.pi), np.random.uniform(2*math.pi))
    # return

    data = []
    for _ in range(args.n_pushes_per_object):
        angle = np.random.uniform(0, 360)
        push_velocity = np.random.uniform(*PUSH_VELOCITY_RANGE)
        offset = [np.random.uniform(*OFFSET_RANGE), np.random.uniform(*OFFSET_RANGE), 0]


        contact_point = None
        amount = 0
        while contact_point is None: 
            amount += 1 
            if amount > 10:
                break
            angle = np.random.uniform(0, 360)
            push_velocity = np.random.uniform(*PUSH_VELOCITY_RANGE)
            # offset = [np.random.uniform(*OFFSET_RANGE), np.random.uniform(*OFFSET_RANGE), 0]
            offset = [0, 0, 0] 
            # print("???")
            # print(args.gui)
            # print(urdf)
            contact_point, normals, cube_orn_at_contact, success, logs = find_contact_point_and_check_push(
                urdf,
                angle,
                push_velocity,
                obj_data["mass"],
                obj_data["friction"],
                obj_data["com"],
                offset,
                logging=True,
                gui=args.gui
            )
        # if contact_point is None: 
        #     print("NO CONTACT POINT") 
        #     print(obj_data) 
        #     find_contact_point_and_check_push(urdf, angle, push_velocity, obj_data["mass"], obj_data["friction"], obj_data["com"], offset, gui=True)
        #     print("?")

        # print("DONE")
        # _, _, logs2 = find_contact_point_and_check_push(urdf, angle, push_velocity, obj_data['mass'], obj_data['friction'], obj_data['com'], offset, logging=True)
        # for i in range(len(logs2)):
        #     if logs[i][0] - logs2[i][0] > 0.1:
        #         print("DIFFERENT")
        # move_on = input('Move on!')
        data.append(((angle, contact_point, normals, body, push_velocity, cube_orn_at_contact), (success, logs)))
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

def init_pool(): 
    np.random.seed() 

def main(args):
    args = save_args(args)
    print(args)
    train_objects, test_objects = get_object_lists(args)
    data_root_path = os.path.join("learning", "data", "pushing", args.data_root_name)
    # print(train_objects)
    # print(test_objects)

    # Basically just setting up these functions to generate the individual parameters for the meshes.
    # Test samegeo basically is a test set that uses the same geometry as the training data (both were used in GNP)
    train_objects_path = os.path.join(data_root_path, "training_data")
    if not os.path.exists(train_objects_path):
        os.mkdir(train_objects_path)
        train_args = SimpleNamespace(
            directory_name=train_objects_path,
            objects=train_objects,
            number_of_trials=args.n_property_samples_train,
        )
        generate_object_parameters(train_args)

    test_objects_path = os.path.join(data_root_path, "test_data")
    if not os.path.exists(test_objects_path):
        os.mkdir(test_objects_path)
        test_args = SimpleNamespace(
            directory_name=test_objects_path,
            objects=test_objects,
            number_of_trials=args.n_property_samples_test,
        )
        generate_object_parameters(test_args)

    test_samegeo_objects_path = os.path.join(data_root_path, "test_samegeo_data")
    if not os.path.exists(test_samegeo_objects_path):
        os.mkdir(test_samegeo_objects_path)
        test_samegeo_args = SimpleNamespace(
            directory_name=test_samegeo_objects_path,
            objects=train_objects,
            number_of_trials=args.n_property_samples_test,
        )
        generate_object_parameters(test_samegeo_args)

    num_processes = 2 if not args.gui else 1 
    # num_processes = 1
    print("Num Processes: ", num_processes)
    pool = multiprocessing.Pool(processes=num_processes, initializer=init_pool)
    process_func = partial(process_single_object, args=args)
    object_data_list = []

    print(len(train_objects), len(test_objects))

    if not os.path.exists(os.path.join(data_root_path, "train_dataset.pkl")):
        for i in range(args.n_property_samples_train * len(train_objects)):
            path = os.path.join(train_objects_path, f"{i}.pkl")
            with open(path, "rb") as handle:
                object_data_list.append(pickle.load(handle))
        # load up train objects make the set
        # num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
        # results = pool.map(process_func, object_data_list)
        with tqdm.tqdm(total=len(object_data_list), desc="Processing objects") as pbar:
            results = []
            for result in pool.imap_unordered(process_func, object_data_list):
                results.append(result)
                pbar.update()

        path = os.path.join(data_root_path, "train_dataset.pkl")
        with open(path, "wb") as handle:
            pickle.dump(results, handle)
    
    if not os.path.exists(os.path.join(data_root_path, "samegeo_test_dataset.pkl")):
        object_data_list = []
        for i in range(args.n_property_samples_test * len(train_objects)):
            path = os.path.join(test_samegeo_objects_path, f"{i}.pkl")
            with open(path, "rb") as handle:
                object_data_list.append(pickle.load(handle))

        with tqdm.tqdm(total=len(object_data_list), desc="Processing objects") as pbar:
            results = []
            for result in pool.imap_unordered(process_func, object_data_list):
                results.append(result)
                pbar.update()
        path = os.path.join(data_root_path, "samegeo_test_dataset.pkl")
        with open(path, "wb") as handle:
            pickle.dump(results, handle)

    # if not os.path.exists(os.path.join(data_root_path, "test_dataset.pkl")):
    object_data_list = []
    for i in range(args.n_property_samples_test * len(test_objects)):
        path = os.path.join(test_objects_path, f"{i}.pkl")
        with open(path, "rb") as handle:
            object_data_list.append(pickle.load(handle))

    with tqdm.tqdm(total=len(object_data_list), desc="Processing objects") as pbar:
        results = []
        for result in pool.imap_unordered(process_func, object_data_list):
            results.append(result)
            pbar.update()
    path = os.path.join(data_root_path, "test_dataset.pkl")
    with open(path, "wb") as handle:
        pickle.dump(results, handle)

    pool.close()
    pool.join()
    # store the set, likely as a huge .pkl
    # if it's __too__ large, then you can split the dataset up into grouped/individual objects.

    # do the same thing as above for evaluation data for the GNP loop.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # define command line arguments
    # Consume object lists
    # parser.add_arguments...

    # TODO: Figure out if all of these arguments are neccesary
    parser.add_argument("--train-objects-fname", type=str, default='train_objects')
    parser.add_argument("--test-objects-fname", type=str, default='test_objects')
    parser.add_argument("--data-root-name", type=str, required=True)
    parser.add_argument("--n-property-samples-train", type=int, default=1)
    parser.add_argument("--n-property-samples-test", type=int, default=1)
    parser.add_argument("--n-pushes-per-object", type=int,default=3)
    parser.add_argument("--gui", action='store_true', default=False)
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
