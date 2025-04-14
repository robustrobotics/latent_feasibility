

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
import tracemalloc

import pb_robot 
import pybullet as p
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
            # object_angle = np.random.uniform(0, 2 * math.pi) 
            # offset = np.random.normal(loc=0, scale=OFFSET_STD_DEV)
            # push_velocity = np.random.uniform(*PUSH_VELOCITY_RANGE) 
            # print("DEBUG MODE")
            # if _ == 0:
            # push_angle = 0
            object_angle = 0 
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
    return data 