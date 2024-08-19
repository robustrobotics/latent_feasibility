from scipy.spatial.transform import Rotation as R
import numpy as np
import IPython
import pybullet as p
import pybullet_data
import math
import time
import argparse

import os
import pb_robot
from pb_robot.body import Body
import pb_robot.panda_controls



def get_relative_transform(object_transform, object_id_2):
    """
    Calculate the transform (position and orientation) of object_id_2 relative to object_id_1.

    :param object_id_1: ID of the first object (reference object)
    :param object_id_2: ID of the second object (object to transform)
    :return: A tuple (relative_position, relative_orientation) where:
             - relative_position is a 3D vector (x, y, z)
             - relative_orientation is a quaternion (x, y, z, w)
    """

    # Get the world position and orientation of the first object
    pos1 = object_transform[:3, 3]
    orn1 = R.from_matrix(object_transform[:3, :3]).as_quat() 

    
    # Get the world position and orientation of the second object
    pos2, orn2 = p.getBasePositionAndOrientation(object_id_2)
    
    # Convert orientations to rotation matrices
    rot_matrix_1 = R.from_quat(orn1).as_matrix()
    rot_matrix_2 = R.from_quat(orn2).as_matrix()
    
    # Calculate the relative position
    relative_position = np.dot(rot_matrix_1.T, np.array(pos2) - np.array(pos1))
    
    # Calculate the relative orientation
    relative_rot_matrix = np.dot(rot_matrix_1.T, rot_matrix_2)
    # relative_orientation = R.from_matrix(relative_rot_matrix).as_quat()

    ret = np.eye(4)
    ret[:3, :3] = relative_rot_matrix
    ret[:3, 3] = relative_position

    
    return ret

def is_fallen(orientation_matrix, threshold=30): 
    """
    Check if the object has fallen based on the orientation matrix.
    """
    euler_angles = R.from_matrix(orientation_matrix).as_euler('xyz', degrees=True) 
    if abs(euler_angles[0]) > threshold or abs(euler_angles[1]) > threshold: 
        return True
    return False

def run_sim(object_urdf, push_angle, object_angle, push_velocity=0.1, offset=0, gui=False):
    """
    Description: Run a simulation of pushing an object with a robot arm.
    @ Params:
    - object_urdf: The URDF file of the object to push.
    - push_angle: The angle at which to push the object (in radians). 
    - object_angle: The angle at which the object is placed (in radians).
    - push_velocity: The velocity at which to push the object. (Unsure if it's feasible to vary this.)
    - offset: The offset angle to spawn the object. (Unsure if it's feasible to vary this.)
    - gui: Whether to use the GUI for visualization.
    @ Returns:
    - The difference in transformation matrices between the initial and final states.
    - The contact points between the object and the pusher.
    - The initial transformation matrix.
    - Whether the object has fallen. This might not be correctly implmented.
    """
    # print(object_urdf)
    time_per_step = 1.0 / 240 
    ids = pb_robot.utils.connect(use_gui=gui)
    # print(ids)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(time_per_step)
    pb_robot.utils.set_default_camera()
    robot = pb_robot.panda.Panda()
    floor = p.loadURDF("plane.urdf")
    controls = pb_robot.panda_controls.PandaControls(robot.arm)
    Z = 0.11 # TODO: This should be lower, but null space resolution is needed. 
    start_radius = 0.3
    start_pose = [
        start_radius * np.cos(push_angle),
        start_radius * np.sin(push_angle),
        Z,
    ]
    start_orientation = R.from_euler(
        "xyz", [math.pi / 2, -math.pi / 2, push_angle + math.pi / 2]
    ).as_matrix()

    start_matrix = np.eye(4)
    start_matrix[:3, :3] = start_orientation
    start_matrix[:3, 3] = start_pose

    newq = robot.arm.ComputeIK(start_matrix)
    robot.arm.SetJointValues(newq)
    hand = robot.arm.hand
    hand.Close()

    # Dynamics were basically those that were found for GraspNP. 
    p.changeDynamics(
        hand.id,
        1,
        lateralFriction=2.0,
        rollingFriction=0.001,
        spinningFriction=0.001,
        contactStiffness=30000,
        contactDamping=1000,
        physicsClientId=ids,
    )

    # Spawn the object at the given angle, at a distance slightly larger than the robot's reach. 
    object_pose = [
        [0.6 * np.cos(push_angle + offset), 0.6 * np.sin(push_angle + offset), 0.5],
        [0, 0, np.sin(object_angle / 2), np.cos(object_angle / 2)],
    ]
    object_id = p.loadURDF(object_urdf, object_pose[0], object_pose[1])

    start_rot_matrix = R.from_matrix(np.array(start_matrix)[:3, :3]).as_matrix()


    p.changeDynamics(
        object_id,
        -1,
        mass=0.2,
        lateralFriction=0.1,
        spinningFriction=0.0001,
        rollingFriction=0.0001,
    )


    hand.Open()
    push_object_id = p.loadURDF("cube_small.urdf", [0.41*np.cos(push_angle), 0.41*np.sin(push_angle), 0.1], [0, 0, np.sin(push_angle / 2), np.cos(push_angle / 2)], globalScaling=1.2)
    p.changeVisualShape(push_object_id, -1, rgbaColor=[0, 1, 0, 1])

    # This basically magically grabs the object to perfection. 
    push_object_body = Body(push_object_id) 
    robot.arm.Grab(push_object_body, np.linalg.inv(get_relative_transform(robot.arm.GetEETransform(), push_object_id)))
    p.changeDynamics(push_object_id, -1, lateralFriction=1.0, spinningFriction=0.001, rollingFriction=0.001, mass=1.5)
    finger_joint = robot.joints_from_names(['panda_finger_joint1', 'panda_finger_joint2'])
    finger_joint_ids = [j.jointID for j in finger_joint]

    qnew = robot.arm.GetJointValues()
    qnew[0] += 0.3

    for i in range(50):
        p.setJointMotorControlArray(robot.id, finger_joint_ids, controlMode=p.POSITION_CONTROL, targetPositions=[0.005, 0.005], forces=[150, 150])
        p.stepSimulation()

    p.setGravity(0, 0, -9.81)


    for i in range(100):
        p.stepSimulation()

    start_position, start_orientation = p.getBasePositionAndOrientation(object_id)

    # Essentally, do a form of P control on the arm to ensure that it follows the trajectory we want. 
    # TODO: Might want to do some null space resolution to ensure the arm doesn't crash into the ground. 
    def do_iter(index):
        q = robot.arm.GetJointValues()
        # robot.arm.SetJointValues(q)
        pose = robot.arm.ComputeFK(q)
        # hand.Open()
        p.setJointMotorControlArray(robot.id, finger_joint_ids, controlMode=p.POSITION_CONTROL, targetPositions=[0.005, 0.005], forces=[100, 100]) 
        rot_matrix = pose[:3, :3]
        rot_vector = R.from_matrix(rot_matrix).as_rotvec()
        position = pose[:3, 3]

        jacobian = robot.arm.GetJacobian(q)
        inv_jacobian = np.linalg.pinv(jacobian)
        rotation_error = start_rot_matrix @ np.linalg.inv(rot_matrix)
        rotation_error_vector = R.from_matrix(rotation_error).as_rotvec()

        push_force = push_velocity * np.cos(push_angle)
        push_force_y = push_velocity * np.sin(push_angle)

        expected_position = (index * time_per_step) * np.array(
            [push_force, push_force_y, 0]
        ) + np.array(
            [start_radius * np.cos(push_angle), start_radius * np.sin(push_angle), Z]
        )

        # 5 was chosen arbitrarily as a scaling factor
        target_velocity = np.array(
            [
                push_force + (expected_position[0] - position[0]) / time_per_step / 5,
                push_force_y + (expected_position[1] - position[1]) / time_per_step / 5,
                -(position[2] - Z) / time_per_step / 5,
                rotation_error_vector[0] / time_per_step / 5,
                rotation_error_vector[1] / time_per_step / 5,
                rotation_error_vector[2] / time_per_step / 5,
            ]
        )
        
        # Should read Russ's notes to figure out how this actually works.
        # Right now it is basically a black box, on Seiji's advice. 
        joint_velocity = inv_jacobian @ target_velocity

        for j in range(7):
            p.setJointMotorControl2(
                robot.arm.bodyID,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=joint_velocity[j],
            )
        p.stepSimulation()
        if gui:
            time.sleep(1.0 / 240)


    first_contact = None 
    index_of_contact = None
    for i in range(3 * 240):
        # Check for contact between the robot's hand and the object
        # print(push_object_body.get_pose())
        if i % 10:
            contact_points = p.getContactPoints(bodyA=push_object_id, bodyB=object_id)

            if contact_points: 
                index_of_contact = i
                cube_pos, cube_orn = p.getBasePositionAndOrientation(object_id)
                cube_orn_at_contact = cube_orn

                contact_normals = []
                contact_pts = []

                cube_inv_pos, cube_inv_orn = p.invertTransform(cube_pos, cube_orn)
                for contact in contact_points:
                    world_pos = contact[5]

                    world_normal = contact[7]
                    local_normal = p.multiplyTransforms(
                        [0, 0, 0], cube_inv_orn, world_normal, [1, 0, 0, 1]
                    )[0]
                    contact_normals.append(local_normal)

                    local_pos = p.multiplyTransforms(
                        cube_inv_pos, cube_inv_orn, world_pos, [0, 0, 0, 1]
                    )[0]
                    contact_pts.append(local_pos)

                first_contact = [np.mean(contact_pts, axis=0), np.sum(contact_normals, axis=0)] 
                break


        do_iter(i)

    if first_contact is None: 
        p.disconnect()
        return None, None, None, None
    num_iters = 0 

    for i in range(3 * 240): 
        if i % 10 == 0:
            contact_points = p.getContactPoints(bodyA=push_object_id, bodyB=object_id)
            if not contact_points:
                num_iters += 1 
                if num_iters == 8: # No contact for 1/3rd of a second 
                    break
            else: 
                num_iters = 0

        do_iter(i + index_of_contact) 

    final_position, final_orientation = p.getBasePositionAndOrientation(object_id) 

    #  transformation @ inital = final 

    def get_transformation_matrix(position, orientation):
        ret = np.eye(4)
        ret[:3, 3] = position
        ret[:3, :3] = R.from_quat(orientation).as_matrix()
        return ret

    initial = get_transformation_matrix(start_position, start_orientation)
    final = get_transformation_matrix(final_position, final_orientation)
    fallen = is_fallen(final[:3, :3])
    difference = final @ np.linalg.inv(initial) 

    p.disconnect()

    return difference, first_contact, initial, fallen


def find_contact_point_and_check_push(
    object_name,
    angle_degrees,
    push_velocity,
    object_mass,
    object_friction,
    com_shift,
    translation,
    gui=False,
    logging=False,
):
    """
    THIS IS OLD NOW. 

    Run the simulation given a specific amount of parameters.
    Returns the tuple with the contact point, whether the push was "successful", and the trajectory.

    However, this uses a ball as a pusher, which isn't overwhelmingly realistic. 
    Check "run_sim" for a more realistic simulation. 
    """

    # print("ANGLE DEGREES: ", angle_degrees)

    angle_radians = math.radians(angle_degrees)

    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # print(p.getPhysicsEngineParameters())
    # fixedTimeStep: 1/240

    plane_id = p.loadURDF("plane.urdf")
    cube_id = p.loadURDF(object_name, [0, 0, 0.1], globalScaling=1)
    sphere_id = p.loadURDF(
        "sphere_small.urdf",
        [0.6 * math.cos(angle_radians), 0.6 * math.sin(angle_radians), 0.1],
        globalScaling=1.5,
    )

    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    p.changeDynamics(cube_id, -1, mass=object_mass, lateralFriction=object_friction)
    # p.changeDynamics(cube_id, -1, mass=object_mass, lateralFriction=object_friction)
    # p.changeDynamics(sphere_id, -1, mass=object_mass, lateralFriction=object_friction)
    p.changeDynamics(
        sphere_id,
        -1,
        lateralFriction=1.0,
        mass=3.5,
        spinningFriction=0.0001,
        rollingFriction=0.0001,
    )
    # p.changeDynamics(sphere_id, -1, lateralFriction=0.3, mass=1.5, spinningFriction=0.01, rollingFriction=0.01)

    # cube_pos = [
    #     cube_pos[0] + com_shift[0] + translation[0],
    #     cube_pos[1] + com_shift[1] + translation[1],
    #     cube_pos[2] + com_shift[2] + translation[2],
    # ]
    # p.resetBasePositionAndOrientation(cube_id, cube_pos, cube_orn)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    initial_distance = calculate_distance(cube_id, sphere_id)
    # print(f"Initial Distance: {initial_distance}")
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=30,
            cameraPitch=52,
            cameraTargetPosition=[0, 0, 0],
        )

    force_magnitude = 1
    force_x = -force_magnitude * math.cos(angle_radians)
    force_y = -force_magnitude * math.sin(angle_radians)

    SETTLING_STEPS = 3000
    SIMULATION_STEPS_CONTACT = 4800
    SIMULATION_STEPS = 1200
    SUCCESS_DISTANCE_THRESHOLD = 0.5  # Maximum distance to consider push successful

    for _ in range(SETTLING_STEPS):
        p.stepSimulation()
    # print(p.getBasePositionAndOrientation(cube_id))

    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    p.resetBasePositionAndOrientation(cube_id, [0, 0, cube_pos[2]], cube_orn)
    sphere_pos, sphere_orn = p.getBasePositionAndOrientation(sphere_id)
    # p.resetBasePositionAndOrientation(sphere_id, [sphere_pos[0], sphere_pos[1], sphere_pos[2] + 0.05], sphere_orn)
    # push_velocity /= 2.5
    contact_points = []
    contact_normals = []
    positions = []


    cube_orn_at_contact = None
    for step in range(SIMULATION_STEPS_CONTACT):
        p.resetBaseVelocity(
            sphere_id,
            linearVelocity=[push_velocity * force_x, push_velocity * force_y, 0],
        )
        p.stepSimulation()

        step_contacts = p.getContactPoints(bodyA=cube_id, bodyB=sphere_id)
        if step_contacts:
            # print(step)
            cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
            cube_orn_at_contact = cube_orn
            cube_inv_pos, cube_inv_orn = p.invertTransform(cube_pos, cube_orn)
            for contact in step_contacts:
                world_pos = contact[5]

                world_normal = contact[7]
                local_normal = p.multiplyTransforms(
                    [0, 0, 0], cube_inv_orn, world_normal, [1, 0, 0, 1]
                )[0]
                contact_normals.append(local_normal)

                local_pos = p.multiplyTransforms(
                    cube_inv_pos, cube_inv_orn, world_pos, [0, 0, 0, 1]
                )[0]
                # print(local_pos)
                contact_points.append(local_pos)
            break

        # you want orientation of the block at the time of contact

        if gui:
            time.sleep(0.0001)
    # print(p.getBasePositionAndOrientation(cube_id))
    if contact_points == []:
        # print("No contact point found.")
        p.disconnect()
        return None, None, None, None, None

    for step in range(SIMULATION_STEPS):
        if logging and step % 24 == 0:
            cube_pos, cube_orientation = p.getBasePositionAndOrientation(cube_id)
            cube_both = []
            for x in cube_pos:
                cube_both.append(x)
            for x in cube_orientation:
                cube_both.append(x)
            positions.append(tuple(cube_both))
        p.resetBaseVelocity(
            sphere_id,
            linearVelocity=[push_velocity * force_x, push_velocity * force_y, 0],
        )
        p.stepSimulation()

        if gui:
            time.sleep(0.001)

    if gui:
        time.sleep(3)

    final_distance = calculate_distance(cube_id, sphere_id)
    # print(f"Final Distance: {final_distance}")

    p.disconnect()

    push_successful = final_distance <= SUCCESS_DISTANCE_THRESHOLD

    # print(contact_points)
    if contact_points:
        # Average contact point
        avg_contact_point = [
            sum(c[i] for c in contact_points) / len(contact_points) for i in range(3)
        ]
        sum_normals = [sum(c[i] for c in contact_normals) for i in range(3)]
        return (
            avg_contact_point,
            sum_normals,
            cube_orn_at_contact,
            push_successful,
            positions if logging else None,
        )
    else:
        raise Exception("I don't think this case matters anymore")
        return None, None, push_successful, positions if logging else None


def calculate_distance(bodyA, bodyB):
    posA, _ = p.getBasePositionAndOrientation(bodyA)
    posB, _ = p.getBasePositionAndOrientation(bodyB)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(posA, posB)))


def main():
    run_sim("../object_models/shapenet-sem-v2/tmp_urdfs/Primitive::Box_511914985260486656_0.231m_0.097f_0.016cx_0.045cy_-0.127cz.urdf", 0.830488760921489, 1.774212669402351, 0.08895227139064858, 0.0207827933289618, True)
    return

    parser = argparse.ArgumentParser(
        description="Simulate object pushing and evaluate success."
    )
    parser.add_argument(
        "--angle", type=float, default=90, help="Angle of push in degrees"
    )
    parser.add_argument(
        "--mass", type=float, default=0.00001, help="Mass of the objects"
    )
    parser.add_argument(
        "--friction", type=float, default=0.5, help="Friction of the objects"
    )
    parser.add_argument("--velocity", type=float, default=2, help="Push velocity")
    parser.add_argument("--gui", action="store_true", help="Use GUI for visualization")
    parser.add_argument("--logging", action="store_true", help="Log object positions")
    args = parser.parse_args()

    translation = [0, 0, 0]
    com_shift = [0, 0, 0]

    try:
        contact_point, push_successful, positions = find_contact_point_and_check_push(
            "Primitive::Box_3161924807452160_0.188m_0.274f_-0.012cx_0.035cy_-0.011cz.urdf",
            args.angle,
            args.velocity,
            args.mass,
            args.friction,
            com_shift,
            translation,
            args.gui,
            args.logging,
        )

        if contact_point:
            print(f"Contact Point at {args.angle} degrees: {contact_point}")
        else:
            print(f"No contact point found at {args.angle} degrees")

        print(
            "The push was successful."
            if push_successful
            else "The push was not successful."
        )

        if args.logging and positions:
            print("Object positions logged:")
            for pos in positions:
                print(pos)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":

    # run_sim("sphere_small.urdf", 0, 0)
    # object_urdf = "cube_small.urdf"  # Replace with your object's URDF file
    # object_pose = [[0.5, 0, 0], [0, 0, 0, 1]]  # Replace with your object's pose
    # grasp_object(object_urdf, object_pose)
    main()
