import IPython 
import numpy
import os
import copy
import pb_robot
import rut
import pybullet as p
import time

import pybullet_data
import pb_robot.helper

def testGrasping(robot):
    # pb_robot.helper.getDirectory()hjh
    # objects_path = pb_robot.helper.getDirectory()
    # pybullet_data_path = pybullet_data.getDataPath()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    tool = pb_robot.body.createBody("sphere_small.urdf")
    tool.set_point([0.4, 0, 0.25])
    grasp_tsr = pb_robot.tsrs.panda_tool_handle.handle_grasp(tool)
    grasp_worldF = rut.util.SampleTSRForPose(grasp_tsr)
    grasp_toolF = numpy.dot(numpy.linalg.inv(tool.get_transform()), grasp_worldF)

    robot.arm.SetJointValues(robot.arm.ComputeIK(grasp_worldF))
    finger_joint = robot.joints_from_names(['panda_finger_joint1', 'panda_finger_joint2'])
    finger_joint_ids = [j.jointID for j in finger_joint]

    qnew = robot.arm.GetJointValues()
    qnew[0] += 0.3
    # raw_input("Grasp?")

    i = 0
    while True:
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py
        # https://github.com/matpalm/tcn_grasping/blob/master/kuka_env.py
        # https://github.com/matpalm/tcn_grasping/blob/master/run_random_grasps.py
        # https://github.com/bulletphysics/bullet3/issues/1936
        p.setJointMotorControlArray(robot.id, finger_joint_ids, controlMode=p.POSITION_CONTROL, targetPositions=[0.005, 0.005], forces=[30, 35]) #, targetVelocities=[0, 0], positionGains=[0.1, 0.1], velocityGains=[1, 1])
   
        
        if i > 5000:
            p.setJointMotorControlArray(robot.id, robot.arm.jointsID, controlMode=p.POSITION_CONTROL, targetPositions=qnew, targetVelocities=[0]*7, forces=robot.arm.torque_limits, positionGains=[0.1]*7, velocityGains=[1]*7)

        p.stepSimulation()
        time.sleep(0.01) 
        i += 1

def turningPath_nut(nut_pose, sample_count=12, turn_count=0.75):
    # Want spin nut about z axis (blue). 

    number_turns = int(sample_count*turn_count)
    rotz_quat = pb_robot.geometry.quat_from_axis_angle([0, 0, -1], (numpy.pi*(2.0/sample_count)))
    rotz = pb_robot.geometry.tform_from_pose(((0, 0, 0), rotz_quat))

    nut_path = numpy.zeros((number_turns+1, 4, 4))
    nut_path[0, :, :] = nut_pose
    for i in range(number_turns):
        nut_pose = numpy.dot(nut_pose, rotz)
        nut_path[i+1, :, :] = nut_pose
    return nut_path

def testWrenchPath(robot):
    part_pose = numpy.eye(4)
    part_pose[0:3, 3] = [0.4, 0, 0.25] 
    grasp_objF = numpy.array([[ 1.        ,  0.        ,  0.        , -0.03729338],
       [ 0.        , -1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        , -1.        ,  0.098     ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    nut_path = turningPath_nut(part_pose)
    hand_path = rut.util.actionPath_hand(nut_path, grasp_objF)

    for i in range(len(hand_path)):
        pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(hand_path[i]))

    # Pick an IK 
    while True:
        startq = robot.arm.ComputeIK(hand_path[0])
        prev_ik = copy.deepcopy(startq)
        for i in range(1, len(hand_path)):
            prev_ik = robot.arm.ComputeIK(hand_path[i], seed_q=prev_ik)
            if prev_ik is None: continue
        break

    robot.arm.SetJointValues(startq) 
    # raw_input("Go?")
    robot.arm.control.cartImpedancePath(hand_path, [9000]*6)    
    return hand_path



if __name__ == '__main__':
    p.connect(p.GUI)
    p.loadURDF("/home/urop/code/object_models/box-data/urdfs/Box_Test.urdf")
    time.sleep(1000) 
    p.disconnect() 
   # IPython.embed()