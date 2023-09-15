from mimetypes import init
import numpy as np
import os
import pb_robot
import pybullet as p

from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from pb_robot.planners.antipodalGraspPlanner import (
    GraspableBody,
    GraspSampler,
    GraspSimulationClient,
    GraspStabilityChecker
)


class GraspingAgent:

    def __init__(self, object_name, object_properties, init_pose=None, use_gui=False):
        # TODO: Adapt to also work with boxes.
        self.primitive_root = os.path.join(os.environ['PRIMITIVE_ROOT'], 'urdfs')
        self.shapenet_root = os.path.join(os.environ['SHAPENET_ROOT'], 'urdfs')

        self.client_id = pb_robot.utils.connect(use_gui=use_gui)
        pb_robot.utils.set_pbrobot_clientid(self.client_id)

        self.robot = pb_robot.panda.Panda()
        self.robot.hand.Open()

        floor_file = 'models/short_floor.urdf'
        self.floor = pb_robot.body.createBody(floor_file)

        self.graspable_body = graspablebody_from_vector(object_name, object_properties)

        object_dataset, object_identifier = self.graspable_body.object_name.split('::')
        if 'ShapeNet' in object_dataset:
            src_path = os.path.join(self.shapenet_root, f'{object_identifier}.urdf')
        else:
            src_path = os.path.join(self.primitive_root, f'{object_identifier}.urdf')
        self.pb_body = pb_robot.body.createBody(src_path)

        if init_pose is None:
            init_pose = self.sample_initial_pose()

        self.set_object_pose(init_pose, find_stable_z=True)

    def sample_initial_pose(self, n_grasp_attempts=10):
        """
        Return a pose such that at least one grasp is valid.
        """
        labeler = GraspStabilityChecker(
            self.graspable_body,
            stability_direction='all',
            label_type='relpose',
            recompute_inertia=True,
            show_pybullet=False
        )
        pose_attempt = 1
        while True:
            print(f'[TAMP] Attempt {pose_attempt} at finding initial pose...')
            pose_attempt += 1

            # Sample random pose.
            x, y, quat = self._sample_random_xy_quat()
            init_pose = self.set_object_pose(
                pose=((x, y, 1.0), quat),
                find_stable_z=True
            )

            for gx in range(n_grasp_attempts):
                print(f'[TAMP] Grasp #{gx+1}...')
                # TODO: Sample random grasp at pose.
                grasp = self._sample_grasp_action()
                label = labeler.get_label(grasp)
                return init_pose
                # TODO: Check if the grasp is stable.
                if label:
                    labeler.disconnect()
                    print('[TAMP] Found initial pose.')
                    return init_pose

    def disconnect(self):
        p.disconnect(self.client_id)

    def set_object_pose(self, pose, find_stable_z=True):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        self.pb_body.set_base_link_pose(pose)
        if find_stable_z:
            # TODO: Figure out why there is extra space here.
            z = pb_robot.placements.stable_z(self.pb_body, self.floor)
            pos, orn = pose
            pose = ((pos[0], pos[1], z), orn)
            self.pb_body.set_base_link_pose(pose)
        return pose

    def execute_first_action(self, plan, wait=False):
        """
        Execute the first action in the plan.
        :return: True or False based on whether the grasp succeeded.
        """
        grasp, place = plan[0], plan[1]
        # Check if the  grasp is stable.
        labeler = GraspStabilityChecker(
            self.graspable_body,
            stability_direction='all',
            label_type='relpose',
            recompute_inertia=True,
            show_pybullet=False
        )
        label = labeler.get_label(grasp)
        labeler.disconnect()
        # If grasp is stable, change pose of object.
        print(f'[TAMP] Executing grasp... label={label}')
        # if label:
        #     self.set_object_pose(
        #         pose=place,
        #         find_stable_z=False
        #     )
        if wait:
            input('Continue?')
        return label

    def sample_plan(self, horizon, wait=False):
        init_world = self._save_world()
        plan = []
        for hx in range(horizon):
            grasp = self._sample_grasp_action()
            plan.append(grasp)
            if wait:
                input('Next action?')
            placement = self._sample_place_action(grasp)
            plan.append(placement)
            if wait:
                input('Next action?')

        self._restore_world(init_world)
        return plan

    def _save_world(self):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        robot_conf = self.robot.arm.GetJointValues()
        object_pose = self.pb_body.get_base_link_pose()
        return (robot_conf, object_pose)

    def _restore_world(self, world):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        robot_conf, object_pose = world
        self.robot.arm.SetJointValues(robot_conf)
        self.pb_body.set_base_link_pose(object_pose)

    def _sample_grasp_action(self, force_range=(5, 20), max_attempts=1000):
        for ax in range(0, max_attempts):
            # Step (1): Sample valid antipodal grasp for object.
            sampler = GraspSampler(
                graspable_body=self.graspable_body,
                antipodal_tolerance=10,
                show_pybullet=False
            )
            # if ax < 10: print('Connected to:', sampler.sim_client.pb_client_id)
            grasp = sampler.sample_grasps(
                num_grasps=1,
                force_range=force_range
            )[0]
            sampler.disconnect()

            # Step (2): Calculate world transform of gripper.
            pb_robot.utils.set_pbrobot_clientid(self.client_id)
            ee_obj = grasp.ee_relpose
            obj_pose = self.pb_body.get_base_link_pose()

            ee_world = pb_robot.geometry.multiply(obj_pose, ee_obj)
            #pb_robot.viz.draw_point(ee_world[0])

            # Step (3): Compute IK.
            ee_world_tform = pb_robot.geometry.tform_from_pose(ee_world)
            q_grasp = self.robot.arm.ComputeIK(ee_world_tform)
            if q_grasp is None:
                continue
            if not self.robot.arm.IsCollisionFree(q_grasp, obstacles=[self.floor]):
                continue
            self.robot.arm.SetJointValues(q_grasp)

            #sampler.sim_client.tm_show_grasp(grasp)
            # sampler.disconnect()
            return grasp

    def _sample_random_xy_quat(self):
        placement_x_range = (0.5, 0.7)
        placement_y_range = (0.0, 0.4)

         # Step (1): Sample x, y location on table.
        x = np.random.uniform(*placement_x_range)
        y = np.random.uniform(*placement_y_range)

        # Step (2): Sample object rotation.
        quat = pb_robot.transformations.random_quaternion()

        return x, y, quat

    def _sample_place_action(self, grasp, max_attempts=1000):
        pb_robot.utils.set_pbrobot_clientid(self.client_id)
        for ax in range(0, max_attempts):
            x, y, quat = self._sample_random_xy_quat()

            self.pb_body.set_base_link_pose(((x, y, 1.0), quat))

            client = GraspSimulationClient(self.graspable_body, False)
            # pb_aabb = p.getAABB(self.pb_body.id, physicsClientId=self.client_id)
            # print(pb_aabb)
            # pb_robot.utils.set_pbrobot_clientid(self.client_id)
            # pb_robot.viz.draw_aabb(pb_aabb)
            aabb = client.tm_get_aabb(((x, y, 1.0), quat))
            floor_aabb = p.getAABB(self.floor.id, physicsClientId=self.client_id)
            center = pb_robot.aabb.get_aabb_center(aabb)
            extent = pb_robot.aabb.get_aabb_extent(aabb)
            client.disconnect()
            pb_robot.utils.set_pbrobot_clientid(self.client_id)

            z = (floor_aabb[1] + extent/2 + (self.pb_body.get_base_link_point() - center))[2]

            # z = pb_robot.placements.stable_z(self.pb_body, self.floor)
            self.pb_body.set_base_link_pose(((x, y, z), quat))

            # Step (3): Compute world frame of gripper with given grasp.
            ee_obj = grasp.ee_relpose
            obj_pose = self.pb_body.get_base_link_pose()

            ee_world = pb_robot.geometry.multiply(obj_pose, ee_obj)
            # pb_robot.viz.draw_point(ee_world[0])

            # Step (4): Check IK.
            ee_world_tform = pb_robot.geometry.tform_from_pose(ee_world)
            q_grasp = self.robot.arm.ComputeIK(ee_world_tform)
            if q_grasp is None:
                continue
            if not self.robot.arm.IsCollisionFree(q_grasp, obstacles=[self.floor]):
                continue
            self.robot.arm.SetJointValues(q_grasp)
            return ((x, y, z), quat)


if __name__ == '__main__':
    object_name = 'ShapeNet::Desk_fe2a9f23035580ce239883c5795189ed'
    # object_name = 'ShapeNet::ComputerMouse_379e93edfd0cb9e4cc034c03c3eb69d'
    #object_name = 'ShapeNet::WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d'
    # object_name = 'ShapeNet::WallLamp_8be32f90153eb7281b30b67ce787d4d3'    
    object_name = 'ShapeNet::PersonStanding_3dfe62d56a28f464c17f8c1c27c3df1'
    object_name = 'Primitive::Box_553711050557875456'
    #object_name = 'ShapeNet::DrinkingUtensil_c491d8bc77f4fb1ca4c322790a683350'
    graspable_body = GraspableBody(object_name=object_name, com=(0, 0, 0), mass=0.1, friction=1.0)

    agent = GraspingAgent(
        object_name=object_name,
        object_properties=np.array([0.0, 0.0, 0.0, 0.3, 0.1]),
        # init_pose=((0.4, 0., 1.), (0, 0, 0, 1)),
        use_gui=True
    )
    plan = agent.sample_plan(horizon=10, wait=False)
    for ix in range(10):
        agent.execute_first_action((plan[2*ix], plan[2*ix+1]), wait=True)

    pb_robot.utils.wait_for_user()