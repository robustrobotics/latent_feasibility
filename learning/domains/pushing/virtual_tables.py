"""
A way to simulate different virtual tables without creating meshes in Pybullet.
"""

from typing import Callable, List
from datetime import datetime
import copy

import matplotlib.pyplot as plt
import numpy as np

# What's left to do:
# - [ ] Visualize box along trajectory to track orientation (change color for resting poses)
# - [ ] Visualize COM.
# - [ ] Create specific table instances.


class SimulatedTable(object):
    """
    This class makes a quasistatic assumption and an object 'smallness' assumption, 
    e.g. if the COM of the block is 'off the table', then the block is considered to have 
    'fallen,' regardless of block geometry.
    """

    def __init__(self,
                 block_com_relative_to_centroid: np.ndarray,
                 is_on_table: Callable[[np.ndarray], float],
                 start_pose: np.ndarray,
                 start_ori: float,
                 goal_pos: np.ndarray):
        self.block_com = block_com_relative_to_centroid
        self.is_on_table_fn = is_on_table
        self.init_pose = np.array([
            [np.cos(start_ori), -np.sin(start_ori), start_pose[0]],
            [np.sin(start_ori),  np.cos(start_ori), start_pose[1]],
            [0.0,                0.0,               1.0]
        ])

        self.pose = copy.deepcopy(self.init_pose)
        self.goal_pos = goal_pos

        self.full_traj = []
        self.rest_poses = []

    def apply_push_trajectory(self, pose_traj: List[np.ndarray]) -> bool:
        """
            traj (List[np.ndarray]): a timestepped sequence of object centroid _not_ COM, relative to the starting pose
                                    of the block. Pose is represnted by 3x3 SE(2) matrices.

                                    The internal state of the block will be updated to travel along this trajectory,
                                     relative to the present pose of the block.
                                     We assume a linear interpolation between provided trajectory points.

            Returns (bool): Whether or not the block had fallen over the course of the trajectory.
        """

        # first, compute what would be the trajectory of the block
        sim_pose_traj = np.array([self.pose @ _Tp for _Tp in pose_traj])

        self.rest_poses.append(sim_pose_traj[0])

        # add to full recorded trajectory and check for block falls
        for _Tp in sim_pose_traj:
            self.full_traj.append(_Tp)

            if self.is_on_table_fn(_Tp[:2, 2] + self.block_com) < 0.0:
                return True  # return True, since the block fell

        # Block has not fallen, update pose and return False
        self.pose = sim_pose_traj[-1]
        return False

    def get_distance_to_goal(self) -> float:
        return np.linalg.norm(self.goal_pos - self.pose[:2, 2])

    def reset(self):
        self.pose = copy.deepcopy(self.init_pose)
        self.full_traj = []
        self.rest_poses = []

    def visualize(self, show: bool = False, min_x=0.0, max_x=2.0, min_y=0.0, max_y=2.0, res=0.01):
        """
            show (bool): whether to use `maptolitlib.pyplot.show`. If se to `False` (default), then
                         save an instance of the figure in the current directory 
                         (named with current timestamp).
        """
        _, ax = plt.subplots()

        # first, create a plot of the table
        x = np.arange(min_x, max_x + res, res)
        y = np.arange(min_y, max_y + res, res)
        X, Y = np.meshgrid(x, y)
        XY = np.stack((X, Y), axis=-1)

        # plot a filled contour that visualizes table
        Z_on_table = self.is_on_table_fn(XY)
        ax.contourf(X, Y, Z_on_table, [0.0, 1.0], zorder=-1)

        # plot the full trajectory taken by the block
        if len(self.full_traj) > 0:
            _full_traj = np.array(self.full_traj)
            full_traj_x = _full_traj[:, 0, 2]
            full_traj_y = _full_traj[:, 1, 2]
            ax.plot(full_traj_x, full_traj_y, color='b', linewidth=2.5, zorder=1)

        # plot the resting positions of the block
        if len(self.rest_poses) > 0:
            _rest_poses = np.array(self.rest_poses)
            rest_pos_x = _rest_poses[:, 0, 2]
            rest_pos_y = _rest_poses[:, 1, 2]
            ax.scatter(rest_pos_x, rest_pos_y, color='r', s=8, zorder=2)

        ax.scatter([self.goal_pos[0]], [self.goal_pos[1]], color='g', zorder=1)


        ax.set_aspect('equal', adjustable='box')

        if show:
            plt.show()
            return

        uid = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        plt.savefig(f'table_{uid}.png')


class RingTable(SimulatedTable):
    def __init__(self,
                 block_com_relative_to_centroid,
                 start_pose,
                 start_ori,
                 goal_pos):

        def is_on_table(xy):
            # if there is a batching dimension...
            return 1.0 - np.linalg.norm(np.ones(2) - xy, axis=-1) 

        super().__init__(block_com_relative_to_centroid,
                         is_on_table, start_pose, start_ori, goal_pos)


if __name__ == '__main__':
    rt = RingTable(np.zeros(2), np.ones(2), 0.0, np.zeros(2))
    rt.visualize(show=True)

    traj = [np.eye(3) for _ in range(0, 10)]
    for i in range(len(traj)):
        traj[i][0, 2] += 0.1 * i

    rt.apply_push_trajectory(traj)
    rt.visualize(show=True)