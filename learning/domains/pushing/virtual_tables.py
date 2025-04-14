"""
A way to simulate different virtual tables without creating meshes in Pybullet.
"""

from typing import Callable, List
from datetime import datetime
import copy

from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np


class SimulatedTable(object):
    """
    This class makes a quasistatic assumption and an object 'smallness' assumption, 
    e.g. if the COM of the block is 'off the table', then the block is considered to have 
    'fallen,' regardless of block geometry.
    """

    def __init__(self,
                 block_width: float,
                 block_length: float,
                 block_com_relative_to_centroid: np.ndarray,
                 is_on_table: Callable[[np.ndarray], float],
                 start_pose: np.ndarray,
                 start_ori: float,
                 goal_pos: np.ndarray):

        self.block_width = block_width
        self.block_length = block_length
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

        self.fallen = False

    def apply_push_trajectory(self, pose_traj: List[np.ndarray], number_interp_points=1) -> bool:
        """
            traj (List[np.ndarray]): a timestepped sequence of object centroid _not_ COM, relative to the starting pose
                                    of the block. Pose is represnted by 3x3 SE(2) matrices.

                                    The internal state of the block will be updated to travel along this trajectory,
                                     relative to the present pose of the block.
                                     We assume a linear interpolation between provided trajectory points.

            number_interp_points (int): number of (linear) interpolation points to place in between timesteps
                                        to check for falling.

            Returns (bool): Whether or not the block had fallen over the course of the trajectory.
        """

        # if self.fallen:
        #     return True

        # first, compute what would be the trajectory of the block
        sim_pose_traj = np.array([self.pose @ _Tp for _Tp in pose_traj])

        self.rest_poses.append(sim_pose_traj[0])

        # add to full recorded trajectory and check for block falls
        for _idx, _Tp in enumerate(sim_pose_traj):
            self.full_traj.append(_Tp)

            if _idx == 0:
                _com_pos = (
                    _Tp @ np.array([self.block_com[0], self.block_com[1], 1.0]))[:2]
                if self.is_on_table_fn(_com_pos) < 0.0:
                    self.pose = _Tp
                    self.fallen = True
                    return True  # return True, since the block fell
            else:
                _Tpm1 = sim_pose_traj[_idx - 1]
                _pm1, _angtm1 = self.se2_to_xytheta(_Tpm1)

                _p, _ang = self.se2_to_xytheta(_Tp)

                # an easier way to interpolate the 'right way' around
                _rtm1_r = R.from_rotvec(np.array([[0.0, 0.0, _angtm1],
                                                 [0.0, 0.0, _ang]]))
                _rot_interpolator = Slerp([0, 1], _rtm1_r)

                _ps = np.array([
                    (_p - _pm1) * _t
                    for _t in np.linspace(0.0, 1.0, num=number_interp_points)
                ]) + _pm1

                _rs = _rot_interpolator(np.linspace(
                    0.0, 1.0, num=number_interp_points)).as_rotvec()[:, 2]

                for _ip, _ir in zip(_ps, _rs):
                    _T = self.xytheta_to_se2(_ip, _ir)
                    _com_pos = (
                        _T @ np.array([self.block_com[0], self.block_com[1], 1.0]))[:2]
                    if self.is_on_table_fn(_com_pos) < 0.0:
                        self.pose = _Tp
                        self.fallen = True
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
        self.fallen = False

    def visualize(self, show: bool = False, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, res=0.01):
        """
            show (bool): whether to use `maptolitlib.pyplot.show`. If set to `False` (default), then
                         save an instance of the figure in the current directory 
                         (named with current timestamp).
        """
        fig, ax = plt.subplots()

        # first, create a plot of the table
        x = np.arange(min_x, max_x + res, res)
        y = np.arange(min_y, max_y + res, res)
        X, Y = np.meshgrid(x, y)
        XY = np.stack((X, Y), axis=-1)

        # plot a filled contour that visualizes table
        Z_on_table = self.is_on_table_fn(XY)
        ax.contourf(X, Y, Z_on_table, [-100.0, 0.0, 100.0],
                    colors=['w', 'gray', 'gray'], zorder=-1)

        # plot the full trajectory taken by the block
        block_plots = []
        if len(self.full_traj) > 0:
            _full_traj = np.array(self.full_traj)

            # plot com
            coms = _full_traj @ np.array([self.block_com[0],
                                         self.block_com[1], 1.0])
            ax.scatter(coms[:, 0], coms[:, 1], color='black', s=8, zorder=1)

            # plot block

            for _Tp in _full_traj:
                _pos, _ang = self.se2_to_xytheta(_Tp)
                _ang = np.rad2deg(_ang)

                _patch = patches.Rectangle(_pos - np.array([self.block_length / 2, self.block_width / 2]),
                                           self.block_length,
                                           self.block_width,
                                           angle=_ang,
                                           rotation_point='center')
                block_plots.append(_patch)

            ax.add_collection(
                PatchCollection(block_plots, fc='none', ec='black', zorder=1)
            )

        # plot the resting positions of the block
        if len(self.rest_poses) > 0:
            _rest_poses = np.array(self.rest_poses)
            # print(_rest_poses)

            # plot com
            coms = _rest_poses @ np.array([self.block_com[0],
                                          self.block_com[1], 1.0])
            ax.scatter(coms[:, 0], coms[:, 1], color='red', s=5, zorder=1)

            for _Tp in _rest_poses:
                _pos, _ang = self.se2_to_xytheta(_Tp)
                _ang = np.rad2deg(_ang)

                _patch = patches.Rectangle(_pos - np.array([self.block_length / 2, self.block_width / 2]),
                                           self.block_length,
                                           self.block_width,
                                           angle=_ang,
                                           rotation_point='center')
                block_plots.append(_patch)

        n_rest = len(self.rest_poses)
        print("n_rest", n_rest)
        # print("block_plots length", len(block_plots))
        
        # Use a more distinct colormap with clear separation between colors
        cmap = plt.cm.get_cmap('viridis', n_rest)
        
        # Directly map each patch to a specific color index in the colormap
        # This ensures maximum color difference between consecutive patches
        for i, patch in enumerate(block_plots):
            # Calculate normalized color position to spread colors across the colormap
            color_pos = i / (n_rest - 1) if n_rest > 1 else 0.5
            color = cmap(color_pos)
            
            # Make edge color more visible
            patch.set_edgecolor(color)
            patch.set_linewidth(2.0)  # Thicker lines for better visibility
            ax.add_patch(patch)
        
        # Create a colormap for the colorbar with evenly distributed colors
        # Use a ListedColormap to ensure even spacing
        colors = [cmap(i/(n_rest-1)) for i in range(n_rest)] if n_rest > 1 else [cmap(0.5)]
        custom_cmap = plt.cm.colors.ListedColormap(colors)
        
        # Create a properly normalized colormap
        # Ensure we have at least 2 boundaries for BoundaryNorm
        if n_rest > 0:
            bounds = np.arange(0.5, n_rest+1.5)
        else:
            bounds = np.array([0, 1])  # Provide default bounds when n_rest is 0
            
        # Ensure n_colors is at least 1 for the norm
        n_colors = max(1, n_rest)
        norm = plt.cm.colors.BoundaryNorm(bounds, n_colors)
        
        # Create a scalar mappable with the custom colormap
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        
        # Add goal marker
        ax.scatter([self.goal_pos[0]], [self.goal_pos[1]], color='g', marker='*', s=100, zorder=2)

        # Create a colorbar with evenly distributed discrete steps
        cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(1, n_rest+1), orientation='vertical', label='Push Step Number')
        cbar.set_ticklabels(range(1, n_rest+1))
        ax.set_aspect('equal', adjustable='box')

        if show:
            plt.show()
            return

        uid = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        plt.savefig(f'table_{uid}.png')

    def se2_to_xytheta(self, _Tp):
        _pos = _Tp[:2, 2]
        _ang = np.arctan2(_Tp[1, 0], _Tp[0, 0])
        return _pos, _ang

    def xytheta_to_se2(self, p, r):
        T = np.eye(3)
        T[0:2, 2] = p
        T[:2, :2] = np.array([[np.cos(i * r), -np.sin(i * r)],
                              [np.sin(i * r),  np.cos(i * r)]])
        return T

# use SDF-like tricks to represent the table surface (faster than messing with explicit geom)
# handy cheatsheet: https://iquilezles.org/articles/distfunctions/
# NOTE: everything in cheatsheet above has been flipped by a _sign_


class BoxTable(SimulatedTable):
    def __init__(self,
                 block_width: float,
                 block_length: float,
                 block_com_relative_to_centroid,
                 goal_loc_x,
                 goal_loc_y,
                 table_length=5.0,
                 table_width=2.5
                 ):
        table_center = np.array([table_length / 2, table_width / 2])
        start_pos = np.array([0.2*table_length, 0.2*table_width])
        goal_pos = np.array([goal_loc_x, goal_loc_y])

        # Store table dimensions as class attributes
        self.table_length = table_length
        self.table_width = table_width

        def is_on_table(xy):
            return np.min(table_center - np.abs(table_center - xy), axis=-1)

        super().__init__(block_width,
                         block_length,
                         block_com_relative_to_centroid,
                         is_on_table,
                         start_pos,
                         0.0,
                         goal_pos)


class BeamTable(SimulatedTable):
    def __init__(self,
                 block_width: float,
                 block_length: float,
                 block_com_relative_to_centroid,
                 table_length=4.0,
                 narrow_width=0.5,
                 platform_rad=0.5):

        table_center = np.array(
            [table_length / 2 + platform_rad, platform_rad])
        start_pos = np.array([platform_rad, platform_rad])
        goal_pos = np.array([platform_rad + table_length, platform_rad])

        def is_on_table(xy):
            on_narrow = np.min(
                np.array([table_length / 2, narrow_width / 2]) - np.abs(table_center - xy), axis=-1)
            on_start_platform = platform_rad - \
                np.linalg.norm(start_pos - xy, ord=np.inf, axis=-1)
            on_goal_platform = platform_rad - \
                np.linalg.norm(goal_pos - xy, ord=np.inf, axis=-1)

            return np.maximum(on_narrow, np.maximum(on_start_platform, on_goal_platform))

        super().__init__(block_width, block_length, block_com_relative_to_centroid,
                         is_on_table, start_pos, 0.0, goal_pos)


class RingTable(SimulatedTable):
    def __init__(self,
                 block_width: float,
                 block_length: float,
                 block_com_relative_to_centroid,
                 ring_center=np.ones(2) * 2.5,
                 outer_rad=2.0,
                 inner_rad=1.5,
                 platform_rad=0.5,
                 ):

        track_middle_rad_disp = np.array([(outer_rad + inner_rad) / 2, 0.0])
        start_pos = ring_center - track_middle_rad_disp
        goal_pos = ring_center + track_middle_rad_disp

        def is_on_table(xy):
            # the ring-like tables
            dist_from_center = np.linalg.norm(ring_center - xy, axis=-1)
            outer_ring = outer_rad - dist_from_center
            inner_ring = inner_rad - dist_from_center

            # the squares placed at the beginning/end
            dist_from_start = np.linalg.norm(
                start_pos - xy, ord=np.inf, axis=-1)
            dist_from_goal = np.linalg.norm(
                goal_pos - xy, ord=np.inf, axis=-1)

            start_platform = platform_rad - dist_from_start
            goal_platform = platform_rad - dist_from_goal

            return np.maximum(np.maximum(
                np.minimum(outer_ring, -inner_ring),
                start_platform),
                goal_platform
            )

        super().__init__(block_width,
                         block_length,
                         block_com_relative_to_centroid,
                         is_on_table,
                         start_pos,
                         start_ori=0.0,
                         goal_pos=goal_pos)


if __name__ == '__main__':

    traj1 = [np.eye(3) for _ in range(0, 10)]
    turn_rate = np.pi / 3
    for i in range(len(traj1)):
        traj1[i][0, 2] += 0.3 * i
        traj1[i][:2, :2] = np.array([[np.cos(i * turn_rate), -np.sin(i * turn_rate)],
                                     [np.sin(i * turn_rate),  np.cos(i * turn_rate)]])

    traj2 = [np.eye(3) for _ in range(0, 10)]
    for i in range(len(traj1)):
        traj2[i][1, 2] += 0.3 * i
        # traj2[i][:2, :2] = np.array([[np.cos(i * turn_rate), -np.sin(i * turn_rate)],
        #                          [np.sin(i * turn_rate),  np.cos(i * turn_rate)]])

    # there are optional keywords that specify geometric params of each of the tables
    # (see classes above)
    bxt = BoxTable(0.3, 0.5, np.ones(2) * 0.1)
    print(f'off table after push1: {bxt.apply_push_trajectory(traj1)}')
    print(f'off table after push2: {bxt.apply_push_trajectory(traj2)}')
    bxt.visualize(show=True)

    bmt = BeamTable(0.3, 0.5, np.ones(2) * 0.1)
    print(f'off table after push1: {bmt.apply_push_trajectory(traj1)}')
    print(f'off table after push2: {bmt.apply_push_trajectory(traj2)}')
    bmt.visualize(show=True)

    rt = RingTable(0.3, 0.5, np.ones(2) * 0.1)
    print(f'off table after push1: {rt.apply_push_trajectory(traj1)}')
    print(f'off table after push2: {rt.apply_push_trajectory(traj2)}')
    rt.visualize(show=True)

    # can reset for a new run too:
    rt.reset()
    print(f'off table after push2: {rt.apply_push_trajectory(traj2)}')
    print(f'off table after push1: {rt.apply_push_trajectory(traj1)}')
    rt.visualize(show=True)

    # can query distance to goal:
    print(f'distance to goal on ringtable: {rt.get_distance_to_goal()}m')
