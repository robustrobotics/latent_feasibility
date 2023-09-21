import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import os
import open3d as o3d
import sys
sys.path.extend(["pybullet_planning"])

# from pybullet_tools.utils import multiply
# from robots.panda.exp_manual import HAND_TO_DEPTH

data_file = "panda_dataset/09-13-2023-15:00:28.pkl"

def to_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(list(pose[1])).as_matrix()
    matrix[:3, 3] = np.array(pose[0])
    return matrix

def generate_pointcloud(data, ee_pose, depth_scale=1):

    intrinsics = data[2][0] # data["camera_image"]["camera_matrix"][0]
    depth_data = o3d.geometry.Image((data[1]).astype(np.float32))
    rgb_data = o3d.geometry.Image(data[0].astype(np.uint8))
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(np.array(rgb_data).shape[0], np.array(rgb_data).shape[1], intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_data, depth_data, convert_rgb_to_intensity=False, depth_trunc=1.0)

    # panda_hand_T_depth = data["hand_T_depth"]
    # world_T_panda_hand = data["world_T_hand"]
    world_T_panda_hand = np.eye(4)
    world_T_panda_hand[0:3, 3] = ee_pose[0]
    world_T_panda_hand[0:3, 0:3] = Rotation.from_quat(ee_pose[1]).as_matrix()
    # print(ee_pose[0])
    panda_hand_T_depth = np.eye(4)
    panda_hand_T_depth[0:3, 3] = np.array([0.036499, -0.034889, 0.0594])
    panda_hand_T_depth[0:3, 0:3] = Rotation.from_quat(np.array([0.00252743, 0.0065769, 0.70345566, 0.71070423])).as_matrix()
    # import ipdb; ipdb.set_trace()
    # extrinsic = to_matrix(multiply(world_T_panda_hand, panda_hand_T_depth))
    extrinsic = world_T_panda_hand@panda_hand_T_depth
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(intrinsic),
        extrinsic=np.linalg.inv(extrinsic))
    
    # Filter out plane.
    # import ipdb; ipdb.set_trace()
    plane, inliers = pcd.segment_plane(
        distance_threshold=0.04,
        ransac_n=5,
        num_iterations=500
    )
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # TODO: Remove blue points.
    colors = np.asarray(outlier_cloud.colors)
    blue = colors[:, 2] > 0.4
    blue_idx = np.arange(blue.shape[0])[blue]
    outlier_cloud = outlier_cloud.select_by_index(blue_idx, invert=True)

    _, ind = outlier_cloud.remove_radius_outlier(nb_points=128, radius=0.02)
    outlier_cloud = outlier_cloud.select_by_index(ind)

    # TODO: Remove points far from look pose.
    points = np.asarray(outlier_cloud.points)
    look_point = np.array([0.4, 0.0, 0.0])
    dists = np.linalg.norm(points - look_point, axis=1)
    ind = np.arange(points.shape[0])[dists < 0.35]
    outlier_cloud = outlier_cloud.select_by_index(ind)

    return outlier_cloud

if __name__ == "__main__":
    datas, pcds, ee_poses = [], [], []
    object_name = "hammer"
    for qx in range(0, 5):
        with open(f'tamp/depth_imgs/{object_name}_{qx}.pkl', 'rb') as handle:
            datas.append(pickle.load(handle))
        with open(f'tamp/depth_imgs/ee_pose_{qx}.pkl', 'rb') as handle:
            ee_poses.append(pickle.load(handle))

    for data, ee_pose in zip(datas, ee_poses):
        pcd = generate_pointcloud(data, ee_pose)
        pcds.append(pcd)

    print(pcds)
    o3d.visualization.draw_geometries(pcds)
    aligned_pcds = [pcds[0]]
    for pcd in pcds[1:]:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd,
            pcds[0],
            0.02,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        aligned_pcds.append(pcd.transform(reg_p2p.transformation))

    o3d.visualization.draw_geometries(pcds)

    final_pcd = pcds[0]
    for pcd in pcds[1:]:
        final_pcd += pcd
    
    o3d.visualization.draw_geometries([final_pcd])

    hull, _ = final_pcd.compute_convex_hull()
    import ipdb; ipdb.set_trace()

    # def plot_registered():
    #     current_transformation = np.identity(4)

    #     pcds[0].estimate_normals()
    #     pcds[1].estimate_normals()

    #     # point cloud registration
    #     result_icp = o3d.pipelines.registration.registration_icp(
    #         pcds[0], pcds[1], 20, current_transformation,
    #         o3d.pipelines.registration.TransformationEstimationPointToPlane())

    #     newpc0 = copy.deepcopy(pcds[0])
    #     newpc0.transform(result_icp.transformation)

    #     o3d.visualization.draw_geometries([newpc0, pcds[1]])