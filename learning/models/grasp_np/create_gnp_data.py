import argparse
from email.policy import default
from re import T
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import shutil

from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from learning.domains.grasping.grasp_data import GraspParallelDataLoader
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody
from types import SimpleNamespace


def transform_points(local_points, all_points, finger1, finger2, ee, viz_data=False):
    midpoint = (finger1 + finger2)/2
    new_x = (finger1 - finger2)/np.linalg.norm(finger1 - finger2)
    new_y = (midpoint - ee)/np.linalg.norm(midpoint - ee)
    new_z = np.cross(new_x, new_y)

    # Build transfrom from world frame to grasp frame.
    R, t = np.hstack([new_x[:, None], new_y[:, None], new_z[:, None]]), midpoint
    tform = np.eye(4)
    tform[0:3, 0:3] = R
    tform[0:3, 3] = t
    inv_tform = np.linalg.inv(tform)

    local_xyzs = local_points[:, 0:3]
    new_xyzs = np.hstack([local_xyzs, np.ones((local_xyzs.shape[0], 1))])
    new_xyzs = (inv_tform@new_xyzs.T).T[:, 0:3]
    local_normals = local_points[:, 3:6]
    new_normals = np.hstack([local_normals, np.ones((local_normals.shape[0], 1))])
    new_normals = (inv_tform@new_normals.T).T[:, 0:3]

    new_points = np.concatenate(
        [new_xyzs, new_normals, local_points[:, 6:]], axis=1
    )
    dist_to_finger = np.linalg.norm(finger1-midpoint)
    new_points[:, 0] /= dist_to_finger
    new_points[:, 1:3] /= 0.03

    if viz_data:
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')

        all_xyzs = all_points[:2048, 0:3]
        all_normals = all_points[:2048, 3:6]
        normal_lines = np.zeros((all_normals.shape[0], 3, 2), dtype='float32')
        normal_lines[:, :, 0] = all_xyzs
        normal_lines[:, :, 1] = all_xyzs + 0.01*all_normals

        ax0.scatter(all_xyzs[:, 0], all_xyzs[:, 1], all_xyzs[:, 2], color='k', alpha=0.2, s=5)
        ax0.scatter(*finger1, color='r', s=50)
        ax0.scatter(*finger2, color='g', s=50)
        for lx in range(0, normal_lines.shape[0]):
            ax0.plot(normal_lines[lx, 0, :], normal_lines[lx ,1, :], normal_lines[lx, 2, :], color='y', linewidth=1, alpha=0.2)

        normal_lines = np.zeros((local_normals.shape[0], 3, 2), dtype='float32')
        normal_lines[:, :, 0] = local_xyzs
        normal_lines[:, :, 1] = local_xyzs + 0.01*local_normals
        ax1.scatter(local_xyzs[:, 0], local_xyzs[:, 1], local_xyzs[:, 2], color='k', alpha=0.2, s=5)
        for lx in range(0, normal_lines.shape[0]):
            ax1.plot(normal_lines[lx, 0, :], normal_lines[lx ,1, :], normal_lines[lx, 2, :], color='y', linewidth=1, alpha=0.2)
        ax1.scatter(*finger1, color='r', s=50)
        ax1.scatter(*finger2, color='g', s=50)
        ax1.scatter(*ee, color='b')
        finger1_line = np.vstack([finger1, ee])
        finger2_line = np.vstack([finger2, ee])
        ax1.plot(finger1_line[:, 0], finger1_line[:, 1], finger1_line[:, 2], color='r')
        ax1.plot(finger2_line[:, 0], finger2_line[:, 1], finger2_line[:, 2], color='r')

        normal_lines = np.zeros((new_normals.shape[0], 3, 2), dtype='float32')
        normal_lines[:, :, 0] = new_points[:, 0:3]
        normal_lines[:, :, 1] = new_points[:, 0:3] + 0.1*new_normals
        ax2.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], color='k', alpha=0.2)
        ax2.scatter(0, 0, 0, color='r')
        for lx in range(0, normal_lines.shape[0]):
            ax2.plot(normal_lines[lx, 0, :], normal_lines[lx ,1, :], normal_lines[lx, 2, :], color='y', linewidth=2, alpha=0.2)
        

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        plt.show()

    return new_points.astype('float32'), tform


def process_geometry(train_dataset, radius=0.02, skip=1, verbose=True):
    object_names = train_dataset['object_data']['object_names']
    object_properties = train_dataset['object_data']['object_properties']

    all_grasps = train_dataset['grasp_data']['grasps'][::skip]
    all_ids = train_dataset['grasp_data']['object_ids'][::skip]
    all_labels = train_dataset['grasp_data']['labels'][::skip]
    all_forces = train_dataset['grasp_data']['forces'][::skip]
    if 'raw_grasps' in train_dataset['grasp_data']:
        all_raw_grasps = train_dataset['grasp_data']['raw_grasps'][::skip]
    else:
        all_raw_grasps = [None]*len(all_grasps)
    gx = 0

    # Collect all mesh points for each object.
    all_points_per_objects = {}
    for grasp_vector, object_id in zip(all_grasps, all_ids):
        mesh_points = grasp_vector[3:, :]
        if object_id not in all_points_per_objects:
            all_points_per_objects[object_id] = mesh_points
        else:
            all_points_per_objects[object_id] = np.concatenate([all_points_per_objects[object_id], mesh_points], axis=0)

    # For each grasp, find close points and convert them to the local grasp frame.
    new_geometries_dict = defaultdict(list)  # obj_id -> [grasp__points]
    new_midpoints_dict = defaultdict(list)  # obj_id -> [grasp_midpoint]
    new_labels_dict = defaultdict(list)  # obj_id -> [grasp_label]
    new_forces_dict = defaultdict(list)  # obj_id -> [forces]
    new_grasp_points_dict = defaultdict(list)  # obj_id -> [finger_pt_1, finger_pt_2]
    new_curvatures_dict = defaultdict(list)  # obj_id -> [finger1_gauss concat finger1_mean, finger2_gauss concat finger2_mean]
    new_normals_dict = defaultdict(list)  # obj_id -> [finger1_normal, finger2_normal]
    new_grasp_tforms_dict = defaultdict(list) # obj_id -> [tforms {4x4 homogeneous matrix}]

    new_raw_grasps_dict = defaultdict(list)
    new_meshes_dict = defaultdict(list)

    for grasp_vector, object_id, force, label, raw_grasp in zip(all_grasps, all_ids, all_forces, all_labels, all_raw_grasps):
        if verbose:
            print(f'Converting grasp {gx}/{len(all_ids)}...')
        gx += 1

        object_name = object_names[object_id]
        object_property = object_properties[object_id]

        graspable_body = graspablebody_from_vector(object_name, object_property)

        finger1 = grasp_vector[0, 0:3]
        finger2 = grasp_vector[1, 0:3]
        ee = grasp_vector[2, 0:3]
        normal1 = grasp_vector[0, 3:6]
        normal2 = grasp_vector[1, 3:6]
        curvatures_finger1 = grasp_vector[0, 6:]
        curvatures_finger2 = grasp_vector[1, 6:]

        midpoint = (finger1 + finger2)/2.0

        # Only sample points that are within 2cm of a grasp point.
        candidate_points = all_points_per_objects[object_id]
        d1 = np.linalg.norm(candidate_points[:, :3]-finger1, axis=1)
        d2 = np.linalg.norm(candidate_points[:, :3]-finger2, axis=1)
        to_keep = np.logical_or(d1 < radius, d2 < radius)

        points = all_points_per_objects[object_id][to_keep][:512, :]  # Keep 512 points closest to fingers.
        n_found = points.shape[0]
        if verbose:
            print(f'{n_found}/{to_keep.sum()} points found for grasp {gx}, object {object_id}')

        # Put everything in grasp point ref frame for better generalization. (Including grasp points)
        if gx % 200 == 0:
            viz_data = False
        else:
            viz_data = False
        points, world_to_grasp_tform = transform_points(
            local_points=points,
            finger1=finger1,
            finger2=finger2,
            ee=ee,
            viz_data=viz_data,
            all_points=candidate_points
        )

        # Assemble dataset.
        new_grasp_points_dict[object_id].append(np.array([finger1, finger2], dtype='float32'))
        new_curvatures_dict[object_id].append(np.array([curvatures_finger1, curvatures_finger2], dtype='float32'))
        new_grasp_tforms_dict[object_id].append(world_to_grasp_tform)
        new_geometries_dict[object_id].append(points)
        new_normals_dict[object_id].append(np.array([normal1, normal2], dtype='float32'))
        new_midpoints_dict[object_id].append(midpoint)
        new_forces_dict[object_id].append(force)
        new_labels_dict[object_id].append(label)
        new_meshes_dict[object_id] = all_points_per_objects[object_id][:512, :]
        new_raw_grasps_dict[object_id].append(raw_grasp)

    dataset = {
        'grasp_data': {
            'object_meshes': new_meshes_dict,
            'grasp_points': new_grasp_points_dict,
            'grasp_tforms': new_grasp_tforms_dict,
            'grasp_curvatures': new_curvatures_dict,
            'grasp_geometries': new_geometries_dict,
            'grasp_normal': new_normals_dict,
            'grasp_midpoints': new_midpoints_dict,
            'grasp_forces': new_forces_dict,
            'labels': new_labels_dict
        },
        'object_data': train_dataset['object_data'],
        'metadata': train_dataset['metadata']
    }
    if 'raw_grasps' in train_dataset['grasp_data']:
        dataset['grasp_data']['raw_grasps'] = new_raw_grasps_dict
    return dataset

def process_and_save(func_args):
    with open(func_args.dataset_path, 'rb') as handle:
        dataset = pickle.load(handle)

    new_dataset = process_geometry(
        dataset,
        radius=func_args.radius,
        skip=1,
        verbose=True
    )

    with open(func_args.out_path, 'wb') as handle:
        pickle.dump(new_dataset, handle)


DATA_ROOT = 'learning/data/grasping'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dataset-fname', type=str, required=True)
    parser.add_argument('--out-dataset-fname', type=str, required=True)
    parser.add_argument('--radius', type=float, default=0.03)
    parser.add_argument('--n-processes', type=int, default=1)
    args = parser.parse_args()
    print(args)

    # ----- Top-level dataset files -----
    in_data_root_path = os.path.join(DATA_ROOT, args.in_dataset_fname)
    out_data_root_path = os.path.join(DATA_ROOT, args.out_dataset_fname)
    if not os.path.exists(out_data_root_path):
        os.mkdir(out_data_root_path)

    in_args_path = os.path.join(in_data_root_path, 'args.pkl')
    out_args_path = os.path.join(out_data_root_path, 'args.pkl')
    if not os.path.exists(out_args_path):
        shutil.copy(in_args_path, out_args_path)

    # ----- Object files -----
    in_objects_path = os.path.join(in_data_root_path, 'objects')
    out_objects_path = os.path.join(out_data_root_path, 'objects')
    if not os.path.exists(out_objects_path):
        os.mkdir(out_objects_path)

    in_train_geo_train_props = os.path.join(in_objects_path, 'train_geo_train_props.pkl')
    in_train_geo_test_props = os.path.join(in_objects_path, 'train_geo_test_props.pkl')
    in_test_geo_test_props = os.path.join(in_objects_path, 'test_geo_test_props.pkl')
    out_train_geo_train_props = os.path.join(out_objects_path, 'train_geo_train_props.pkl')
    out_train_geo_test_props = os.path.join(out_objects_path, 'train_geo_test_props.pkl')
    out_test_geo_test_props = os.path.join(out_objects_path, 'test_geo_test_props.pkl')
    if not os.path.exists(out_train_geo_train_props):
        shutil.copy(in_train_geo_train_props, out_train_geo_train_props)
    if not os.path.exists(out_train_geo_test_props):
        shutil.copy(in_train_geo_test_props, out_train_geo_test_props)
    if not os.path.exists(out_test_geo_test_props):
        shutil.copy(in_test_geo_test_props, out_test_geo_test_props)

    # ----- Grasp paths -----
    out_grasps_path = os.path.join(out_data_root_path, 'grasps')
    if not os.path.exists(out_grasps_path):
        os.mkdir(out_grasps_path)

    out_training_phase_path = os.path.join(out_grasps_path, 'training_phase')
    if not os.path.exists(out_training_phase_path):
        os.mkdir(out_training_phase_path)

    in_fitting_phase_path = os.path.join(in_data_root_path, 'grasps', 'fitting_phase')
    out_fitting_phase_path = os.path.join(out_grasps_path, 'fitting_phase')
    if not os.path.exists(out_fitting_phase_path):
        os.mkdir(out_fitting_phase_path)

    # ----- Training phase grasps -----
    train_grasps_path = os.path.join(out_training_phase_path, 'train_grasps.pkl')
    if True: # not os.path.exists(train_grasps_path):
        train_data_path = os.path.join(
            in_data_root_path,
            'grasps',
            'training_phase',
            'train_grasps.pkl'
        )
        with open(train_data_path, 'rb') as handle:
            train_dataset = pickle.load(handle)

        new_train_dataset = process_geometry(
            train_dataset,
            radius=args.radius,
            skip=1,
        )

        with open(train_grasps_path, 'wb') as handle:
            pickle.dump(new_train_dataset, handle)

    val_grasps_path = os.path.join(out_training_phase_path, 'val_grasps.pkl')
    if not os.path.exists(val_grasps_path):
        val_data_path = os.path.join(
            in_data_root_path,
            'grasps',
            'training_phase',
            'val_grasps.pkl'
        )
        with open(val_data_path, 'rb') as handle:
            val_dataset = pickle.load(handle)

        new_val_dataset = process_geometry(
            val_dataset,
            radius=args.radius,
            skip=1
        )

        with open(val_grasps_path, 'wb') as handle:
            pickle.dump(new_val_dataset, handle)

    # ----- Fitting phase grasps -----
    fitting_tasks = []
    for fitting_fname in os.listdir(in_fitting_phase_path):
        out_fitting_data_path = os.path.join(out_fitting_phase_path, fitting_fname)
        if not os.path.exists(out_fitting_data_path):
            fitting_data_path = os.path.join(in_fitting_phase_path, fitting_fname)
            fitting_task = SimpleNamespace(
                dataset_path=fitting_data_path,
                radius=args.radius,
                out_path=out_fitting_data_path
            )
            fitting_tasks.append(fitting_task)

    worker_pool = multiprocessing.Pool(
        processes=args.n_processes,
        maxtasksperchild=1
    )
    worker_pool.map(process_and_save, fitting_tasks)
    worker_pool.close()
