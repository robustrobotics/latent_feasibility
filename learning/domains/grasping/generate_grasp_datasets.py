import argparse
import random
import os
import pickle
import sys
import time

import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure
from pb_robot.planners.antipodalGraspPlanner import (
    GraspableBodySampler,
    GraspSampler,
    GraspSimulationClient,
    GraspableBody,
    GraspStabilityChecker
)


def vector_from_graspablebody(graspable_body):
    """ Represent the intrinsic parameters of a body as a vector. """
    vector = np.array(graspable_body.com + (graspable_body.mass, graspable_body.friction))
    return vector


def graspablebody_from_vector(object_name, vector):
    """ Instatiate a GraspableBody given a vector of parameters. """
    graspable_body = GraspableBody(object_name=object_name,
                                   com=tuple(vector[0:3]),
                                   mass=vector[3],
                                   friction=vector[4])
    return graspable_body

def sample_grasps_and_Xs(graspable_body, n_grasps, n_points_per_object, curvature_rads, grasps=None):
    # Sample new point cloud for object.
    start_mesh = time.time()

    mesh_name = os.path.join(os.environ['SHAPENET_ROOT'], 'mesh_samples', graspable_body.object_name)
    sim_client = GraspSimulationClient(graspable_body, False)
    sim_client.disconnect()
    if not os.path.exists(mesh_name):
        print('Computing mesh features...')
        points, indices = sim_client.mesh.sample(n_points_per_object*n_grasps, return_index=True)
        mesh_points_all = np.array(points, dtype='float32')
        mesh_points = np.hstack([mesh_points_all,
                                np.ones((n_points_per_object*n_grasps, 1), dtype='float32')])
        mesh_points = (sim_client.mesh_tform @ (mesh_points.T)).T[:, 0:3].reshape((n_grasps, n_points_per_object, 3))
        # TODO: Sample normals and curvatures.
        mesh_normals = np.array(sim_client.mesh.face_normals[indices, :], dtype='float32').reshape((n_grasps, n_points_per_object, 3))


        if len(curvature_rads) > 0:
            mesh_gaussian_curvs = np.concatenate(tuple(map(
                lambda rad: discrete_gaussian_curvature_measure(sim_client.mesh, mesh_points_all, rad)[:, None],
                curvature_rads
            )), axis=-1).reshape((n_grasps, n_points_per_object, 3))
            mesh_mean_curvs = np.concatenate(tuple(map(
                lambda rad: discrete_mean_curvature_measure(sim_client.mesh, mesh_points_all, rad)[:, None],
                curvature_rads
            )), axis=-1).reshape((n_grasps, n_points_per_object, 3))

        else:
            mesh_gaussian_curvs = np.zeros((n_grasps, n_points_per_object, 3)) 
            mesh_mean_curvs = np.zeros((n_grasps, n_points_per_object, 3))

        mesh_curvatures = np.concatenate(
            [mesh_gaussian_curvs, mesh_mean_curvs],
            axis=-1
            )
        mesh_points_with_local_features = np.concatenate(
                [mesh_points, mesh_normals, mesh_curvatures],
                axis=-1
            )  # (n_grasps x n_points_per_object x 12)
        print('Saving mesh features...')
        with open(mesh_name, 'wb') as handle:
            pickle.dump(mesh_points_with_local_features, handle)
    else:
        # print('Loading mesh features...')
        with open(mesh_name, 'rb') as handle:
            mesh_points_with_local_features = pickle.load(handle)
        # Shuffle mesh points.
        all_mesh_points = mesh_points_with_local_features.reshape((-1, 12))
        point_indices = np.random.choice(
            np.arange(0, all_mesh_points.shape[0]),
            size=n_points_per_object*n_grasps,
            replace=True
        )
        all_mesh_points = all_mesh_points[point_indices]
        mesh_points_with_local_features = all_mesh_points.reshape((n_grasps, n_points_per_object, 12))

    #print('Mesh sampling done...')

    if grasps is None:
        # Sample grasp.
        input("SHOULD NOT BE HERE!")
        try:
            random.seed()
            np.random.seed()
            grasp_sampler = GraspSampler(   
                graspable_body=graspable_body,
                antipodal_tolerance=30,
                show_pybullet=False
            )
            grasps = grasp_sampler.sample_grasps(
                num_grasps=n_grasps,
                force_range=(5, 20),
                show_trimesh=False
            )
            grasp_sampler.disconnect()
        except Exception as e:
            grasp_sampler.disconnect()
            raise e

    # Encode grasp as points.
    grasps_vectors = np.array(
        [(g.pb_point1, g.pb_point2, g.ee_relpose[0]) for g in grasps],
        dtype=np.float32
    )

    grasp_normals = np.array(
        [(g.normal1, g.normal2, [0., 0., 0.]) for g in grasps],
        dtype=np.float32
    )

    start_curvs = time.time()
    if len(curvature_rads) > 0:
        # compute Gaussian and mean curvatures at multiple resolutions
        gaussian_curvatures_grasp_point1 = np.concatenate(tuple(map(
            lambda rad: discrete_gaussian_curvature_measure(sim_client.mesh, grasps_vectors[:, 0, :], rad)[:, None],
            curvature_rads
        )), axis=-1)

        mean_curvatures_grasp_point1 = np.concatenate(tuple(map(
            lambda rad: discrete_mean_curvature_measure(sim_client.mesh, grasps_vectors[:, 0, :], rad)[:, None],
            curvature_rads
        )), axis=-1)

        gaussian_curvatures_grasp_point2 = np.concatenate(tuple(map(
            lambda rad: discrete_gaussian_curvature_measure(sim_client.mesh, grasps_vectors[:, 1, :], rad)[:, None],
            curvature_rads
        )), axis=-1)

        mean_curvatures_grasp_point2 = np.concatenate(tuple(map(
            lambda rad: discrete_mean_curvature_measure(sim_client.mesh, grasps_vectors[:, 1, :], rad)[:, None],
            curvature_rads
        )), axis=-1)
    else:
        gaussian_curvatures_grasp_point1 =  np.zeros((n_grasps, 3), dtype=np.float32)
        gaussian_curvatures_grasp_point2 =  np.zeros((n_grasps, 3), dtype=np.float32)
        mean_curvatures_grasp_point1 =  np.zeros((n_grasps, 3), dtype=np.float32)
        mean_curvatures_grasp_point2 =  np.zeros((n_grasps, 3), dtype=np.float32)
    # print('Curv time:', time.time() - start_curvs)

    # Add grasp/object indicator features and curvature information
    curvatures_pts1 = np.concatenate(
        [gaussian_curvatures_grasp_point1, mean_curvatures_grasp_point1],  # Both (N x 3)
        axis=-1
    )
    curvatures_pts2 = np.concatenate(
        [gaussian_curvatures_grasp_point2, mean_curvatures_grasp_point2],  # Both (N x 3)
        axis=-1
    )
    curvatures_pts3 = np.zeros((n_grasps, 1, 6), dtype=np.float32)  # Filler for ee point not on object.
    grasp_curvatures = np.concatenate(
        [curvatures_pts1[:, None, :], curvatures_pts2[:, None, :], curvatures_pts3],
        axis=1
    )  # (N x 3 x 6)

    grasps_with_local_features = np.concatenate(
        [grasps_vectors,  grasp_normals, grasp_curvatures],
        axis=-1
    )

    # Concatenate all relevant vectors (points, indicators, properties).
    Xs = np.concatenate(
        [grasps_with_local_features, mesh_points_with_local_features],
        axis=1
    )
    # print('Done generating')
    del sim_client
    return grasps, Xs

def sample_grasp_X(graspable_body, property_vector, n_points_per_object, curvature_rads, grasp=None):
    # Sample new point cloud for object.
    sim_client = GraspSimulationClient(graspable_body, False)

    # TODO: keeping local mesh points in dataset, but will remove as a nn input in case
    # curvature does not help
    mesh_points = np.array(sim_client.mesh.sample(n_points_per_object, return_index=False),
                           dtype='float32')
    mesh_points = np.hstack([mesh_points,
                             np.ones((n_points_per_object, 1), dtype='float32')])
    mesh_points = (sim_client.mesh_tform @ (mesh_points.T)).T[:, 0:3]
    sim_client.disconnect()

    # Sample grasp.
    if grasp is None:
        try:
            random.seed()
            np.random.seed()
            grasp_sampler = GraspSampler(graspable_body=graspable_body,
                                         antipodal_tolerance=30,
                                         show_pybullet=False)
            force = np.random.uniform(5, 20)   # TODO: magic force numbers alert!
            grasp = grasp_sampler.sample_grasp(force=force, show_trimesh=False)
        except Exception as e:
            grasp_sampler.disconnect()
            raise e
        grasp_sampler.disconnect()

    # Encode grasp as points.
    grasp_points = (grasp.pb_point1, grasp.pb_point2, grasp.ee_relpose[0])

    # compute Gaussian and mean curvatures at multiple resolutions
    gaussian_curvatures_grasp_point1 = np.concatenate(
        tuple(
            map(
                lambda rad: discrete_gaussian_curvature_measure(sim_client.mesh, grasp.pb_point1.reshape(1, 3), rad),
                curvature_rads
            )
        )
    )

    mean_curvatures_grasp_point1 = np.concatenate(
        tuple(
            map(
                lambda rad: discrete_mean_curvature_measure(sim_client.mesh, grasp.pb_point1.reshape(1, 3), rad),
                curvature_rads
            )
        )
    )

    gaussian_curvatures_grasp_point2 = np.concatenate(
        tuple(
            map(
                lambda rad: discrete_gaussian_curvature_measure(sim_client.mesh, grasp.pb_point2.reshape(1, 3), rad),
                curvature_rads
            )
        )
    )

    mean_curvatures_grasp_point2 = np.concatenate(
        tuple(
            map(
                lambda rad: discrete_mean_curvature_measure(sim_client.mesh, grasp.pb_point2.reshape(1, 3), rad),
                curvature_rads
            )
        )
    )

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], color='b')
    # grasp_xs = [p[0] for p in grasp_points]
    # grasp_ys = [p[1] for p in grasp_points]
    # grasp_zs = [p[2] for p in grasp_points]
    # ax.scatter(grasp_xs, grasp_ys, grasp_zs, color='r')
    # bound = 0.1
    # ax.set_xlim(-bound, bound)
    # ax.set_ylim(-bound, bound)
    # ax.set_zlim(-bound, bound)
    # plt.savefig('test.png')

    # Add grasp/object indicator features and curvature information
    grasp_vectors = np.array(grasp_points, dtype='float32')
    grasp_vectors = np.hstack([grasp_vectors, np.eye(3, dtype='float32')])

    # order: [[pt1 gauss curvature x 3, pt1 mean curvature x 3],
    # order:  [pt2 gauss curvature x 3, pt2 gauss curvature x 3]]
    curvatures = np.block([
        [gaussian_curvatures_grasp_point1, mean_curvatures_grasp_point1],
        [gaussian_curvatures_grasp_point2, mean_curvatures_grasp_point2]
    ])
    grasp_vectors = np.vstack([grasp_vectors, curvatures])

    mesh_vectors = np.hstack([mesh_points, np.zeros((n_points_per_object, 3), dtype='float32')])

    # Concatenate all relevant vectors (points, indicators, properties).
    X = np.vstack([grasp_vectors, mesh_vectors])
    props = np.broadcast_to(property_vector, (X.shape[0], len(property_vector)))
    X = np.hstack([X, props])

    return grasp, X


def generate_datasets(dataset_args):
    """ Generate a dataset of grasps and labels for a given object file. """

    assert 'learning/data/grasping' in dataset_args.fname
    with open(dataset_args.objects_fname, 'rb') as handle:
        object_data = pickle.load(handle)['object_data']

    object_grasp_data, object_grasp_forces, object_grasp_ids, object_grasp_labels = [], [], [], []

    for prop_ix in range(len(object_data['object_names'])):
        if dataset_args.object_ix > -1 and dataset_args.object_ix != prop_ix:
            continue
        print('Property sample %d/%d...' % (prop_ix, len(object_data['object_names'])))

        object_id = prop_ix
        object_name = object_data['object_names'][prop_ix]
        property_vector = object_data['object_properties'][prop_ix]

        graspable_body = graspablebody_from_vector(object_name, property_vector)
        print(f'Object name: {object_name}\tID: {dataset_args.object_ix}', property_vector)
        # Sample random grasps with labels.

        grasps, Xs = sample_grasps_and_Xs(
            graspable_body,
            dataset_args.n_grasps_per_object,
            dataset_args.n_points_per_object,
            dataset_args.curvature_radii
        )

        labeler = GraspStabilityChecker(graspable_body,
                                        stability_direction='all',
                                        label_type='relpose',
                                        grasp_noise=dataset_args.grasp_noise,
                                        show_pybullet=False,
                                        recompute_inertia=True)
        print('PBID:', labeler.sim_client.pb_client_id)
        print('Done:', len(grasps), Xs.shape)
        for grasp_ix, (grasp, X) in enumerate(zip(grasps, Xs)):
            print('Grasp %d/%d...' % (grasp_ix, dataset_args.n_grasps_per_object))

            # Get label.
            label = labeler.get_label(grasp)
            print('Label:', label)

            object_grasp_forces.append(grasp.force)
            object_grasp_data.append(X) # grasp points and curvature points are in X
            object_grasp_ids.append(object_id)
            object_grasp_labels.append(int(label))

        labeler.disconnect()

        dataset = {
            'grasp_data': {
                'grasps': object_grasp_data,
                'forces': object_grasp_forces,
                'object_ids': object_grasp_ids,
                'labels': object_grasp_labels
            },
            'object_data': object_data,
            'metadata': dataset_args
        }

    with open('%s' % dataset_args.fname, 'wb') as handle:
        pickle.dump(dataset, handle)

def get_property_vector(name):
    # Sample a new object.
    random.seed()
    np.random.seed()
    graspable_body = GraspableBodySampler.sample_random_object_properties(name)
    # Get property label vector.
    property_vector = vector_from_graspablebody(graspable_body)

    return name, property_vector

def generate_objects(obj_args):
    """ Sample random objects and intrinsic properties for those objects. """
    assert 'learning/data/grasping' in obj_args.fname
    tasks = []
    for obj_ix, name in enumerate(obj_args.object_names):
        # print('Object %d/%d...' % (obj_ix, len(obj_args.object_names)))
        for prop_ix in range(0, obj_args.n_property_samples):
            # print('Property sample %d/%d...' % (prop_ix, obj_args.n_property_samples))
            tasks.append(name)

    pool = mp.Pool(obj_args.n_processes)
    results = pool.map(get_property_vector, tasks)
    pool.close()
    
    object_instance_names = [res[0] for res in results]
    object_instance_properties = [res[1] for res in results]

    dataset = {
        'object_data': {
            'object_names': object_instance_names,
            'object_properties': object_instance_properties,
            'property_names': ['com_x', 'com_y', 'com_z', 'mass', 'friction']
        },
        'metadata': obj_args
    }
    with open('%s' % obj_args.fname, 'wb') as handle:
        pickle.dump(dataset, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        required=True,
                        choices=['objects', 'grasps'])
    parser.add_argument('--fname',
                        type=str,
                        required=True,
                        help='Base name used for saving all dataset files.')
    # args.mode == 'objects' parameters
    parser.add_argument('--object-names',
                        default=[],
                        nargs='+',
                        help='List of all SN or YCB objects to include.')
    parser.add_argument('--n-property-samples',
                        type=int,
                        default=-1,
                        help='Number of object instances per each YcbObject geometry type.')
    # args.mode == 'grasps' parameters
    parser.add_argument('--objects-fname',
                        default='',
                        type=str,
                        help='File with data about objects and object properties.')
    parser.add_argument('--n-points-per-object',
                        type=int,
                        default=-1,
                        help='Number of points to include in each point cloud.')
    parser.add_argument('--n-grasps-per-object',
                        type=int,
                        default=-1,
                        help='Number of grasps to label per object.')
    parser.add_argument('--object-ix',
                        type=int,
                        default=-1,
                        help='Only use the given object in the dataset.')
    parser.add_argument('--grasp-noise',
                        type=float,
                        default=0.0)
    args = parser.parse_args()
    print(args)

    if args.mode == 'objects':
        assert len(args.object_names) > 0
        assert args.n_property_samples > 0
        generate_objects(args)
    elif args.mode == 'grasps':
        assert len(args.objects_fname) > 0
        assert args.n_points_per_object > 0
        assert args.n_grasps_per_object > 0
        generate_datasets(args)
