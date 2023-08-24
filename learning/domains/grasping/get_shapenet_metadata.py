import argparse
import numpy as np
import os
import pickle
from learning.domains.grasping.generate_datasets_for_experiment import parse_ignore_file, get_object_list
from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector, sample_grasps_and_Xs
import multiprocessing as mp
from pb_robot.planners.antipodalGraspPlanner import Grasp, GraspSimulationClient, GraspSampler


def get_object_stats(name):
    graspable_body = graspablebody_from_vector(name, np.array([0.0, 0.0, 0.0, 0.1, 0.1]))
    grasp_sampler = GraspSampler(
        graspable_body=graspable_body,
        antipodal_tolerance=30
    )
    mesh = grasp_sampler.sim_client.mesh
    volume = mesh.volume
    ch_volume = mesh.convex_hull.volume
    bb_volume = mesh.bounding_box.volume
    bounds = mesh.bounds
    # ratio = bb_volume / volume
    # max_dim = np.max(mesh.bounds[1]*2)
    grasps, rejection_rate = grasp_sampler.sample_grasps(50, (5, 20), check_rejection_rate=True)
    grasp_sampler.disconnect()

    if len(grasps) > 0:
        all_midpoints = np.array([(g.pb_point1 + g.pb_point2)/2.0 for g in grasps])
        dists_to_closest = []
        for gx, midpoint in enumerate(all_midpoints):
            other_points = np.concatenate(
                [all_midpoints[:gx, :], all_midpoints[gx + 1:, :]],
                axis=0
            )
            dists = np.linalg.norm(midpoint - other_points, axis=1)
            dists_to_closest.append(np.min(dists))

        avg_min_dist = np.mean(dists_to_closest)
    else:
        avg_min_dist = -1

    return {
        'bb_volume': bb_volume,
        'ch_volume': ch_volume,
        'volume': volume,
        'bounds': bounds,
        'rejection_rate': rejection_rate,
        'avg_min_dist50': avg_min_dist
    }

dataset = 'shapenet'
if __name__ == '__main__':
    if dataset == 'shapenet':
        dataset_root = os.environ['SHAPENET_ROOT']
        urdf_base = 'ShapeNet'
    else:
        dataset_root = os.environ['PRIMITIVE_ROOT']
        urdf_base = 'Box'
    object_names = []
    for fname in os.listdir(os.path.join(dataset_root, 'urdfs')):
        name = f'{urdf_base}::{fname.split(".urdf")[0]}'
        object_names.append(name)

    import ipdb; ipdb.set_trace()
    all_infos = {}

    batch_size = 100
    for ix in range(0, len(object_names)//batch_size):
        print(f'Batch {ix}')
        batch_names = object_names[ix*batch_size:(ix+1)*batch_size]
        worker_pool = mp.Pool(processes=10)
        object_infos = worker_pool.map(get_object_stats, batch_names )
        worker_pool.close()

        for name, info in zip(batch_names, object_infos):
            all_infos[name] = info

        # with open(os.path.join(dataset_root, 'object_infos.pkl'), 'wb') as handle:
        #     pickle.dump(all_infos, handle)
