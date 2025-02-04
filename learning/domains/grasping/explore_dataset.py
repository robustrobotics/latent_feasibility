import argparse
import enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import trimesh
import time
import io
from matplotlib import cm
import pybullet as p
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

from learning.domains.grasping.generate_grasp_datasets import graspablebody_from_vector
from pb_robot.planners.antipodalGraspPlanner import Grasp, GraspSimulationClient, GraspSampler


def get_success_per_test_object(dataset_root, n_bins=10):
    p_stables = []
    n = 0
    for ox in range(100):
        fname = os.path.join(dataset_root, 'grasps', 'fitting_phase', f'fit_grasps_train_geo_object{ox}.pkl')
        with open(fname, 'rb') as handle:
            data = pickle.load(handle)
        p_stable = np.mean(data['grasp_data']['labels'])
        if p_stable >= 0.3 and p_stable <= 0.7:
            n += 1
        print(f'Object {ox}: {p_stable}')
        p_stables.append(p_stable)

    hist = np.histogram(p_stables, bins=n_bins)
    print(hist)
    print(n)
    plt.hist(p_stables, bins=n_bins)
    plt.show()


def inspect_train_dataset(dataset_root):
    train_fname = os.path.join(dataset_root, 'grasps', 'training_phase', 'train_grasps.pkl')
    with open(train_fname, 'rb') as handle:
        train_data = pickle.load(handle)

    object_names = train_data['object_data']['object_names']
    object_properties = train_data['object_data']['object_properties']

    grasps = train_data['grasp_data']['grasps']
    ids = train_data['grasp_data']['object_ids']
    labels = train_data['grasp_data']['labels']
    p_successes = []
    masses, frictions = [], []

    p_success_type = {}
    for ox, name in enumerate(object_names):
        ox_ids = np.array(ids) == ox
        ox_labels = np.array(labels)[ox_ids]
        if len(ox_labels) == 0:
            print('Skipping')
            continue

        p_success = np.mean(ox_labels)
        p_successes.append(p_success)
        if name in p_success_type:
            p_success_type[name].append(p_success)
        else:
            p_success_type[name] = [p_success]

        m, f = object_properties[ox][-2:]
        masses.append(m)
        frictions.append(f)

    print(p_successes)
    print(np.var(p_successes), np.mean([np.var(geom_successes) for _, geom_successes in p_success_type.items()]))
    hist = np.histogram(p_successes, bins=10)
    print(hist)
    plt.hist(p_successes, bins=10)
    plt.xlabel('Object-wise Grasp Success')
    plt.ylabel('Object Counts')
    fname = os.path.join(dataset_root, 'figures', 'success_hist.png')
    plt.savefig(fname)

    plt.clf()
    print('Length:', len(masses), len(frictions))
    plt.scatter(masses, frictions, c=p_successes, cmap=plt.get_cmap('viridis'))
    plt.xlabel('mass')
    plt.ylabel('friction')
    fname = os.path.join(dataset_root, 'figures', 'success_scatter.png')
    plt.savefig(fname)


def generate_object_grid(objects, dataset_figpath):
    names = objects['object_data']['object_names']
    properties = objects['object_data']['object_properties']

    n_objects = len(names)
    n_property_samples = objects['metadata'].n_property_samples
    n_geoms = n_objects // n_property_samples

    images = []
    for ox, (name, prop) in enumerate(zip(names, properties)):
        graspable_body = graspablebody_from_vector(name, prop)
        sim_client = GraspSimulationClient(graspable_body, False)

        axis = sim_client._get_tm_com_axis()
        scene = trimesh.scene.Scene([sim_client.mesh, axis])
        scene.set_camera(angles=(np.pi / 2, 0, np.pi / 4), distance=0.5, center=sim_client.mesh.centroid)
        data = scene.save_image()
        image = np.array(Image.open(io.BytesIO(data)))
        print(image.shape)
        images.append(image)
        sim_client.disconnect()

    for ix in range(0, n_geoms):
        images_for_geom = images[ix * n_property_samples:(ix + 1) * n_property_samples]
        names_for_geom = names[ix * n_property_samples:(ix + 1) * n_property_samples]
        properties_for_geom = properties[ix * n_property_samples:(ix + 1) * n_property_samples]

        plt.clf()
        fig = plt.figure(figsize=(20., 4.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 5),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im, name, prop in zip(grid, images_for_geom, names_for_geom, properties_for_geom):
            mass = prop[-2]
            friction = prop[-1]
            n = name.split('::')[1].split('_')[0]

            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.text(50, 375, f'{n}\nm={mass: .2f}\nf={friction: .2f}',
                    bbox=dict(fill=False, edgecolor='black', linewidth=1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        fname = os.path.join(dataset_figpath, f'geom{ix}_{n}.png')
        print(fname)
        plt.savefig(fname)


def visualize_grasp_dataset(dataset_fname, labels=None, figpath='', prefix=''):
    with open(dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)
    dataset_args = val_data['metadata']
    object_names = val_data['object_data']['object_names']
    object_properties = val_data['object_data']['object_properties']
    n_objects = len(object_names)

    grasp_vectors = val_data['grasp_data']['grasps']
    if labels is None:
        labels = val_data['grasp_data']['labels']

    for ix in range(n_objects):
        if ix not in [2, 3, 46, 49, 51, 54, 65, 68, 83, 170, 213, 249, 256, 258, 267, 286, 287, 292, 337, 338, 351, 396,
                      400, 401, 415, 435, 446, 457, 496]:
            continue
        if (ix + 1) * dataset_args.n_grasps_per_object > len(labels):
            break

        graspable_body = graspablebody_from_vector(object_names[ix], object_properties[ix])

        sim_client = GraspSimulationClient(graspable_body, False)
        grasps = []
        for gx in range(dataset_args.n_grasps_per_object):
            grasp_points = [
                grasp_vectors[ix * dataset_args.n_grasps_per_object + gx][0, 0:3],
                grasp_vectors[ix * dataset_args.n_grasps_per_object + gx][1, 0:3],
                grasp_vectors[ix * dataset_args.n_grasps_per_object + gx][2, 0:3]
            ]
            grasps.append(grasp_points)

        obj_labels = labels[ix * dataset_args.n_grasps_per_object:(ix + 1) * dataset_args.n_grasps_per_object]

        fname = os.path.join(figpath, f'{prefix}object{ix}.png')
        if len(figpath) > 0:
            sim_client.tm_show_grasps(grasps, obj_labels, fname=fname)
        else:
            sim_client.tm_show_grasps(grasps, obj_labels)
        sim_client.disconnect()


def visualize_fitting_acquisition(context_dataset, unlabeled_sample_set, ig_values, max_ig, figpath=''):
    # first we loop through keys in context/unlabeled for all objects currently being evaluated
    for obj_ix in context_dataset['grasp_data']['object_meshes'].keys():

        # construct graspable body
        graspable_body = graspablebody_from_vector(context_dataset['object_data']['object_names'][obj_ix],
                                                   context_dataset['object_data']['object_properties'][obj_ix])
        sim_client = GraspSimulationClient(graspable_body, False)

        # for now, let's just take one of the grasp points for vis for tm_show grasp
        grasps = []
        forces = []
        for grasp_points, force in zip(unlabeled_sample_set['grasp_data']['grasp_points'][obj_ix],
                                       unlabeled_sample_set['grasp_data']['grasp_forces'][obj_ix]):
            grasps.append(grasp_points[0, :])
            forces.append(force)
        sim_client.tm_show_labeled_and_acquired_grasps(
            labeled_grasps=context_dataset['grasp_data']['grasp_points'][obj_ix],
            labels=context_dataset['grasp_data']['labels'][obj_ix],
            labeled_forces=context_dataset['grasp_data']['grasp_forces'][obj_ix],
            acquired_grasps=unlabeled_sample_set['grasp_data']['grasp_points'][obj_ix],
            acquired_info_gain=ig_values,
            acquired_forces=unlabeled_sample_set['grasp_data']['grasp_forces'][obj_ix],
            max_ig=max_ig,
            fname=figpath
        )
        sim_client.disconnect()


def analyze_objects(objects_all):
    objects = objects_all['object_data']
    grasps = objects_all['grasp_data']['labels']
    inertias = []
    volumes = []
    BAD = [2, 3, 46, 49, 51, 54, 65, 68, 83, 170, 213, 249, 256, 258, 267, 286, 287, 292, 337, 338, 351, 396, 400, 401,
           415, 435, 446, 457, 496]
    bad_count = 0
    for ox, (name, props) in enumerate(zip(objects['object_names'], objects['object_properties'])):
        print(ox)
        # if ox > 50: break
        graspable_body = graspablebody_from_vector(name, props)
        sim_client = GraspSimulationClient(graspable_body=graspable_body,
                                           show_pybullet=False,
                                           recompute_inertia=True)
        inertia = p.getDynamicsInfo(sim_client.body_id, -1, sim_client.pb_client_id)[2]
        inertias.append(inertia)
        # if np.any(np.array(inertia) < 1e-4) or np.any(np.array(inertia) > 1e-2):
        #     print('SMALL OR LARGE INERTIA', ox, name, props, inertia)
        avg_correct = np.mean(grasps[ox * 50:(ox + 1) * 50])
        bounding_box = sim_client.mesh.bounding_box
        # if sim_client.mesh.volume > 0.00025 and avg_correct > 0.05 and avg_correct < 0.95:
        #     print(ox, sim_client.mesh.volume, avg_correct)
        if sim_client.mesh.volume > 0.00025 and ox in BAD:
            bad_count += 1
            print('BAD:', bad_count, len(BAD))
        volumes.append(sim_client.mesh.volume)
        # import IPython; IPython.embed(); sys.exit();
        sim_client.disconnect()
    inertia = np.array(inertias)
    # plt.hist(inertia.flatten(), bins=100)
    plt.hist(volumes, bins=100)
    # plt.xscale('log')
    plt.show()
    print(np.min(inertia), np.max(inertia))

import multiprocessing as mp
def get_object_stats(name, props):
    graspable_body = graspablebody_from_vector(name, props)
    sim_client = GraspSimulationClient(graspable_body=graspable_body,
        show_pybullet=False,
        recompute_inertia=True)
    mesh = sim_client.mesh
    volume = mesh.volume
    bb_volume = mesh.convex_hull.volume
    ratio = bb_volume / volume
    # if ratio > 4:
    #     mesh.show()
    max_dim = np.max(mesh.bounds[1]*2)
    sim_client.disconnect()
    return [bb_volume, volume, max_dim, ratio]

def get_effective_sampling_size(name, props):
    """
    For each object, approximate how diverse the set of grasps is.
    """
    graspable_body = graspablebody_from_vector(name, props)
    grasp_sampler = GraspSampler(
        graspable_body=graspable_body,
        antipodal_tolerance=30
    )
    grasps, rejection_rate = grasp_sampler.sample_grasps(50, (5, 20))
    grasp_sampler.disconnect()

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
    if len(grasps) == 0:
        avg_min_dist = -1

    print(rejection_rate, avg_min_dist)
    return (rejection_rate, avg_min_dist)

def get_effective_sampling_size_from_metadata(names):
    metadata_path = os.path.join(os.environ['SHAPENET_ROOT'], 'object_infos.pkl')
    with open(metadata_path, 'rb') as handle:
        metadata = pickle.load(handle)

    results = []
    for name in names:
        obj_info = metadata[name]
        results.append((obj_info['rejection_rate'], obj_info['avg_min_dist50']))
    return results


def show_object_effective_sampling_sizes(train_fname):
    with open(train_fname, 'rb') as handle:
        train_data = pickle.load(handle)

    object_names = train_data['object_data']['object_names'][::5]
    object_properties = train_data['object_data']['object_properties'][::5]

    # pool = mp.Pool(processes=1)
    # results = pool.starmap(get_effective_sampling_size_from_metadata, zip(object_names, object_properties))
    results = get_effective_sampling_size_from_metadata(object_names)
    # for name, prop in zip(object_names[::5], object_properties[::5]):
    #    get_object_stats(name, prop)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].hist([res[0] for res in results], bins=25)
    axes[0].set_xlabel('Rejection Rate')
    axes[0].set_ylabel('Count')

    axes[1].hist([res[1] for res in results if res[1] > 0], bins=25)
    axes[1].set_xlabel('Avg Min Dist')
    axes[1].set_ylabel('Count')

    # plt.show()
    plt.savefig('/home/mnosew/snv2_hist.png')

def show_object_size_distributions(train_fname):
    with open(train_fname, 'rb') as handle:
        train_data = pickle.load(handle)

    object_names = train_data['object_data']['object_names']
    object_properties = train_data['object_data']['object_properties']

    pool = mp.Pool(processes=20)
    results = pool.starmap(get_object_stats, zip(object_names, object_properties))
    #for name, prop in zip(object_names[::5], object_properties[::5]):
    #    get_object_stats(name, prop)

    fig, axes = plt.subplots(nrows=4, ncols=1)
    axes[0].hist([res[0] for res in results], bins=25)
    axes[0].set_xlabel('BB Volume')
    axes[0].set_ylabel('Count')

    axes[1].hist([res[1] for res in results], bins=25)
    axes[1].set_xlabel('Volume')
    axes[1].set_ylabel('Count')

    axes[2].hist([res[2] for res in results], bins=25)
    axes[2].set_xlabel('Max Dimension')
    axes[2].set_ylabel('Count')

    axes[3].hist([res[3] for res in results], bins=25)
    axes[3].set_xlabel('Ratio')
    axes[3].set_ylabel('Count')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()
    print(args)

    dataset_root = os.path.join('learning', 'data', 'grasping', args.dataset_name)

    dataset_figpath = os.path.join(dataset_root, 'figures')
    if not os.path.exists(dataset_figpath):
        os.mkdir(dataset_figpath)
    object_meshes_path = os.path.join(dataset_figpath, 'object_meshes')
    if not os.path.exists(object_meshes_path):
        os.mkdir(object_meshes_path)

    train_objects_path = os.path.join(dataset_root, 'objects', 'train_geo_train_props.pkl')
    with open(train_objects_path, 'rb') as handle:
        train_objects = pickle.load(handle)
    train_meshes_path = os.path.join(object_meshes_path, 'train_meshes')
    if not os.path.exists(train_meshes_path):
        os.mkdir(train_meshes_path)
    # generate_object_grid(train_objects, train_meshes_path)

    val_objects_path = os.path.join(dataset_root, 'objects', 'test_geo_test_props.pkl')
    with open(val_objects_path, 'rb') as handle:
        val_objects = pickle.load(handle)
    val_meshes_path = os.path.join(object_meshes_path, 'val_meshes')
    if not os.path.exists(val_meshes_path):
        os.mkdir(val_meshes_path)
    # generate_object_grid(val_objects, val_meshes_path)

    # inspect_train_dataset(dataset_root)
    # sys.exit()

    train_grasps = os.path.join(dataset_root, 'grasps', 'training_phase', 'train_grasps.pkl')
    show_object_effective_sampling_sizes(train_grasps)
    sys.exit()
    show_object_size_distributions(train_grasps)
    figpath = os.path.join(dataset_figpath, 'train_grasps')
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    with open(train_grasps, 'rb') as handle:
        train_grasps = pickle.load(handle)
    analyze_objects(train_grasps)
    sys.exit()
    # visualize_grasp_dataset(train_grasps, figpath=figpath)

    # get_success_per_test_object(dataset_root)
