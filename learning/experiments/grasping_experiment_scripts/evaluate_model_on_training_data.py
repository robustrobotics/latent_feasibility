from learning.active.utils import ActiveExperimentLogger
from learning.experiments.grasping_experiment_scripts.run_grasping_experiments import EXPERIMENT_ROOT

from sklearn.metrics import precision_score, recall_score, average_precision_score
import torch
from torch.utils.data import DataLoader
import argparse
import json
import pickle
import os

from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.train_grasp_np import check_to_cuda, get_loss


def main(args, logger, perturb_rad):
    # load model and data
    model = logger.get_neural_process(0)
    model = model.cuda() if torch.cuda.is_available() else model

    with open(args.train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    with open(args.val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)

    train_dataset = CustomGNPGraspDataset(
        data=train_data,
        add_mesh_curvatures=args.add_mesh_curvatures,
        add_mesh_normals=args.add_mesh_normals
    )
    val_dataset_eval = CustomGNPGraspDataset(
        data=val_data, context_data=train_data,
        add_mesh_curvatures=args.add_mesh_curvatures,
        add_mesh_normals=args.add_mesh_normals
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,  # number of grasps doesn't matter, since we are only using known prop anyway
        collate_fn=lambda x: custom_collate_fn(x, rand_grasp_num=False, add_mesh_normals=args.add_mesh_normals),
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset_eval,
        collate_fn=lambda x: custom_collate_fn(x, rand_grasp_num=False, add_mesh_normals=args.add_mesh_normals),
        batch_size=args.batch_size,
        shuffle=False
    )

    # some initial parameters to re-simulate training
    alpha = 0.0 if model.input_features['object_properties'] else 1.0
    val_loss_bce_scale_factor = float(len(train_dataloader.dataset.hp_grasp_geometries[5])) / \
                                len(val_dataloader.dataset.hp_grasp_geometries[5])

    # pass through dataset though model
    model.eval()
    means = []
    train_loss, train_kld, train_bce, train_probs, train_targets = 0, 0, 0, [], []
    with torch.no_grad():
        for bx, (context_data, target_data, object_data) in enumerate(train_dataloader):

            c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels = \
                check_to_cuda(context_data)
            c_sizes = torch.ones_like(c_forces) * c_forces.shape[1] / 50
            t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces, t_labels = \
                check_to_cuda(target_data)
            meshes, object_properties = check_to_cuda(object_data)

            # sample a sub collection of the target set to better represent model adaptation phase in training
            max_n_grasps = c_grasp_geoms.shape[1]
            max_n_grasps = 25
            n_grasps = torch.randint(low=1, high=max_n_grasps, size=(1,))
            if torch.cuda.is_available():
                n_grasps = n_grasps.cuda()

            # select random indices used for this evaluation and then select data from arrays
            n_indices = torch.randperm(max_n_grasps)[:n_grasps]
            n_c_grasp_geoms = c_grasp_geoms[:, n_indices, :, :]
            n_c_grasp_points = c_grasp_points[:, n_indices, :, :]
            n_c_curvatures = c_curvatures[:, n_indices, :, :]
            n_c_normals = c_normals[:, n_indices, :, :]
            n_c_midpoints = c_midpoints[:, n_indices, :]
            n_c_forces = c_forces[:, n_indices]
            n_c_labels = c_labels[:, n_indices]
            n_c_sizes = torch.ones_like(n_c_forces) * n_grasps / 50

            perturbation = 2 * (torch.rand(5) - 0.5) * torch.tensor(perturb_rad)

            y_probs, q_z = model.forward(
                (c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels, c_sizes),
                (t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces),
                (meshes, object_properties),
                known_property_perturbation=perturbation
            )
            means.append(q_z.loc)
            y_probs = y_probs.squeeze()

            q_z_n, _, _ = model.forward_until_latents(
                (n_c_grasp_geoms, n_c_grasp_points, n_c_curvatures, n_c_normals, n_c_midpoints, n_c_forces, n_c_labels,
                 n_c_sizes),
                meshes)

            batch_loss, batch_bce, batch_kld = get_loss(y_probs, t_labels, q_z, q_z_n,
                                                        use_informed_prior=True,
                                                        alpha=alpha)
            train_loss += batch_loss.item()
            train_bce += batch_bce.item()
            train_kld += batch_kld.item()

            train_probs.append(y_probs.flatten())
            train_targets.append(t_labels.flatten())

        train_loss /= len(train_dataloader.dataset)
        train_bce /= len(train_dataloader.dataset)
        train_kld /= len(train_dataloader.dataset)
        train_probs_all = torch.cat(train_probs).cpu()
        train_preds_all = (train_probs_all > 0.5).float()
        train_targets_all = torch.cat(train_targets).cpu()
        train_prec = precision_score(y_true=train_targets_all, y_pred=train_preds_all)
        train_recall = recall_score(y_true=train_targets_all, y_pred=train_preds_all)
        train_avg_prec = average_precision_score(y_true=train_targets_all, y_score=train_probs_all)

        print(
            f'Train Loss: {train_loss}\tBCE: {train_bce}\tKLD: {train_kld}'
            f'\tTrain Precision: {train_prec}\tTrain Recall: {train_recall}\tAvg Prec: {train_avg_prec}'
        )

    means = []
    val_loss, val_kld, val_bce, val_probs, val_targets = 0, 0, 0, [], []
    with torch.no_grad():
        for bx, (context_data, target_data, object_data) in enumerate(val_dataloader):

            c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels = \
                check_to_cuda(context_data)
            c_sizes = torch.ones_like(c_forces) * c_forces.shape[1] / 50
            t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces, t_labels = \
                check_to_cuda(target_data)
            meshes, object_properties = check_to_cuda(object_data)

            # sample a sub collection of the target set to better represent model adaptation phase in training
            max_n_grasps = c_grasp_geoms.shape[1]
            max_n_grasps = 25
            n_grasps = torch.randint(low=1, high=max_n_grasps, size=(1,))
            if torch.cuda.is_available():
                n_grasps = n_grasps.cuda()
            # select random indices used for this evaluation and then select data from arrays
            n_indices = torch.randperm(max_n_grasps)[:n_grasps]
            n_c_grasp_geoms = c_grasp_geoms[:, n_indices, :, :]
            n_c_grasp_points = c_grasp_points[:, n_indices, :, :]
            n_c_curvatures = c_curvatures[:, n_indices, :, :]
            n_c_normals = c_normals[:, n_indices, :, :]
            n_c_midpoints = c_midpoints[:, n_indices, :]
            n_c_forces = c_forces[:, n_indices]
            n_c_labels = c_labels[:, n_indices]
            n_c_sizes = torch.ones_like(n_c_forces) * n_grasps / 50

            perturbation = 2 * (torch.rand(5) - 0.5) * torch.tensor(perturb_rad)

            y_probs, q_z = model.forward(
                (c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels, c_sizes),
                (t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces),
                (meshes, object_properties),
                known_property_perturbation=perturbation
            )
            means.append(q_z.loc)
            y_probs = y_probs.squeeze()

            q_z_n, _, _ = model.forward_until_latents(
                (n_c_grasp_geoms, n_c_grasp_points, n_c_curvatures, n_c_normals, n_c_midpoints, n_c_forces, n_c_labels,
                 n_c_sizes),
                meshes)

            batch_loss, batch_bce, batch_kld = get_loss(y_probs, t_labels, q_z, q_z_n,
                                                        use_informed_prior=True,
                                                        bce_scale_factor=val_loss_bce_scale_factor,
                                                        alpha=alpha)
            val_loss += batch_loss.item()
            val_bce += batch_bce.item()
            val_kld += batch_kld.item()

            val_probs.append(y_probs.flatten())
            val_targets.append(t_labels.flatten())

        val_loss /= len(val_dataloader.dataset)
        val_bce /= len(val_dataloader.dataset)
        val_kld /= len(val_dataloader.dataset)
        val_probs_all = torch.cat(val_probs).cpu()
        val_preds_all = (val_probs_all > 0.5).float()
        val_targets_all = torch.cat(val_targets).cpu()
        val_prec = precision_score(y_true=val_targets_all, y_pred=val_preds_all)
        val_recall = recall_score(y_true=val_targets_all, y_pred=val_preds_all)
        val_avg_prec = average_precision_score(y_true=val_targets_all, y_score=val_probs_all)
        print(
            f'Val Loss: {val_loss}\tBCE: {val_bce}\tKLD: {val_kld}\t'
            f'Val Precision: {val_prec}\tVal Recall: {val_recall}\tVal Avg Prec: {val_avg_prec}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--uniform-perturb-rad', type=float, nargs=5, required=True, default=(0.0, 0.0, 0.0, 0.0, 0.0))
    script_args = parser.parse_args()

    # obtain the logger and the training args to initialize model
    metadata_exp_path = os.path.join(EXPERIMENT_ROOT, script_args.exp_name)
    if not os.path.exists(metadata_exp_path):
        print('Experiment does not exist: ' + script_args.exp_name)

    log_lookup_path = os.path.join(metadata_exp_path, 'logs_lookup.json')
    with open(log_lookup_path, 'r') as handle:
        log_lookup = json.load(handle)

    exp_path = log_lookup['training_phase']
    logger = ActiveExperimentLogger(exp_path, use_latents=True)
    training_args = logger.args
    main(training_args, logger, script_args.uniform_perturb_rad)
