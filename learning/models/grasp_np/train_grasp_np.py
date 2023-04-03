import argparse
import copy

import numpy as np
import pickle
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.grasp_neural_process import CustomGraspNeuralProcess


def check_to_cuda(tensor_list):
    if torch.cuda.is_available():
        return [tensor.cuda() for tensor in tensor_list]
    else:
        return tensor_list


def get_accuracy(y_probs, target_ys, test=False, save=False):
    assert (y_probs.shape == target_ys.shape)
    if test == True:
        if y_probs.shape[0] > 100000:
            n_grasps = 50
        else:
            n_grasps = 10
        per_obj_probs = y_probs.view(-1, n_grasps)
        per_obj_target = target_ys.view(-1, n_grasps)
        per_obj_acc = ((per_obj_probs > 0.5) == per_obj_target).float().mean(dim=1)
        print(per_obj_probs.shape, per_obj_acc.shape)
        if save:
            with open('learning/experiments/metadata/grasp_np/accs.pkl', 'wb') as handle:
                pickle.dump((per_obj_acc, per_obj_target), handle)
                print(per_obj_probs.shape)
        print('HIST:', np.histogram(per_obj_acc.cpu(), bins=10))
        if save:
            with open('learning/experiments/metadata/grasp_np/results_val.pkl', 'wb') as handle:
                pickle.dump((y_probs.cpu().numpy(), target_ys.cpu().numpy()), handle)

    acc = ((y_probs > 0.5) == target_ys).float().mean()
    return acc


def get_loss(y_probs, target_ys, q_z, q_z_n, alpha=1, use_informed_prior=True, bce_scale_factor=1.0):
    bce_loss = F.binary_cross_entropy(y_probs.squeeze(), target_ys.squeeze(), reduction='sum')

    # beta = min(alpha / 100., 1)
    # beta = 1. / (1 + np.exp(-0.05 * (alpha - 50)))  # TODO: Is this still necessary?
    beta = 1
    if use_informed_prior:
        kld_loss = beta * torch.distributions.kl_divergence(q_z, q_z_n).sum()
    else:
        p_z = torch.distributions.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
        kld_loss = beta * torch.distributions.kl_divergence(q_z, p_z).sum()
    # kld_loss.fill_(0.)
    bce_loss = bce_loss * bce_scale_factor
    # kld_loss = 0
    # weight = (1 + alpha)
    return bce_loss + alpha*kld_loss, bce_loss, kld_loss


def train(train_dataloader, val_dataloader, model, logger, n_epochs=10, use_informed_prior=True):
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 10000
    best_weights = None

    val_loss_bce_scale_factor = float(len(train_dataloader.dataset.hp_grasp_geometries[5])) / \
                                 len(val_dataloader.dataset.hp_grasp_geometries[5])
    print('Scale Factor:', val_loss_bce_scale_factor)
    alpha = 0.01
    for ep in range(n_epochs):
        print(f'----- Epoch {ep} -----')
        if ep > 10:
            alpha += 0.1
        if alpha > 1.0:
            alpha = 1.0


        epoch_loss, epoch_bce, epoch_kld, train_probs, train_targets = 0, 0, 0, [], []
        model.train()
        for bx, (context_data, target_data, object_data) in enumerate(train_dataloader):
            # sample a t \in \{0, ..., max_grasps_per_object}
            # sampling t data points (grasps + geoms + labels) from batched objects (make sure the
            # sampling we choose per object have the same indices
            # REMEMBER THEM!!!

            c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels = check_to_cuda(context_data)
            t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces, t_labels = check_to_cuda(target_data)
            c_sizes = torch.ones_like(c_forces)*c_forces.shape[1]/50
            
            meshes, object_properties = check_to_cuda(object_data)

            # sample a sub collection of the target set to better represent model adaptation phase in training
            max_n_grasps = c_grasp_geoms.shape[1]
            # max_n_grasps = 25
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
            n_c_sizes = torch.ones_like(n_c_forces)*n_grasps/50
            
            optimizer.zero_grad()

            # pass forward for max_n_grasps
            y_probs, q_z = model.forward(
                (c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels, c_sizes),
                (t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces),
                (meshes, object_properties)
            )
            y_probs = y_probs.squeeze()

            # pass forward for n_grasps (but the encoder ONLY)
            q_z_n, _, _ = model.forward_until_latents(
                (n_c_grasp_geoms, n_c_grasp_points, n_c_curvatures, n_c_normals, n_c_midpoints, n_c_forces, n_c_labels, n_c_sizes),
                meshes)

            if np.random.rand() > 0.02: ep_use_informed_prior = use_informed_prior
            else: ep_use_informed_prior = False
            loss, bce_loss, kld_loss = get_loss(y_probs, t_labels, q_z, q_z_n,
                                                alpha=alpha, use_informed_prior=ep_use_informed_prior)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += bce_loss.item()
            epoch_kld += kld_loss.item()
            train_probs.append(y_probs.flatten())
            train_targets.append(t_labels.flatten())
        
        epoch_loss /= len(train_dataloader.dataset)
        epoch_bce /= len(train_dataloader.dataset)
        epoch_kld /= len(train_dataloader.dataset)
        train_acc = get_accuracy(
            torch.cat(train_probs).flatten(),
            torch.cat(train_targets).flatten()
        )
        print(f'Train Loss: {epoch_loss}\tBCE: {epoch_bce}\tKLD: {epoch_kld}\tTrain Acc: {train_acc}')

        model.eval()
        means = []
        val_loss, val_kld, val_bce, val_probs, val_targets = 0, 0, 0, [], []
        with torch.no_grad():
            for bx, (context_data, target_data, object_data) in enumerate(val_dataloader):
                
                c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels = \
                    check_to_cuda(context_data)
                c_sizes = torch.ones_like(c_forces)*c_forces.shape[1]/50
                t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces, t_labels = \
                    check_to_cuda(target_data)
                meshes, object_properties = check_to_cuda(object_data)

                # sample a sub collection of the target set to better represent model adaptation phase in training
                max_n_grasps = c_grasp_geoms.shape[1]
                # max_n_grasps = 25
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
                n_c_sizes = torch.ones_like(n_c_forces)*n_grasps/50

                y_probs, q_z = model.forward(
                    (c_grasp_geoms, c_grasp_points, c_curvatures, c_normals, c_midpoints, c_forces, c_labels, c_sizes),
                    (t_grasp_geoms, t_grasp_points, t_curvatures, t_normals, t_midpoints, t_forces),
                    (meshes, object_properties)
                )
                means.append(q_z.loc)
                y_probs = y_probs.squeeze()

                q_z_n, _, _ = model.forward_until_latents(
                    (n_c_grasp_geoms, n_c_grasp_points, n_c_curvatures, n_c_normals, n_c_midpoints, n_c_forces, n_c_labels, n_c_sizes),
                    meshes)

                batch_loss, batch_bce, batch_kld = get_loss(y_probs, t_labels, q_z, q_z_n,
                                     use_informed_prior=use_informed_prior,
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
            val_acc = get_accuracy(
                torch.cat(val_probs),
                torch.cat(val_targets),
                test=True, save=True
            )
            print(f'Val Loss: {val_loss}\tBCE: {val_bce}\tKLD: {val_kld}\tVal Acc: {val_acc}')

            if ep > 20 and val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                logger.save_neural_process(gnp=model, tx=0, symlink_tx0=False)
                print('New best loss: ', val_loss)
        # if val_acc > 0.9:
        #     import IPython; IPython.embed()

    model.load_state_dict(best_weights)
    return model


def print_dataset_stats(dataset, name):
    print(f'----- {name} Dataset Statistics -----')
    print(f'N: {len(dataset)}')
    print(f'Context Shape: {dataset.contexts[0].shape}')
    print(f'Target xs Shape: {dataset.target_xs[0].shape}')
    print(f'Target ys Shape: {dataset.target_xs[0].shape}')


def run(args):
    # set up logger  # args.exp_name
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # build the model # args.5
    model = CustomGraspNeuralProcess(
        d_latents=args.d_latents,
        input_features={
            'mesh_normals': args.add_mesh_normals,  # For both object and grasp meshes.
            'mesh_curvatures': args.add_mesh_curvatures,
            'grasp_normals': True,  # Including in the grasp feature vector post-mesh processing.
            'grasp_curvatures': True,
            'grasp_mesh': True,  # Whether to process local grasp point clouds.
            'object_properties': False  # Latent (False) or ground truth (True)
        },
        d_grasp_mesh_enc=16,
        d_object_mesh_enc=16
    )

    # load datasets
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
        batch_size=args.batch_size,
        collate_fn=lambda x: custom_collate_fn(x, add_mesh_normals=args.add_mesh_normals),
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset_eval,
        collate_fn=lambda x: custom_collate_fn(x, add_mesh_normals=args.add_mesh_normals),
        batch_size=args.batch_size,
        shuffle=False
    )

    # train model
    model = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        logger=logger,
        n_epochs=args.n_epochs,
        use_informed_prior=args.informed_prior_loss
    )

    # save model
    logger.save_dataset(dataset=train_dataset, tx=0)
    logger.save_val_dataset(val_dataset=val_dataset_eval, tx=0)
    logger.save_neural_process(gnp=model, tx=0, symlink_tx0=False)

    return logger.exp_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset-fname', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--d-latents', type=int, required=True)
    parser.add_argument('--n-epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args()
    args.use_latents = False  # NOTE: this is for the specific workaround for block stacking that assumes
    # a different NN architecture
    run(args)
