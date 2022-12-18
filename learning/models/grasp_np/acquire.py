import numpy as np
import torch

from learning.models.grasp_np.train_grasp_np import check_to_cuda

def bald(predictions, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    mp_c1 = torch.mean(predictions, dim=-1)
    mp_c0 = torch.mean(1 - predictions, dim=-1)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=-1)

    bald = m_ent + ent

    return bald

def get_gnp_predictions(gnp, pool_dataloader, n_latent_samples=1, grasp_batch_size=10):
    """
    Return a vector in which we can compute the BALD metric over.
    :return: (N_objects X N_grasps X (n_latent_samples*n_ensembles))
    """
    # TODO: Add option to have multiple latent samples.
    with torch.no_grad():
        all_preds = []
        for context_data, target_data, meshes in pool_dataloader:
            # Sample latents.
            c_grasp_geoms, c_midpoints, c_forces, c_labels = check_to_cuda(context_data)
            if torch.cuda.is_available():
                meshes = meshes.cuda()
            q_z = gnp.encoder_forward(
                (c_grasp_geoms, c_midpoints, c_forces, c_labels),
                meshes
            )

            # Get predictions for each grasp.
            batch_preds = []
            n_target_grasps = target_data[0].shape[1]
            n_target_batches = int(np.ceil(n_target_grasps/grasp_batch_size))
            for bgx in range(0, n_target_batches):
                t_grasp_geoms = target_data[0][:, bgx*grasp_batch_size:(bgx+1)*grasp_batch_size, :, :]
                t_midpoints = target_data[1][:, bgx*grasp_batch_size:(bgx+1)*grasp_batch_size, :]
                t_forces = target_data[2][:, bgx*grasp_batch_size:(bgx+1)*grasp_batch_size]      
                target_xs = \
                    check_to_cuda((t_grasp_geoms, t_midpoints, t_forces))

                grasp_batch_preds = []
                for dx in range(0, gnp.n_decoders):
                    zs = q_z.rsample()
                    y_preds = gnp.conditional_forward(target_xs, meshes, zs, decoder_ix=dx)
                    grasp_batch_preds.append(y_preds)
                batch_preds.append(torch.cat(grasp_batch_preds, dim=2))
            all_preds.append(torch.cat(batch_preds, dim=1))
        return torch.cat(all_preds, dim=0)

def choose_acquisition_data(gnp, pool_dataloader, n_acquire, strategy):
    """ Choose data points with the highest acquisition score. """
    # Get the acquisition score for each.
    if strategy == 'bald':
        preds = get_gnp_predictions(gnp, pool_dataloader, n_latent_samples=2)
        scores = bald(preds).cpu().numpy()
    elif strategy == 'random':
        n_objects = len(pool_dataloader.dataset)
        n_grasps = pool_dataloader.dataset[0][1]['grasp_labels'].shape[0]
        scores = np.random.uniform(size=(n_objects, n_grasps)).astype('float32')

    # Return the n_acquire points with the highest score.
    acquire_indices = np.argsort(scores)[:, ::-1][:, :n_acquire]
    return acquire_indices
