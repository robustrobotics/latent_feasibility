import numpy as np
import torch


def bald(predictions, eps=1e-5):
    """ Get the BALD score for each example.
    :param predictions: (N, K) predictions for N datapoints from K models.
    :return: (N,) The BALD score for each of the datapoints.
    """
    mp_c1 = torch.mean(predictions, dim=1)
    mp_c0 = torch.mean(1 - predictions, dim=1)

    m_ent = -(mp_c1 * torch.log(mp_c1+eps) + mp_c0 * torch.log(mp_c0+eps))

    p_c1 = predictions
    p_c0 = 1 - predictions
    ent_per_model = p_c1 * torch.log(p_c1+eps) + p_c0 * torch.log(p_c0+eps)
    ent = torch.mean(ent_per_model, dim=1)

    bald = m_ent + ent

    return bald

def choose_acquisition_data(samples, gnp, n_acquire, strategy, data_pred_fn, data_subset_fn):
    """ Choose data points with the highest acquisition score
    :param samples: (N,2) An array of unlabelled datapoints which to evaluate.
    :param ensemble: A list of models.
    :param n_acquire: The number of data points to acquire.
    :param strategy: ['random', 'bald'] The objective to use to choose new datapoints.
    :param data_pred_fn: A handler to get predictions specific on the dataset type.
    :prarm data_subset_fn: A handler to select fewer datapoints.
    :return: (n_acquire, 2) - the samples which to label.
    """
    # Get the acquisition score for each.
    if strategy == 'bald':
        preds = data_pred_fn(samples, gnp)
        scores = bald(preds).cpu().numpy()
    elif strategy == 'random':
        scores = np.random.uniform(size=samples.shape[0]).astype('float32')

    # Return the n_acquire points with the highest score.
    acquire_indices = np.argsort(scores)[::-1][:n_acquire]
    return data_subset_fn(samples, acquire_indices)

def acquire_datapoints(gnp, n_samples, n_acquire, strategy, \
        data_sampler_fn, data_pred_fn, logger):
    """ Get new datapoints given the current GraspNP model.
    Uses function handlers for domain specific components (e.g., sampling unlabeled data).
    :param n_samples: How many unlabeled samples to generate.
    :param n_acquire: How many samples to acquire labels for.
    :param strategy: Which acquisition function to use.
    :param data_pred_fn:
    :return: (n_acquire, 2), (n_acquire,) - x,y tuples of the new datapoints.
    """
    # TODO: This function is very slow for ShapeNet data. Consider pre-generating ShapeNet data.
    unlabeled_pool = data_sampler_fn(n_samples)
    import IPython; IPython.embed()
    xs = choose_acquisition_data(
        unlabeled_pool,
        gnp,
        train_dataset,
        n_acquire,
        strategy,
        data_pred_fn
    )

    logger.save_unlabeled_acquisition_data(xs)

    new_data = data_label_fn(xs, logger)

    return new_data, unlabeled_pool
