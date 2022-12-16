import argparse
import os
import pickle
import time
import torch
from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.active_utils import sample_unlabeled_data_train
from learning.models.grasp_np.acquire import acquire_datapoints
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.grasp_neural_process import CustomGraspNeuralProcess
from learning.models.grasp_np.train_grasp_np import train


def select_initial_grasps(fname, n_grasps_per_object):
    """ Select a subset of grasps per object. """
    with open(fname, 'rb') as handle:
        grasp_data = pickle.load(handle)

    for data_field in grasp_data['grasp_data']:
        for ox in grasp_data['grasp_data'][data_field]:
            all_grasps = grasp_data['grasp_data'][data_field][ox]
            grasp_data['grasp_data'][data_field][ox] = all_grasps[:n_grasps_per_object]

    return grasp_data


def get_initial_dataset(data_root_fname, n_train_grasps_per_object, n_val_grasps_per_object):
    """ Return a subset of initial grasps to seed the active learning. """
    print(f'Loading an initial dataset from {data_root_fname}')
    data_root_path = f'learning/data/grasping/{data_root_fname}/grasps/training_phase'

    train_data_path = os.path.join(data_root_path, 'train_grasps.pkl')
    train_data = select_initial_grasps(train_data_path, n_train_grasps_per_object)
    train_dataset = CustomGNPGraspDataset(data=train_data)

    val_data_path = os.path.join(data_root_path, 'val_grasps.pkl')
    val_data = select_initial_grasps(val_data_path, n_val_grasps_per_object)
    val_dataset = CustomGNPGraspDataset(data=val_data, context_data=train_data)

    object_set = train_data['object_data']

    return train_dataset, val_dataset, object_set


def get_dataloaders(args, train_dataset, val_dataset):
    """ Wrap dataset in GNP dataloaders. """
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False
    )
    return train_dataloader, val_dataloader


def active_train(train_dataset, val_dataset, train_dataloader, val_dataloader, data_sampler_fn, data_pred_fn, logger, args):
    """ Main training function 
    :param train_dataset: Object containing the data to iterate over. Can be added to.
    :param val_dataset: 
    :param train_dataloader: The dataloader linked to the given dataset.
    :param val_dataloader:
    :param data_sampler_fn:
    :param data_label_fn:
    :param data_pred_fn:
    :param data_subset_fn:
    :param logger: Object used to keep track of training artifacts.
    :param agent: PandaAgent or None (if args.exec_mode == 'simple-model' or 'noisy-model')
    :param args: Commandline arguments such as the number of acquisition points.
    :return: The fully trained ensemble.
    """
    for tx in range(logger.acquisition_step, args.max_acquisitions):
        print('Acquisition Step: ', tx)
        start_time = time.time()
        if val_dataloader is not None:
            logger.save_val_dataset(val_dataset, tx)

        print('Training ensemble....')
        # Re-Initialize GraspNP at each loop.
        model = CustomGraspNeuralProcess(
            d_latents=args.d_latents,
            n_decoders=args.n_decoders
        )

        # Train GNP with current data.
        model = train(train_dataloader, val_dataloader, model, n_epochs=args.n_epochs)
        print('Done training.')

        logger.save_dataset(dataset=train_dataset, tx=tx)
        logger.save_val_dataset(val_dataset=val_dataset, tx=tx)
        logger.save_neural_process(gnp=model, tx=tx, symlink_tx0=False)

        # TODO: Collect new grasps.
        print('Collecting datapoints...')
        new_data, all_samples = acquire_datapoints(
            gnp=model,
            n_samples=args.n_samples_per_object,
            n_acquire=args.n_acquire_per_object,
            strategy=args.strategy,
            data_sampler_fn=data_sampler_fn,
            data_pred_fn=data_pred_fn,
            logger=logger
        )
        print('Done data collection.')
        logger.save_acquisition_data(new_data, None, tx)#new_data, all_samples, tx)

        # TODO: Add to dataset.
        train_data, val_data = split_data(new_data, n_val=2)
        train_dataset.add_to_dataset(train_data)
        val_dataset.add_to_dataset(val_data)

        print('Time: ' + str((time.time()-start_time)*(1/60)) + ' minutes')


def run_active_train(args):
    # Initial arg checks.
    args.use_latents = False
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Get an initial datasets/dataloaders.
    train_dataset, val_dataset, object_set = get_initial_dataset(
        args.data_root,
        n_train_grasps_per_object=args.init_grasps_per_object,
        n_val_grasps_per_object=args.init_grasps_per_object//5
    )
    train_dataloader, val_dataloader = get_dataloaders(args, train_dataset, val_dataset)

    # Get active learning helper functions.
    data_sampler_fn = lambda n: sample_unlabeled_data_train(n, object_set, train_dataset.object_indices)

    # TODO: Get prediction function used to generate predictions for BALD.
    data_pred_fn = lambda dataset, ensemble: get_predictions(
        dataset, ensemble, N_samples=5, use_latents=True,
        collapse_latents=collapse_latents, collapse_ensemble=collapse_ensemble, keep_latent_ix=keep_latent_ix)


    # Start training.
    print('Starting training from scratch.')
    active_train(train_dataset=train_dataset,
                 val_dataset=val_dataset,
                 train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader,
                 data_sampler_fn=data_sampler_fn,
                 data_pred_fn=data_pred_fn,
                 logger=logger,
                 args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='',
        help='Where results will be saved. Randon number if not specified.')

    # Active learning parameters.
    parser.add_argument('--max-acquisitions', type=int, default=1000,
        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--init-grasps-per-object', type=int, default=10)
    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--n-samples-per-object', type=int, default=25)
    parser.add_argument('--n-acquire-per-object', type=int, default=5)
    parser.add_argument('--strategy', choices=['random', 'bald'])

    # Model/training parameters.
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-decoders', type=int, default=5,
        help='Number of models in the ensemble.')
    parser.add_argument('--d-latents', type=int, default=5)
    parser.add_argument('--n-epochs', type=int, default=50)

    args = parser.parse_args()

    run_active_train(args)
