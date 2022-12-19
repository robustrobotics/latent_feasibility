import argparse
import copy
import os
import pickle
import time
import torch
from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.domains.grasping.active_utils import add_grasps, remove_grasps
from learning.models.grasp_np.acquire import choose_acquisition_data
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.grasp_neural_process import CustomGraspNeuralProcess
from learning.models.grasp_np.train_grasp_np import train


def select_initial_grasps(fname, n_grasps_per_object):
    """ Select a subset of grasps per object. Remaining grasps go to unlabeled pool. """
    with open(fname, 'rb') as handle:
        grasp_data = pickle.load(handle)
    print(grasp_data.keys())
    pool_grasp_data = {
        'grasp_data': {
            'object_meshes': grasp_data['grasp_data']['object_meshes']
        },
        'object_data': grasp_data['object_data'],
        'metadata': grasp_data['metadata']
    }
    for data_field in grasp_data['grasp_data']:
        if data_field == 'object_meshes': continue
        pool_grasp_data['grasp_data'][data_field] = {}
        for ox in grasp_data['grasp_data'][data_field]:
            all_grasps = grasp_data['grasp_data'][data_field][ox]
            grasp_data['grasp_data'][data_field][ox] = all_grasps[:n_grasps_per_object]
            pool_grasp_data['grasp_data'][data_field][ox] = all_grasps[n_grasps_per_object:] 
    return grasp_data, pool_grasp_data


def get_initial_dataset(data_root_fname, n_train_grasps_per_object, n_val_grasps_per_object):
    """ Return a subset of initial grasps to seed the active learning. """
    print(f'Loading an initial dataset from {data_root_fname}')
    data_root_path = f'learning/data/grasping/{data_root_fname}/grasps/training_phase'

    train_data_path = os.path.join(data_root_path, 'train_grasps.pkl')
    train_dict, pool_dict = select_initial_grasps(train_data_path, n_train_grasps_per_object)

    val_data_path = os.path.join(data_root_path, 'val_grasps.pkl')
    val_dict, _ = select_initial_grasps(val_data_path, n_val_grasps_per_object)

    object_set = train_dict['object_data']

    return train_dict, val_dict, pool_dict, object_set


def get_dataloaders(args, train_dict, val_dict, pool_dict):
    """ Wrap dataset in GNP dataloaders. """
    train_dataset = CustomGNPGraspDataset(data=train_dict)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True
    )
    val_dataset = CustomGNPGraspDataset(data=val_dict, context_data=train_dict)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False
    )
    pool_dataset = CustomGNPGraspDataset(data=pool_dict, context_data=train_dict)
    pool_dataloader = DataLoader(
        dataset=pool_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False
    )
    return train_dataloader, val_dataloader, pool_dataloader


def active_train(train_dict, val_dict, pool_dict, logger, args):
    """ Main training function
    :param train_dict: Object containing the data to iterate over. Can be added to.
    :param val_dict:
    :param pool_dict:
    :param logger: Object used to keep track of training artifacts.
    :param args: Commandline arguments such as the number of acquisition points.
    :return: The fully trained GNP.
    """
    for tx in range(logger.acquisition_step, args.max_acquisitions):
        print('Acquisition Step: ', tx)
        start_time = time.time()

        print('Training ensemble....')
        # Re-Initialize GraspNP at each loop.
        model = CustomGraspNeuralProcess(
            d_latents=args.d_latents,
            n_decoders=args.n_decoders
        )

        # Prepare current data.
        train_dataloader, val_dataloader, pool_dataloader = \
            get_dataloaders(args, train_dict, val_dict, pool_dict)

        # Train GNP with current data.
        model = train(train_dataloader, val_dataloader, model, n_epochs=args.n_epochs)
        print('Done training.')

        logger.save_dataset(dataset=train_dict, tx=tx)
        logger.save_val_dataset(val_dataset=val_dict, tx=tx)
        logger.save_neural_process(gnp=model, tx=tx, symlink_tx0=False)

        # Collect new grasps.
        print('Collecting datapoints...')
        acquire_indices = choose_acquisition_data(
            gnp=model,
            pool_dataloader=pool_dataloader,
            n_acquire=args.n_acquire_per_object,
            strategy=args.strategy,
        )
        new_grasps = remove_grasps(pool_dict, acquire_indices)
        add_grasps(train_dict, new_grasps)

        print('--- Train Stats ---')
        print(len(train_dict['grasp_data']['grasp_geometries'][0]))
        print(train_dict['grasp_data']['grasp_geometries'][0][0].shape)

        print('--- Pool Stats ---')
        print(len(pool_dict['grasp_data']['grasp_geometries'][0]))
        print(pool_dict['grasp_data']['grasp_geometries'][0][0].shape)

        print('Done data collection.')
        logger.save_acquisition_data(new_grasps, None, tx)#new_data, all_samples, tx)

        print('Time: ' + str((time.time()-start_time)*(1/60)) + ' minutes')


def run_active_train(args):
    """ Entry function for active learning. Prepare initial datasets. """

    # Initial arg checks.
    args.use_latents = False
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # Get an initial datasets/dataloaders.
    train_dict, val_dict, pool_dict, object_set = get_initial_dataset(
        args.data_root,
        n_train_grasps_per_object=args.init_grasps_per_object,
        n_val_grasps_per_object=10
    )

    # Start training.
    print('Starting training from scratch.')
    active_train(
        train_dict=train_dict,
        val_dict=val_dict,
        pool_dict=pool_dict,
        logger=logger,
        args=args
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='',
        help='Where results will be saved. Randon number if not specified.')

    # Active learning parameters.
    parser.add_argument('--max-acquisitions', type=int, default=1000,
        help='Number of iterations to run the main active learning loop for.')
    parser.add_argument('--init-grasps-per-object', type=int, default=10)
    parser.add_argument('--data-root', type=str, default='')
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
