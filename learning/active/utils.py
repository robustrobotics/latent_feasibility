import re
import numpy as np
import os
import pickle
import time
import torch
import datetime

from torch.utils.data import DataLoader

from learning.models.ensemble import Ensemble
from learning.models.grasp_np.grasp_neural_process import CustomGraspNeuralProcess
from learning.models.mlp_dropout import MLP
from learning.models.latent_ensemble import LatentEnsemble, GraspingLatentEnsemble
from learning.models.pointnet import PointNetClassifier


class ActiveExperimentLogger:

    def __init__(self, exp_path, use_latents=False):
        self.exp_path = exp_path
        self.acquisition_step = 0
        self.tower_counter = 0
        self.use_latents = use_latents

        with open(os.path.join(self.exp_path, 'args.pkl'), 'rb') as handle:
            self.args = pickle.load(handle)
        if 'max_acquisitions' not in self.args:
            self.args.max_acquisitions = 1
        self.update_max_acquisitions()

    def update_max_acquisitions(self):
        """ Check that self.args.max_acquisitions corresponds to the number of
        acquisition steps actually performed.
        """
        aq_files = os.listdir(os.path.join(self.exp_path, 'models'))
        if len(aq_files) == 0:
            return
        max_aq = np.max([int(s.split('_')[-1].split('.')[0]) for s in aq_files if '_' in s]) + 1
        if max_aq < self.args.max_acquisitions:
            print(f'[WARNING] Only {max_aq} acquisition steps have been executed so far.')
            self.args.max_acquisitions = max_aq

    def get_acquisition_params(self):
        """ Return parameters useful for knowing how much data was collected.
        :return: init_dataset_size, n_acquire
        """
        if self.load_particles is not None:
            return 0, 1
        init_dataset = self.load_dataset(0)
        return len(init_dataset) // 4, self.args.n_acquire

    @staticmethod
    def get_experiments_logger(exp_path, args):
        logger = ActiveExperimentLogger(exp_path, use_latents=args.use_latents)
        dataset_path = os.path.join(exp_path, 'datasets')
        dataset_files = os.listdir(dataset_path)
        if len(dataset_files) == 0:
            raise Exception('No datasets found on args.exp_path. Cannot restart active training.')
        txs = []
        for file in dataset_files:
            matches = re.match(r'active_(.*).pkl', file)
            if matches:  # sometimes system files are saved here, don't parse these
                txs += [int(matches.group(1))]
        logger.acquisition_step = max(txs)

        # save potentially new args
        with open(os.path.join(exp_path, 'args_restart.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return logger

    @staticmethod
    def setup_experiment_directory(args):
        """
        Setup the directory structure to store models, figures, datasets
        and parameters relating to an experiment.
        """
        root = 'learning/experiments/logs'
        if not os.path.exists(root):
            os.makedirs(root)

        exp_name = args.exp_name if len(args.exp_name) > 0 else 'exp'
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        exp_dir = f'{exp_name}-{timestamp}'
        exp_path = os.path.join(root, exp_dir)

        os.mkdir(exp_path)
        os.mkdir(os.path.join(exp_path, 'figures'))
        os.mkdir(os.path.join(exp_path, 'towers'))
        os.mkdir(os.path.join(exp_path, 'models'))
        os.mkdir(os.path.join(exp_path, 'datasets'))
        os.mkdir(os.path.join(exp_path, 'val_datasets'))
        os.mkdir(os.path.join(exp_path, 'acquisition_data'))

        with open(os.path.join(exp_path, 'args.pkl'), 'wb') as handle:
            pickle.dump(args, handle)

        return ActiveExperimentLogger(exp_path, use_latents=args.use_latents)

    def save_dataset(self, dataset, tx):
        fname = f'active_{tx}.pkl'
        with open(os.path.join(self.exp_path, 'datasets', fname), 'wb') as handle:
            pickle.dump(dataset, handle)

    def save_particles(self, particles, tx):
        fname = f'particles_{tx}.pkl'
        with open(os.path.join(self.exp_path, 'models', fname), 'wb') as handle:
            pickle.dump(particles, handle)

    def load_particles(self, tx):
        fname = f'particles_{tx}.pkl'
        path = os.path.join(self.exp_path, 'models', fname)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as handle:
            particles = pickle.load(handle)
        return particles

    def save_val_dataset(self, val_dataset, tx):
        fname = f'val_active_{tx}.pkl'
        with open(os.path.join(self.exp_path, 'val_datasets', fname), 'wb') as handle:
            pickle.dump(val_dataset, handle)

    def load_dataset(self, tx):
        fname = f'active_{tx}.pkl'
        path = os.path.join(self.exp_path, 'datasets', fname)
        try:
            with open(path, 'rb') as handle:
                dataset = pickle.load(handle)
            return dataset
        except:
            print(f'active_{tx}.pkl not found on path')
            return None

    def load_val_dataset(self, tx):
        fname = f'val_active_{tx}.pkl'
        path = os.path.join(self.exp_path, 'val_datasets', fname)
        try:
            with open(path, 'rb') as handle:
                val_dataset = pickle.load(handle)
            return val_dataset
        except:
            print(f'val_active_{tx}.pkl not found on path')
            return None

    def get_figure_path(self, fname):
        if not os.path.exists(os.path.join(self.exp_path, 'figures')):
            os.mkdir(os.path.join(self.exp_path, 'figures'))
        return os.path.join(self.exp_path, 'figures', fname)

    def get_towers_path(self, fname):
        return os.path.join(self.exp_path, 'towers', fname)

    def get_ensemble(self, tx):
        """ Load an ensemble from the logging structure.
        :param tx: The active learning iteration of which ensemble to load.
        :return: learning.models.Ensemble object.
        """
        # Load metadata and initialize ensemble.
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'rb') as handle:
            metadata = pickle.load(handle)
        ensemble = Ensemble(base_model=metadata['base_model'],
                            base_args=metadata['base_args'],
                            n_models=metadata['n_models'])

        if self.use_latents and metadata['base_model'] == PointNetClassifier:
            ensemble = GraspingLatentEnsemble(ensemble,
                                              n_latents=metadata['n_latents'],
                                              d_latents=metadata['d_latents'])
        elif self.use_latents:
            ensemble = LatentEnsemble(ensemble,
                                      n_latents=metadata['n_latents'],
                                      d_latents=metadata['d_latents'])

        # Load ensemble weights.
        path = os.path.join(self.exp_path, 'models', f'ensemble_{tx}.pt')
        try:
            ensemble.load_state_dict(torch.load(path, map_location='cpu'))
            return ensemble
        except:
            print(f'ensemble_{tx}.pt not found on path: {path}')
            return None

    def save_ensemble(self, ensemble, tx, symlink_tx0=False):
        """ Save an ensemble within the logging directory. The weights will
        be saved to <exp_name>/models/ensemble_<tx>.pt. Model metadata that
        is needed to initialize the Ensemble class while loading is saved
        to <exp_name>/models/metadata.pkl.

        :ensemble: A learning.model.Ensemble object.
        :tx: The active learning timestep these models represent.
        """
        if self.use_latents:
            self.save_latent_ensemble(ensemble, tx, symlink_tx0)
            return

        # Save ensemble metadata.
        metadata = {'base_model': ensemble.base_model,
                    'base_args': ensemble.base_args,
                    'n_models': ensemble.n_models}
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(metadata, handle)

        # Save ensemble weights.
        path = os.path.join(self.exp_path, 'models', f'ensemble_{tx}.pt')
        torch.save(ensemble.state_dict(), os.path.join(path))

    def get_neural_process(self, tx):
        # load metadata and initialize np
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'rb') as handle:
            metadata = pickle.load(handle)

        gnp = CustomGraspNeuralProcess(
            d_latents=metadata['d_latents'],
            input_features=metadata['input_features'],
            d_grasp_mesh_enc=metadata['d_grasp_mesh_enc'],
            d_object_mesh_enc=metadata['d_object_mesh_enc'],
            n_decoders=metadata['n_decoders']
        )

        # load in the encoder, mesh_encoder subroutine, and decoder weights
        # and insert them into the main np
        try:
            path = os.path.join(self.exp_path, 'models', f'grasp_encoder_{tx}.pt')
            gnp.grasp_geom_encoder.load_state_dict(torch.load(path, map_location='cpu'))

            path = os.path.join(self.exp_path, 'models', f'mesh_encoder_{tx}.pt')
            gnp.mesh_encoder.load_state_dict(torch.load(path, map_location='cpu'))

            path = os.path.join(self.exp_path, 'models', f'np_encoder_{tx}.pt')
            gnp.encoder.load_state_dict(torch.load(path, map_location='cpu'))

            path = os.path.join(self.exp_path, 'models', f'np_decoders_{tx}.pt')
            gnp.decoders.load_state_dict(torch.load(path, map_location='cpu'))
            return gnp

        except FileNotFoundError:
            print('model not found on path: ' + path)
            return None

    def save_neural_process(self, gnp, tx, symlink_tx0):
        # save neural process data (for now, it's just number of latent dimensions)
        metadata = {
            'd_latents': gnp.d_latents,
            'input_features': gnp.input_features,
            'd_grasp_mesh_enc': gnp.d_grasp_mesh_enc,
            'd_object_mesh_enc': gnp.d_object_mesh_enc,
            'n_decoders': gnp.n_decoders
        }
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(metadata, handle)

        # save the np encoder (and mesh encoder subroutine) and the decoder separately
        grasp_enc_path = os.path.join(self.exp_path, 'models', f'grasp_encoder_{tx}.pt')
        mesh_enc_path = os.path.join(self.exp_path, 'models', f'mesh_encoder_{tx}.pt')
        enc_path = os.path.join(self.exp_path, 'models', f'np_encoder_{tx}.pt')
        dec_path = os.path.join(self.exp_path, 'models', f'np_decoders_{tx}.pt')
        
        if tx > 0 and symlink_tx0:
            grasp_enc_src = 'grasp_encoder_0.pt'
            mesh_enc_src = 'mesh_encoder_0.pt'
            enc_src, dec_src = 'np_encoder_0.pt', 'np_decoders_0.pt'
            # overwrite if symlink exists (which is safe, since we are not changing any files)
            try:
                os.symlink(grasp_enc_src, grasp_enc_path)
                os.symlink(mesh_enc_src, mesh_enc_path)
                os.symlink(enc_src, enc_path)
                os.symlink(dec_src, dec_path)
            except FileExistsError:
                print('[INFO]: overwriting gnp symlinks: %s, %s, %s, %s.' %
                      (grasp_enc_path, mesh_enc_path, enc_path, dec_path))
                os.remove(grasp_enc_path)
                os.remove(mesh_enc_path)
                os.remove(enc_path)
                os.remove(dec_path)
                os.symlink(grasp_enc_src, grasp_enc_path)
                os.symlink(mesh_enc_src, mesh_enc_path)
                os.symlink(enc_src, enc_path)
                os.symlink(dec_src, dec_path)
        else:
            torch.save(gnp.grasp_geom_encoder.state_dict(), os.path.join(grasp_enc_path))
            torch.save(gnp.mesh_encoder.state_dict(), os.path.join(mesh_enc_path))
            torch.save(gnp.encoder.state_dict(), os.path.join(enc_path))
            torch.save(gnp.decoders.state_dict(), os.path.join(dec_path))

    def save_latent_ensemble(self, latent_ensemble, tx, symlink_tx0):
        metadata = {'base_model': latent_ensemble.ensemble.base_model,
                    'base_args': latent_ensemble.ensemble.base_args,
                    'n_models': latent_ensemble.ensemble.n_models,
                    'n_latents': latent_ensemble.n_latents,
                    'd_latents': latent_ensemble.d_latents}
        path = os.path.join(self.exp_path, 'models', 'metadata.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(metadata, handle)

        # Save ensemble weights.
        path = os.path.join(self.exp_path, 'models', f'ensemble_{tx}.pt')
        if tx > 0 and symlink_tx0:
            src = 'ensemble_0.pt'
            os.symlink(src, path)
        else:
            torch.save(latent_ensemble.state_dict(), os.path.join(path))

    def get_towers_data(self, tx):
        # Get all tower files at the current acquisition step, in sorted order
        tower_files = []
        all_towers = os.listdir(os.path.join(self.exp_path, 'towers'))
        for tower_file in all_towers:
            match_str = r'labeled_tower_(.*)_(.*)_(.*)_{}.pkl'.format(tx)
            if re.match(match_str, tower_file):
                tower_files.append(tower_file)
        tower_files.sort()
        # Extract the tower data from each tower file
        tower_data = []
        for tower_file in tower_files:
            with open(self.get_towers_path(tower_file), 'rb') as handle:
                tower_tx_data = pickle.load(handle)
            tower_data.append(tower_tx_data)
        return tower_data

    def save_towers_data(self, block_tower, block_ids, label):
        fname = 'labeled_tower_{:%Y-%m-%d_%H-%M-%S}_'.format(datetime.datetime.now()) \
                + str(self.tower_counter) + '_' + str(self.acquisition_step) + '.pkl'
        path = self.get_towers_path(fname)
        with open(path, 'wb') as handle:
            pickle.dump([block_tower, block_ids, label], handle)
        self.tower_counter += 1

    def save_acquisition_data(self, new_data, samples, tx):
        data = {
            'acquired_data': new_data,
            'samples': samples
        }
        path = os.path.join(self.exp_path, 'acquisition_data', f'acquired_{tx}.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)
        # self.acquisition_step = tx+1
        # self.tower_counter = 0
        # self.remove_unlabeled_acquisition_data()

    def remove_unlabeled_acquisition_data(self):
        os.remove(os.path.join(self.exp_path, 'acquired_processing.pkl'))

    def save_unlabeled_acquisition_data(self, data):
        path = os.path.join(self.exp_path, 'acquired_processing.pkl')
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)

    def get_unlabeled_acquisition_data(self):
        path = os.path.join(self.exp_path, 'acquired_processing.pkl')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def get_block_placement_data(self):
        path = os.path.join(self.exp_path, 'block_placement_data.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                block_placements = pickle.load(handle)
            return block_placements
        else:
            return {}

    def save_block_placement_data(self, block_placements):
        block_placement_data = self.get_block_placement_data()
        block_placement_data[self.acquisition_step] = block_placements
        with open(os.path.join(self.exp_path, 'block_placement_data.pkl'), 'wb') as handle:
            pickle.dump(block_placement_data, handle)

    def load_acquisition_data(self, tx):
        path = os.path.join(self.exp_path, 'acquisition_data', f'acquired_{tx}.pkl')
        try:
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
            return data['acquired_data'], data['samples']
        except:
            print(f'acquired_{tx}.pkl not found on path')
            return None, None

    def save_evaluation_tower(self, tower, reward, max_reward, tx, planning_model, task, noise=None):
        if noise:
            tower_file = f'towers_{tx}_{noise}.pkl'
        else:
            tower_file = f'towers_{tx}.pkl'
        tower_height = len(tower)
        tower_key = f'{tower_height}block'
        tower_path = os.path.join(self.exp_path, 'evaluation_towers', task, planning_model)
        if not os.path.exists(tower_path):
            os.makedirs(tower_path)
        if not os.path.isfile(os.path.join(tower_path, tower_file)):
            towers = {}
            print('Saving evaluation tower to %s' % os.path.join(tower_path, tower_file))
        else:
            with open(os.path.join(tower_path, tower_file), 'rb') as f:
                towers = pickle.load(f)
        if tower_key in towers:
            towers[tower_key].append((tower, reward, max_reward))
        else:
            towers[tower_key] = [(tower, reward, max_reward)]
        with open(os.path.join(tower_path, tower_file), 'wb') as f:
            pickle.dump(towers, f)
        print('APPENDING evaluation tower to %s' % os.path.join(tower_path, tower_file))

    def get_evaluation_towers(self, task, planning_model, tx):
        tower_path = os.path.join(self.exp_path, 'evaluation_towers', task, planning_model)
        towers_data = {}
        for file in os.listdir(tower_path):
            if file != '.DS_Store':
                with open(os.path.join(tower_path, file), 'rb') as f:
                    data = pickle.load(f)
                towers_data[os.path.join(tower_path, file)] = data
        return towers_data
