import numpy as np
import pickle
import time
import torch
from random import choices as sample_with_replacement

from copy import deepcopy
from torch.utils.data import DataLoader

from block_utils import World, Environment, Object, Quaternion, Pose, Position, get_rotated_block, ZERO_POS, QUATERNIONS
from learning.domains.towers.generate_tower_training_data import sample_random_tower, vectorize
from learning.domains.towers.tower_data import TowerDataset, TowerSampler, unprocess
from tower_planner import TowerPlanner

from agents.panda_agent import PandaClientAgent


def sample_sequential_data(block_set, dataset, n_samples, prerotate=False):
    """ Generate n_samples random towers. Each tower has the property that its
    base (the tower until the last block) is stable. To ensure this, we start
    with all the stable towers plus base cases of single block towers.
    :param block_set: List of blocks that can be used in the towers.
    :param dataset: List of current towers that have been built.
    :param n_samples: Number of random towers to consider.
    :param max_blocks
    :return: Dict containining numpy arrays of the towers sorted by size.
    """
    print('Generating data sequentially...')
    keys = ['2block', '3block', '4block', '5block']

    # initialize a dictionary of lists to store the generated data
    sampled_towers = {k: {} for k in keys}
    for k in keys:
        sampled_towers[k]['towers'] = []
        sampled_towers[k]['labels'] = []
        if block_set is not None:
            sampled_towers[k]['block_ids'] = []

    # Gather list of all stable towers. Towers should be of blocks that are rotated in "Block" format.
    stable_towers = []
    # Get all single block stable towers.
    for block in block_set:
        for orn in QUATERNIONS:
            new_block = deepcopy(block)
            new_block.pose = Pose(ZERO_POS, orn)
            rot_block = get_rotated_block(new_block)
            rot_block.pose = Pose((0., 0., rot_block.dimensions.z/2.), (0, 0, 0, 1))
            new_block.pose = Pose((0., 0., rot_block.dimensions.z/2.), new_block.pose.orn)
            if prerotate:
                stable_towers.append([rot_block])
            else:
                stable_towers.append([new_block])

    # Get all stable towers from the dataset.
    for k in keys[:3]:
        if dataset is None:
            break
        tower_tensors = unprocess(dataset.tower_tensors[k].cpu().numpy().copy())
        tower_labels = dataset.tower_labels[k]
        for ix, (tower_vec, tower_label) in enumerate(zip(tower_tensors, tower_labels)):
            if tower_label == 1:
                block_tower = []
                for bx in range(tower_vec.shape[0]):
                    block = Object.from_vector(tower_vec[bx, :])
                    if block_set is not None:
                        block.name = 'obj_'+str(int(dataset.tower_block_ids[k][ix, bx]))
                    block_tower.append(block)
                stable_towers.append(block_tower)
            
    # maintain block info my block name 
    # TODO: this will NOT work if using random blocks
    block_lookup = {}    
    for block in block_set:
        block_lookup[block.name] = block

    # Sample random towers by randomly choosing a stable base then trying to add a block.
    for ix in range(n_samples):
        # Choose a stable base.
        tower_ix = np.random.choice(np.arange(0, len(stable_towers)))
        base_tower = stable_towers[tower_ix]
        base_rotated = [get_rotated_block(b) for b in base_tower]
        # Choose a block that's not already in the tower.
        remaining_blocks = {}
        for k in block_lookup:
            used = False
            for block in base_tower:
                if k == block.name:
                    used = True
            if not used:
                remaining_blocks[k] = block_lookup[k]
                
        # if we switch block sets during training then the remaining_blocks list
        # will be longer than block_set - len(base_tower)
        #assert(len(remaining_blocks) == len(block_set) - len(base_tower))

        new_block = deepcopy(np.random.choice(list(remaining_blocks.values())))
        
        # Choose an orientation.
        orn = QUATERNIONS[np.random.choice(np.arange(0, len(QUATERNIONS)))]
        new_block.pose = Pose(ZERO_POS, orn)
        rot_block = get_rotated_block(new_block)
        
        # Sample a displacement.
        base_dims = np.array(base_rotated[-1].dimensions)[:2]
        new_dims = np.array(rot_block.dimensions)[:2]
        max_displacements_xy = (base_dims+new_dims)/2.
        noise_xy = np.clip(0.5*np.random.randn(2), -0.95, 0.95)
        rel_xy = max_displacements_xy*noise_xy

        # Calculate the new pose.
        base_pos = np.array(base_rotated[-1].pose.pos)[:2]
        pos_xy = base_pos + rel_xy
        pos_z = np.sum([b.dimensions.z for b in base_rotated]) + rot_block.dimensions.z/2.
        rot_block.pose = Pose((pos_xy[0], pos_xy[1], pos_z), (0, 0, 0, 1))
        new_block.pose = Pose((pos_xy[0], pos_xy[1], pos_z), new_block.pose.orn)
        
        # Add block to tower.
        if prerotate:
            new_tower = base_rotated + [rot_block]
        else:
            new_tower = base_tower + [new_block]

        if False:
            w1 = World(base_rotated + [rot_block])
            w2 = World(base_tower + [new_block])
            env = Environment([w1, w2], vis_sim=True, vis_frames=True)
            for tx in range(240):
                env.step(vis_frames=True)
                time.sleep(1/240.)
            env.disconnect()
        # Save that tower in the sampled_towers dict
        n_blocks = len(new_tower)
        sampled_towers['%dblock' % n_blocks]['towers'].append(vectorize(new_tower))
    
        # save block id
        if block_set is not None:
            block_ids = [block.get_id() for block in new_tower]
            sampled_towers['%dblock' % n_blocks]['block_ids'].append(block_ids)
    
    # convert all the sampled towers to numpy arrays
    for k in keys:
        sampled_towers[k]['towers'] = np.array(sampled_towers[k]['towers'])
        if sampled_towers[k]['towers'].shape[0] == 0:
            sampled_towers[k]['towers'] = sampled_towers[k]['towers'].reshape((0, int(k[0]), 21))
        sampled_towers[k]['labels'] = np.zeros((sampled_towers[k]['towers'].shape[0],))
        if block_set is not None:
            sampled_towers[k]['block_ids'] = np.array(sampled_towers[k]['block_ids'])
    return sampled_towers

def sample_unlabeled_data(n_samples, block_set=None, prerotate=False, range_n_blocks=(2, 5), include_index=-1, ignore_rot=False):
    """ Generate n_samples random towers. For now each sample can also have
    random blocks. We should change this later so that the blocks are fixed 
    (i.e., chosen elsewhere) and we only sample the configuration.
    :param n_samples: Number of random towers to consider.
    :param block_set (optional): blocks to use in towers. generate new blocks if None
    :return: Dict containining numpy arrays of the towers sorted by size.
    """
    # initialize a dictionary of lists to store the generated data
    #print('Generating data randomly...')
    sampled_towers = {}
    for i in range(range_n_blocks[0], range_n_blocks[1]+1):
        k = f'{i}block'
        sampled_towers[k] = {}
        sampled_towers[k]['towers'] = []
        sampled_towers[k]['labels'] = []
        if block_set is not None:
            sampled_towers[k]['block_ids'] = []


    min_n_blocks = range_n_blocks[0]
    max_n_blocks = min(range_n_blocks[1], len(block_set))

    # sample random towers and add them to the lists in the dictionary
    for ix in range(n_samples):
        n_blocks = np.random.randint(min_n_blocks, max_n_blocks+1)
        # get n_blocks, either from scratch or from the block set
        if block_set is not None and include_index < 0: 
            blocks = np.random.choice(block_set, n_blocks, replace=False)
        elif block_set is not None:
            extra_blocks = block_set[:include_index] + block_set[include_index+1:]
            blocks = [block_set[include_index]] + list(np.random.choice(extra_blocks, n_blocks-1, replace=False))
        else:
            blocks = [Object.random(f'obj_{ix}') for ix in range(n_blocks)]
        # sample a new tower
        tower = sample_random_tower(blocks, ignore_rot=ignore_rot)
        rotated_tower = [get_rotated_block(b) for b in tower]
        # and save that tower in the sampled_towers dict
        if prerotate:
            sampled_towers['%dblock' % n_blocks]['towers'].append(vectorize(rotated_tower))
        else:
            sampled_towers['%dblock' % n_blocks]['towers'].append(vectorize(tower))
        if block_set is not None:
            block_ids = [block.get_id() for block in rotated_tower]
            sampled_towers['%dblock' % n_blocks]['block_ids'].append(block_ids)
    
    # convert all the sampled towers to numpy arrays
    for k in sampled_towers.keys():
        sampled_towers[k]['towers'] = np.array(sampled_towers[k]['towers'])
        sampled_towers[k]['labels'] = np.zeros((sampled_towers[k]['towers'].shape[0],))
        if sampled_towers[k]['towers'].shape[0] == 0:
            sampled_towers[k]['towers'] = sampled_towers[k]['towers'].reshape((0, int(k[0]), 21))
        if block_set is not None:
            sampled_towers[k]['block_ids'] = np.array(sampled_towers[k]['block_ids'])

    return sampled_towers

def sample_next_block(n_samples, bases={}, block_set=None):
    # if the dataset is empty, we are sampling the 2-block bases of the towers
    if bases == {}:
        return sample_unlabeled_data(n_samples, block_set=block_set, range_n_blocks=(2,2))
    # if the dataset is non-empty, then for each tower in the dataset we need to sample
    # a bunch of options for the next block to be placed on top
    else:
        assert len(list(bases.keys())) == 1, 'I want all the towers to be the same height cuz i\'m rushing'
        base_n_blocks_key = list(bases.keys())[0]
        base_n_blocks = int(base_n_blocks_key.strip('block'))
        n_towers = bases[base_n_blocks_key]['towers'].shape[0]
        n_new_blocks_per_base = np.ceil(n_samples/n_towers).astype(int)
        new_towers = []
        new_block_ids = []
        for i in range(n_towers):
            # pull out some information about the tower we're working on
            current_tower = bases[base_n_blocks_key]['towers'][i]
            top_block = current_tower[-1]
            top_block_dims = top_block[4:7]
            top_block_pos = top_block[7:10]
            top_of_tower_height = top_block_pos[2] + top_block_dims[2]/2

            # get n_new_blocks_per_base new blocks to add on top of the tower
            if block_set is None:
                new_top_blocks = [Object.random(f'obj_{ix}') for ix in range(n_new_blocks_per_base)]
            else:
                # we can't duplicate blocks, so only choose new top blocks from the 
                # blocks that aren't already in the base
                block_ids_already_in_tower = list(bases[base_n_blocks_key]['block_ids'][i])
                remaining_blocks = [b for b in block_set if b.name.strip('obj_') not in block_ids_already_in_tower]
                new_top_blocks = sample_with_replacement(remaining_blocks, k=n_new_blocks_per_base)
                # save the block ids for each of the new towers
                new_top_block_ids = [b.name.strip('obj_') for b in new_top_blocks]
                new_block_ids_local = np.zeros([n_new_blocks_per_base, base_n_blocks+1])
                new_block_ids_local[:,:-1] = block_ids_already_in_tower
                new_block_ids_local[:,-1] = new_top_block_ids
                new_block_ids.append(new_block_ids_local)

            # get random rotations for each block
            orns = sample_with_replacement(QUATERNIONS, k=n_new_blocks_per_base)

            # apply the rotations to each block
            rotated_blocks = []
            for orn, block in zip(orns, new_top_blocks):
                block.pose = Pose(ZERO_POS, orn)
                rotated_blocks.append(get_rotated_block(block))

            # figure out how far each block can be moved w/ losing contact w/ the block below
            dims_xy = np.array([rb.dimensions for rb in rotated_blocks])[:,:2]
            # figure out how far each block can be moved w/ losing contact w/ the block below
            max_displacements_xy = (top_block_dims[:2] + dims_xy)/2.
            # sample unscaled noise (clip bceause random normal can exceed -1, 1)
            noise_xy = np.clip(0.5*np.random.randn(n_new_blocks_per_base, 2), -0.95, 0.95)
            # and scale the noise by the max allowed displacement
            rel_xy = max_displacements_xy * noise_xy
            # and get the actual pos by the difference to the top block pos
            pos_xy = top_block_pos[:2] + rel_xy

            # calculate the height of each block
            pos_z = np.array([rb.dimensions.z/2 + top_of_tower_height for rb in rotated_blocks])
            pos_xyz = pos_xyz = np.hstack([pos_xy, pos_z[:,None]])

            for pos, orn, block in zip(pos_xyz, orns, new_top_blocks):
                block.pose = Pose(Position(*pos), orn)
                block.rotation = orn

            # create an array to hold all the new towers
            new_towers_local = np.zeros([n_new_blocks_per_base, current_tower.shape[0] + 1, current_tower.shape[1]])
            # add the existing base of the tower to the array
            new_towers_local[:,:-1] = current_tower
            # and add the new top block to each tower
            new_towers_local[:,-1] = vectorize(new_top_blocks)
            new_towers.append(new_towers_local)

        # package the new towers into a dict of the appropriate format. include
        # block_ids if we are using a block set
        new_towers = np.concatenate(new_towers)
        new_samples =   {'towers': new_towers,
            'labels': np.zeros(new_towers.shape[0])}
        if block_set is not None:
            new_samples['block_ids'] = np.concatenate(new_block_ids)

        return {f'{base_n_blocks+1}block': new_samples}

def get_sequential_predictions(dataset, ensemble, use_latents=False):
    """
    Make a separate prediction for each of the sub-towers in a tower.
    Return stable only if all sub-towers are stable. This is for the
    model that assumes the base of each tower is stable.
    """
    preds = []
    # Create TowerDataset object.
    tower_dataset = TowerDataset(dataset, augment=False)
    tower_sampler = TowerSampler(dataset=tower_dataset,
                                 batch_size=64,
                                 shuffle=False)
    tower_loader = DataLoader(dataset=tower_dataset,
                              batch_sampler=tower_sampler)

    # Iterate through dataset, getting predictions for each.
    for tensor, block_ids, _ in tower_loader:
        sub_tower_preds = []
        block_ids = block_ids.long()  # indices must be long, byte, or bool tensors
        for n_blocks in range(2, tensor.shape[1]+1):
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            with torch.no_grad():
                if use_latents:
                    sub_tower_preds.append(ensemble.forward(
                        tensor[:, :n_blocks, 4:], block_ids[:, :n_blocks], N_samples=10, collapse_ensemble=True, collapse_latents=True).squeeze(dim=-1))
                else:
                    sub_tower_preds.append(ensemble.forward(tensor[:, :n_blocks, :]).mean(-1, keepdim=True))
        sub_tower_preds = torch.stack(sub_tower_preds, dim=0)
        #print('SubTowerPreds:', sub_tower_preds.shape)
        #preds.append(sub_tower_preds[-1,:,:])

        # multiply then threshold
        preds.append((sub_tower_preds.prod(dim=0) > 0.5).float())
        # threshold then "multiply" (what we used for RSS)
        # preds.append((  sub_tower_preds > 0.5).all(dim=0).float())

        #print(preds[-1].shape)
    return torch.cat(preds, dim=0)

def get_predictions(dataset, ensemble, use_latents=False, N_samples=10, collapse_latents=False, collapse_ensemble=False, keep_latent_ix=-1):
    """
    :param dataset: A tower_dict structure.
    :param ensemble: The Ensemble model which to use for predictions.
    :return: Return (N, K) array of flat predictions. Predictions are 
    ordered by tower size.
    """
    preds = []
    # Create TowerDataset object.
    tower_dataset = TowerDataset(dataset, augment=False)
    tower_sampler = TowerSampler(dataset=tower_dataset,
                                 batch_size=64,
                                 shuffle=False)
    tower_loader = DataLoader(dataset=tower_dataset,
                              batch_sampler=tower_sampler)

    # Iterate through dataset, getting predictions for each.
    for tensor, block_ids, labels in tower_loader:
        N_batch = tensor.shape[0]
        block_ids = block_ids.long() # indices must be long, byte, or bool tensors
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            if use_latents:
                # take samples from the ensemble and the latents separately, to approximate computing MI
                # I(y ; theta) and I(y ; z) separately
                # latent_samples = ensemble.forward(
                #     tensor[...,4:], block_ids, N_samples=N_samples, collapse_ensemble=True, collapse_latents=False).reshape(N_batch, -1)
                # ensemble_samples = ensemble.forward(
                #     tensor[...,4:], block_ids, N_samples=N_samples, collapse_ensemble=False, collapse_latents=True).reshape(N_batch, -1)
                # pred_samples = torch.cat([latent_samples, ensemble_samples], axis=1)
                # preds.append(pred_samples)

                # takes samples from the joint distribution from the joint distribution to compute I(y ; theta, z)
                preds.append(ensemble.forward(
                    tensor[...,4:], block_ids, N_samples=N_samples, collapse_ensemble=collapse_ensemble, collapse_latents=collapse_latents, keep_latent_ix=keep_latent_ix).reshape(N_batch, -1))
            else:
                preds.append(ensemble.forward(tensor))
    return torch.cat(preds, dim=0)

def get_predictions_single_ensemble(dataset, ensemble):
    """
    :param dataset: A tower_dict structure.
    :param ensemble: The Ensemble model which to use for predictions.
    :return: Return (N, K) array of flat predictions. Predictions are 
    ordered by tower size.
    """
    preds = []
    # Create TowerDataset object.
    tower_dataset = TowerDataset(dataset, augment=False)
    tower_sampler = TowerSampler(dataset=tower_dataset,
                                 batch_size=64,
                                 shuffle=False)
    tower_loader = DataLoader(dataset=tower_dataset,
                              batch_sampler=tower_sampler)

    # Iterate through dataset, getting predictions for each.
    for tensor, _, _ in tower_loader:
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            preds.append(ensemble.forward(tensor))
    return torch.cat(preds, dim=0)


def get_labels(dataset, exec_mode, agent, logger, xy_noise, save_tower=False, label_subtowers=False):
    """ Takes as input a dictionary from the get_subset function. 
    Augment it with stability labels. 
    :param dataset:
    :param exec_mode: str in ['simple-model', 'noisy-model', 'sim', 'real']
    :param agent: PandaAgent or None (if exec_mode == 'simple-model' or 'noisy-model')
    :return:
    """
    subtower_dataset = {'%dblock' % k : {} for k in [2,3,4,5]}
    for k in subtower_dataset:
        subtower_dataset[k]['towers'] = []
        subtower_dataset[k]['block_ids'] = []
        subtower_dataset[k]['labels'] = []
        
    block_placements = 0
        
    tp = TowerPlanner(stability_mode='contains')
    for k in dataset.keys():
        n_towers, n_blocks, _ = dataset[k]['towers'].shape
        labels = np.ones((n_towers,))

        for tower_ix in range(0, n_towers):
            print(f'Collecting tower {tower_ix+1}/{n_towers} for {k} towers...')
            # convert tower to Block representation.
            block_tower = []
            for block_jx in range(n_blocks): 
                vec_block = deepcopy(dataset[k]['towers'][tower_ix, block_jx, :])
                # only add noise if we are using the noisy-model exec_mode
                if exec_mode == 'noisy-model':
                    vec_block[7:9] += np.random.randn(2)*xy_noise
                block = Object.from_vector(vec_block) # block is already rotated
                if 'block_ids' in dataset[k].keys():
                    block.name = 'obj_'+str(dataset[k]['block_ids'][tower_ix, block_jx])
                block_tower.append(block)
            #  Use tp to check for stability.
            if exec_mode == 'simple-model' or exec_mode == 'noisy-model':
                
                # iterate through each subtower until it falls (is not constructable)
                subtowers = [block_tower[:k_sub] for k_sub in list(range(2,len(block_tower)+1))]
                for k_sub, subtower in enumerate(subtowers, 2):
                    rot_subtower = [get_rotated_block(b) for b in subtower]
                    label = float(tp.tower_is_constructable(rot_subtower))
                    
                    # add to labeled dataset    
                    subtower_dataset['%dblock' % k_sub]['towers'].append(dataset[k]['towers'][tower_ix, :k_sub, :])
                    if 'block_ids' in subtower_dataset['%dblock' % k_sub]:
                        subtower_dataset['%dblock' % k_sub]['block_ids'].append(dataset[k]['block_ids'][tower_ix, :k_sub])
                    subtower_dataset['%dblock' % k_sub]['labels'].append(label)
                    
                    # save tower file
                    if save_tower:
                        if 'block_ids' in dataset[k].keys():
                            logger.save_towers_data(dataset[k]['towers'][tower_ix, :k_sub, :], 
                                                    dataset[k]['block_ids'][tower_ix, :k_sub],
                                                    label)
                        else:
                            logger.save_towers_data(dataset[k]['towers'][tower_ix, :k_sub, :], 
                                                    None,
                                                    label)
                    # stop when tower falls
                    if label == 0.0:
                        block_placements += k_sub
                        labels[tower_ix] = 0.0
                        break
            # NOTE(izzy): the way this is implemented currently, it looks like the 'sim' and 'real'
            # agents do not support the subtower strategy
            else:
                vis = True
                success = False
                real = (exec_mode == 'real')
                # if planning fails, reset and try again
                while not success:
                    success, label = agent.simulate_tower(block_tower, vis, real=real)
                    print(f"Received success: {success}, label: {label}")
                    if not success:
                        if real:
                            input('Resolve conflict causing planning to fail, then press \
                                    enter to try again.')
                            if isinstance(agent, PandaClientAgent):
                                agent.restart_services()
                        else: # in sim
                            input('Should reset sim. Not yet handled. Exit and restart training.')
                labels[tower_ix] = label
                if 'block_ids' in dataset[k].keys():
                    logger.save_towers_data(dataset[k]['towers'][tower_ix, :, :], 
                                            dataset[k]['block_ids'][tower_ix, :],
                                            labels[tower_ix])
                else:
                    logger.save_towers_data(dataset[k]['towers'][tower_ix, :, :], 
                                            None,
                                            labels[tower_ix])
        dataset[k]['labels'] = labels
    
    if save_tower:
        # save block placement data
        logger.save_block_placement_data(block_placements)
        
    if label_subtowers:
        # vectorize labeled dataset and return
        for ki, k in enumerate(subtower_dataset, start=2):
            if subtower_dataset[k]['towers'] == []:
                subtower_dataset[k]['towers'] = np.zeros((0,ki,21))
                subtower_dataset[k]['block_ids'] = np.zeros((0,ki))
                subtower_dataset[k]['labels'] = np.zeros(0)
            subtower_dataset[k]['towers'] = np.array(subtower_dataset[k]['towers'])
            subtower_dataset[k]['block_ids'] = np.array(subtower_dataset[k]['block_ids'])
            subtower_dataset[k]['labels'] = np.array(subtower_dataset[k]['labels'])
        return subtower_dataset
    else:
        return dataset


def get_subset(samples, indices):
    """ Given a tower_dict structure and indices that are flat,
    return a tower_dict structure with only those indices.
    :param samples: A tower_dict structure.
    :param indices: Which indices of the original structure to select.
    """
    keys = samples.keys()
    selected_towers = {k: {'towers': [], 'block_ids': []} for k in keys}
    
    # Initialize tower ranges.
    start = 0
    for k in keys:
        end = start + samples[k]['towers'].shape[0]
        tower_ixs = indices[np.logical_and(indices >= start,
                                        indices < end)] - start
        selected_towers[k]['towers'] = samples[k]['towers'][tower_ixs,...]
        if 'block_ids' in selected_towers[k].keys():
            selected_towers[k]['block_ids'] = samples[k]['block_ids'][tower_ixs,...]
        start = end
    return selected_towers


class PoolSampler:

    def __init__(self, pool_fname):
        self.pool_fname = pool_fname
        self.keys = ['2block', '3block', '4block', '5block']
        with open(self.pool_fname, 'rb') as handle:
            self.pool = pickle.load(handle)

    def sample_unlabeled_data(self, n_samples):
        """
        Return all examples that haven't been chosen so far.
        """
        return self.pool

    def get_subset(self, samples, indices):
        """
        Remove chosen examples from the pool.
        """
        # NOTE(izzy): uhhh, where is samples used in here?

        selected_towers = {k: {'towers': []} for k in self.pool.keys()}

        start = 0
        for k in self.keys:
            end = start + self.pool[k]['towers'].shape[0]
            
            tower_ixs = indices[np.logical_and(indices >= start,
                                        indices < end)] - start
            selected_towers[k]['towers'] = self.pool[k]['towers'][tower_ixs,...]
 
            mask = np.ones(self.pool[k]['towers'].shape[0], dtype=bool)

            mask[tower_ixs] = False
            self.pool[k]['towers'] = self.pool[k]['towers'][mask,...]
            self.pool[k]['labels'] = self.pool[k]['labels'][mask,...]
            
            start = end
        
        return selected_towers
        


if __name__ == '__main__':
    data = sample_unlabeled_data(1000)
    for k in data.keys():
        print(data[k]['towers'].shape)

    indices = np.random.randint(0, 1000, 10)
    selected_towers = get_subset(data, indices)
    print(indices)
    for k in selected_towers.keys():
        print(selected_towers[k]['towers'].shape)

    labeled_towers = get_labels(selected_towers)
    for k in labeled_towers.keys():
        print(labeled_towers[k]['labels'])

    print('----- Test adding new data to dataset -----')
    with open('learning/data/random_blocks_(x40000)_5blocks_all.pkl', 'rb') as handle:
        towers_dict = pickle.load(handle)

    dataset = TowerDataset(towers_dict, augment=True, K_skip=10000)
    sampler = TowerSampler(dataset, 10, False)
    print('----- Initial batches -----')
    for batch_ixs in sampler:
        print(batch_ixs)

    loader = DataLoader(dataset=dataset,
                        batch_sampler=sampler)

    print('Num Initial Towers:', len(dataset))
    print('Initial indices per category:')
    print(dataset.get_indices())

    dataset.add_to_dataset(labeled_towers)
    print('Num Updated Towers:', len(dataset))
    print('Updated indices per category:')
    print(dataset.get_indices())

    print('----- Updated batches -----')
    for batch_ixs in sampler:
        print(batch_ixs)

    
    # print(len(loader))

    # print('----- Pool Sampler Test -----')
    # sampler = PoolSampler('learning/data/random_blocks_(x40000)_5blocks_uniform_mass.pkl')
    # pool = sampler.sample_unlabeled_data(10)
    # for k in sampler.keys:
    #     print(pool[k]['towers'].shape) 

    # sampler.get_subset(np.array([0, 1, 2, 3, 4, 20000, 20005]))
    # for k in sampler.keys:
    #     print(pool[k]['towers'].shape) 


    print('----- Sequential Sampler Test -----')
    with open('learning/data/block_set_10.pkl', 'rb') as handle:
        block_set = pickle.load(handle)

    with open('learning/experiments/logs/towers-con-init-random-blocks-10-fcgn-f1val-100k-20201213-152931/datasets/active_10.pkl', 'rb') as handle:
        dataset = pickle.load(handle)

    data_sampler_fn = lambda n_samples: sample_sequential_data(block_set, dataset, n_samples)
    unlabeled = data_sampler_fn(10000)
