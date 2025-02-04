import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


def convert_dict_to_float32(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[key] = np.array(value).astype('float32')
    return new_dict


class CustomGNPGraspDataset(Dataset):
    def __init__(self, data, context_data=None, add_mesh_normals=False, add_mesh_curvatures=False):
        """ Given data dictionaries, format them for the GNP model.

        If just data is given, generate a context for each grasp that includes all
        the other contexts for that object.

        If context_data is also given, use all of context_data as the contexts and
        only data as the targets. Note that object IDs should correspond in each
        dataset for proper functionality.
        """
        self.add_mesh_normals = add_mesh_normals
        self.add_mesh_curvatures = add_mesh_curvatures

        # Each of these is a list of length #objects.
        (self.cp_grasp_geometries,
         self.cp_grasp_points,
         self.cp_grasp_curvatures,
         self.cp_grasp_normals,
         self.cp_grasp_midpoints,
         self.cp_grasp_forces,
         self.cp_grasp_labels,
         self.cp_full_meshes,
         self.cp_object_properties) = self.process_raw_data(context_data)
        (self.hp_grasp_geometries,
         self.hp_grasp_points,
         self.hp_grasp_curvatures,
         self.hp_grasp_normals,
         self.hp_grasp_midpoints,
         self.hp_grasp_forces,
         self.hp_grasp_labels,
         self.hp_full_meshes,
         self.hp_object_properties) = self.process_raw_data(data)
        self.object_indices = sorted(self.hp_grasp_geometries.keys())

    def process_raw_data(self, data):
        if data is None:
            return None, None, None, None, None, None, None, None, None
        else:
            # TODO: nn-simplify - if curvature works, remove geoms
            grasp_geometries = {}
            for k, v in data['grasp_data']['grasp_geometries'].items():
                meshes = [arr[:256, :] for arr in v]
                for mx in range(len(meshes)):
                    while meshes[mx].shape[0] != 256:
                        n_dup = 256 - meshes[mx].shape[0]
                        meshes[mx] = np.concatenate([meshes[mx], meshes[mx][:n_dup, :]], axis=0)
                grasp_geometries[k] = np.array(meshes).astype('float32')

            grasp_points = convert_dict_to_float32(data['grasp_data']['grasp_points'])
            grasp_curvatures = convert_dict_to_float32(data['grasp_data']['grasp_curvatures'])
            grasp_normals = convert_dict_to_float32(data['grasp_data']['grasp_normal'])
            grasp_midpoints = convert_dict_to_float32(data['grasp_data']['grasp_midpoints'])
            grasp_forces = convert_dict_to_float32(data['grasp_data']['grasp_forces'])
            grasp_labels = convert_dict_to_float32(data['grasp_data']['labels'])
            full_meshes = convert_dict_to_float32(data['grasp_data']['object_meshes'])
            object_properties = {
                ox: v.astype('float32') 
                    for ox, v in enumerate(data['object_data']['object_properties'])
            }

            return (
                grasp_geometries,
                grasp_points,
                grasp_curvatures,
                grasp_normals,
                grasp_midpoints,
                grasp_forces,
                grasp_labels,
                full_meshes,
                object_properties
            )

    def _process_mesh_features(self, full_meshes, grasp_geometries, object_scale, local_grasp_scale):
        object_mesh = full_meshes[:, 0:3]/object_scale
        grasp_geoms = grasp_geometries[:, :, 0:3]/local_grasp_scale
        if self.add_mesh_normals:
            object_mesh = np.concatenate([object_mesh, full_meshes[:, 3:6]], axis=-1)
            grasp_geoms = np.concatenate([grasp_geoms, grasp_geometries[:, :, 3:6]], axis=-1)
        if self.add_mesh_curvatures:
            object_curvs = np.tanh(full_meshes[:, 6:]/0.1)
            grasp_curvs = np.tanh(grasp_geometries[:, :, 6:]/0.1)
            object_mesh = np.concatenate([object_mesh, object_curvs], axis=-1)
            grasp_geoms = np.concatenate([grasp_geoms, grasp_curvs], axis=-1)
        return object_mesh, grasp_geoms

    def _process_object_properties(self, object_properties):
        scaled_com = object_properties[0:3] / 0.15
        scaled_mass = (object_properties[3:5] - 0.2)/0.1
        return np.concatenate([scaled_com, scaled_mass])

    def __getitem__(self, ix):
        ox = self.object_indices[ix]
        object_scale, local_grasp_scale = 0.15, 1.0  # 0.1, 0.01
        if self.cp_grasp_geometries is None:
            cp_data = None
        else:
            cp_object_mesh, cp_grasp_geoms = self._process_mesh_features(
                full_meshes=self.cp_full_meshes[ox],
                grasp_geometries=self.cp_grasp_geometries[ox],
                local_grasp_scale=local_grasp_scale,
                object_scale=object_scale
            )
            cp_object_properties = self._process_object_properties(
                self.cp_object_properties[ox]
            )
            cp_data = {
                'object_mesh': cp_object_mesh,
                'object_properties': cp_object_properties,
                'grasp_geometries': cp_grasp_geoms,
                'grasp_points': self.cp_grasp_points[ox] / object_scale,
                'grasp_curvatures': np.tanh(self.cp_grasp_curvatures[ox] / 0.1),
                'grasp_normals': self.cp_grasp_normals[ox],
                'grasp_forces': (self.cp_grasp_forces[ox] - 12.5) / 7.5,
                'grasp_midpoints': self.cp_grasp_midpoints[ox] / object_scale,
                'grasp_labels': self.cp_grasp_labels[ox]
            }

        hp_object_mesh, hp_grasp_geoms = self._process_mesh_features(
            full_meshes=self.hp_full_meshes[ox],
            grasp_geometries=self.hp_grasp_geometries[ox],
            local_grasp_scale=local_grasp_scale,
            object_scale=object_scale
        )
        hp_object_properties = self._process_object_properties(
            self.hp_object_properties[ox]
        )
        hp_data = {
            'object_mesh': hp_object_mesh,
            'object_properties': hp_object_properties,
            'grasp_geometries': hp_grasp_geoms,
            'grasp_points': self.hp_grasp_points[ox] / object_scale,
            'grasp_curvatures': np.tanh(self.hp_grasp_curvatures[ox] / 0.1),
            'grasp_normals': self.hp_grasp_normals[ox],
            'grasp_forces': (self.hp_grasp_forces[ox] - 12.5) / 7.5,
            'grasp_midpoints': self.hp_grasp_midpoints[ox] / object_scale,
            'grasp_labels': self.hp_grasp_labels[ox]
        }
        return cp_data, hp_data

    def __len__(self):
        return len(self.hp_grasp_geometries)


def custom_collate_fn(items, rand_grasp_num=True, add_mesh_normals=False):
    """
    Decide how many context and target points to add.
    """
    if items[0][0] is not None:
        n_context = items[0][0]['grasp_geometries'].shape[0]
        n_target = items[0][1]['grasp_geometries'].shape[0]
    else:
        max_context = items[0][1]['grasp_geometries'].shape[0] + 1
        if rand_grasp_num:
            n_context = np.random.randint(low=40, high=max_context)
            max_target = max_context - n_context
            n_target = np.random.randint(max_target)
        else:
            n_context = max_context
            n_target = 0
    # print(f'n_context: {n_context}\tn_target: {n_target}')

    context_geoms,context_grasp_points, context_grasp_curvatures, context_grasp_normals, \
        context_midpoints, context_forces, context_labels = [], [], [], [], [], [], []
    target_geoms, target_grasp_points, target_grasp_curvatures, target_grasp_normals, \
        target_midpoints, target_forces, target_labels = [], [], [], [], [], [], []
    full_meshes = []
    object_properties = []
    for context_data, heldout_data in items:
        full_meshes.append(heldout_data['object_mesh'].swapaxes(0, 1))
        object_properties.append(heldout_data['object_properties'])
        if context_data is None:
            all_context_geoms = heldout_data['grasp_geometries']
            all_context_grasp_points = heldout_data['grasp_points']
            all_context_grasp_curvatures = heldout_data['grasp_curvatures']
            all_context_grasp_normals = heldout_data['grasp_normals']
            all_context_midpoints = heldout_data['grasp_midpoints']
            all_context_forces = heldout_data['grasp_forces']
            all_context_labels = heldout_data['grasp_labels']

            # We are training and will reuse context pool.
            random_ixs = np.random.permutation(all_context_geoms.shape[0])
            context_ixs = random_ixs[:n_context]
            target_ixs = random_ixs[:(n_context + n_target)]

            context_geoms.append(all_context_geoms[context_ixs, ...].swapaxes(1, 2))
            context_grasp_points.append(all_context_grasp_points[context_ixs, ...])
            context_grasp_curvatures.append(all_context_grasp_curvatures[context_ixs, ...])
            context_grasp_normals.append(all_context_grasp_normals[context_ixs, ...])
            context_midpoints.append(all_context_midpoints[context_ixs, ...])
            context_forces.append(all_context_forces[context_ixs])
            context_labels.append(all_context_labels[context_ixs])

            target_geoms.append(all_context_geoms[target_ixs, ...].swapaxes(1, 2))
            target_grasp_points.append(all_context_grasp_points[target_ixs, ...])
            target_grasp_curvatures.append(all_context_grasp_curvatures[target_ixs, ...])
            target_grasp_normals.append(all_context_grasp_normals[target_ixs, ...])
            target_midpoints.append(all_context_midpoints[target_ixs, ...])
            target_forces.append(all_context_forces[target_ixs])
            target_labels.append(all_context_labels[target_ixs])
        else:
            # We are testing and will keep context and targets separate.
            context_geoms.append(context_data['grasp_geometries'].swapaxes(1, 2))
            context_grasp_points.append(context_data['grasp_points'])
            context_grasp_curvatures.append(context_data['grasp_curvatures'])
            context_grasp_normals.append(context_data['grasp_normals'])
            context_midpoints.append(context_data['grasp_midpoints'])
            context_forces.append(context_data['grasp_forces'])
            context_labels.append(context_data['grasp_labels'])

            target_geoms.append(heldout_data['grasp_geometries'].swapaxes(1, 2))
            target_grasp_points.append(heldout_data['grasp_points'])
            target_grasp_curvatures.append(heldout_data['grasp_curvatures'])
            target_grasp_normals.append(heldout_data['grasp_normals'])
            target_midpoints.append(heldout_data['grasp_midpoints'])
            target_forces.append(heldout_data['grasp_forces'])
            target_labels.append(heldout_data['grasp_labels'])

    context_geoms = torch.Tensor(np.array(context_geoms).astype('float32'))
    context_grasp_points = torch.Tensor(np.array(context_grasp_points).astype('float32'))
    context_grasp_curvatures = torch.Tensor(np.array(context_grasp_curvatures).astype('float32'))
    context_grasp_normals = torch.Tensor(np.array(context_grasp_normals).astype('float32'))
    context_midpoints = torch.Tensor(np.array(context_midpoints).astype('float32'))
    context_forces = torch.Tensor(np.array(context_forces).astype('float32'))
    context_labels = torch.Tensor(np.array(context_labels).astype('float32'))

    target_geoms = torch.Tensor(np.array(target_geoms).astype('float32'))
    target_grasp_points = torch.Tensor(np.array(target_grasp_points).astype('float32'))
    target_grasp_curvatures = torch.Tensor(np.array(target_grasp_curvatures).astype('float32'))
    target_grasp_normals = torch.Tensor(np.array(target_grasp_normals).astype('float32'))
    target_midpoints = torch.Tensor(np.array(target_midpoints).astype('float32'))
    target_forces = torch.Tensor(np.array(target_forces).astype('float32'))
    target_labels = torch.Tensor(np.array(target_labels).astype('float32'))

    full_meshes = torch.Tensor(np.array(full_meshes).astype('float32'))
    object_properties = torch.Tensor(np.array(object_properties).astype('float32'))
    # Add random rotation.
    rot_mat = torch.qr(torch.randn(3, 3))[0]
    # import IPython; IPython.embed()

    # context_midpoints = context_midpoints@rot_mat
    # target_midpoints = target_midpoints@rot_mat

    context_grasp_points[:, :, 0] = context_grasp_points[:, :, 0]@rot_mat
    context_grasp_points[:, :, 1] = context_grasp_points[:, :, 1]@rot_mat
    context_grasp_normals = context_grasp_normals@rot_mat
    target_grasp_points[:, :, 0] = target_grasp_points[:, :, 0]@rot_mat
    target_grasp_points[:, :, 1] = target_grasp_points[:, :, 1]@rot_mat
    target_grasp_normals = target_grasp_normals@rot_mat

    full_meshes[:, :3, :] = (full_meshes[:, :3, :].transpose(1, 2)@rot_mat).transpose(1, 2)
    if add_mesh_normals:
        full_meshes[:, 3:6, :] = (full_meshes[:, 3:6, :].transpose(1, 2)@rot_mat).transpose(1, 2)
    object_properties[:, 0:3] = object_properties[:, 0:3]@rot_mat

    return (
        (context_geoms,
         context_grasp_points,
         context_grasp_curvatures,
         context_grasp_normals,
         context_midpoints,
         context_forces,
         context_labels),
        (target_geoms,
         target_grasp_points,
         target_grasp_curvatures,
         target_grasp_normals,
         target_midpoints,
         target_forces,
         target_labels),
        (full_meshes,
         object_properties)
    )


def custom_collate_fn_all_grasps(items):
    return custom_collate_fn(items, rand_grasp_num=False)


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader

    train_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-gnp/grasps/training_phase/train_grasps.pkl'
    val_dataset_fname = 'learning/data/grasping/train-sn100-test-sn10-robust-gnp/grasps/training_phase/val_grasps.pkl'
    print('Loading train dataset...')
    with open(train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    print('Loading val dataset...')
    with open(val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)

    print('Loading Train Dataset')
    train_dataset = CustomGNPGraspDataset(data=train_data)
    print('Loading Val Dataset')
    val_dataset = CustomGNPGraspDataset(data=val_data, context_data=train_data)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        collate_fn=custom_collate_fn,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        collate_fn=custom_collate_fn,
        batch_size=64,
        shuffle=False
    )

    for batch in train_dataloader:
        print(f'---- {len(batch)} ----')
        for elem1 in batch:
            for elem in elem1:
                print(elem.shape)

    for batch in val_dataloader:
        print(f'---- {len(batch)} ----')
        for elem1 in batch:
            for elem in elem1:
                print(elem.shape)
