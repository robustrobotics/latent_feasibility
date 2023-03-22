import argparse
import copy

import trimesh
import pickle
import os

OBJECT_STORAGE_PKLS = ['train_geo_train_props.pkl', 'test_geo_test_props.pkl', 'train_geo_test_props.pkl']

def run(args):
    dataset_path = args.dataset
    for file_to_process in OBJECT_STORAGE_PKLS:
        path_to_object_pickle = os.path.join(dataset_path, 'objects', file_to_process)

        try:
            with open(path_to_object_pickle, 'rb') as handle:
                object_sample_dict = pickle.load(handle)


            # next, find COM indices
            index_ref = object_sample_dict['object_data']['property_names']
            i_com_x, i_com_y, i_com_z = index_ref.index('com_x'), index_ref.index('com_y'), index_ref.index('com_z')

            # next, loop through all objects and properties and set com to be the center of mass
            # and then set them

            new_object_properties = []
            for object_name, object_property in zip(
                    object_sample_dict['object_data']['object_names'],
                    object_sample_dict['object_data']['object_properties']
            ):
                _, object_hash = object_name.split('::', 1)[1].split('_', 1)
                mesh_path = os.path.join(os.environ['SHAPENET_ROOT'], 'visual_models', object_hash + '_centered.obj')
                com = trimesh.load(mesh_path).center_mass

                new_object_property = copy.deepcopy(object_property)
                new_object_property[i_com_x], new_object_property[i_com_y], new_object_property[i_com_z] = com
                new_object_properties.append(new_object_property)

            object_sample_dict['object_data']['object_properties'] = new_object_properties
            with open(path_to_object_pickle, 'wb') as handle:
                pickle.dump(object_sample_dict, handle)

        except EOFError:
            print('Looks like ' + file_to_process + ' reached EOF quicker than expected.')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    run(args)
