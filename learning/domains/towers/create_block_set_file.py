import csv
import pickle
import numpy as np
import argparse

from block_utils import Object, Dimensions, Color, Position
from learning.domains.grasping.generate_primitive_data import URDF_TEMPLATE

# this is used to generate a block_set.pkl file from random block_utils.Object() 
# blocks using the parameters there
def random_block_set(args):
    dim_range = (args.block_min_dim, args.block_max_dim)
    block_set = [Object.random('obj_'+str(n), dim_range=dim_range) 
                 for n in range(args.n_blocks)]
    pkl_filename = 'block_set_'+str(args.n_blocks)+'.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(block_set, f)

def string_to_list(string):
    string = string.replace('[', '').replace(']','')
    list_string = string.split(' ')
    list_float = [float(s) for s in list_string if s != '']
    return list_float

# this is used to generate a block_set.pkl file from a .csv file 
# the .csv files are generated in scripts/generate_block_set.py (for physical block sets)
def block_set_from_csv(args):
    block_set = []
    object_set = {
        'object_data': {
            'object_names': [],
            'object_properties': [],
            'property_names': ['com_x', 'com_y', 'com_z', 'mass', 'friction']
        },
        'metadata': {}
    }
    with open(args.csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_i, row in enumerate(csv_reader):
            if row_i > 0:
                id, dimensions, com, mass = row

                if args.output_mode == 'robot-pkl':
                    block = Object(
                        'obj_'+id,
                        Dimensions(*np.multiply(string_to_list(dimensions), .01)), # g --> kg
                        float(mass)*.001, # g --> kg
                        Position(*np.multiply(string_to_list(com), .01)), # cm --> m
                        Color(*np.random.rand(3))
                    )
                    print(block.dimensions, block.mass, block.com)
                    block_set.append(block)
                elif args.output_mode == 'urdf':
                    side_lengths = np.multiply(string_to_list(dimensions), .01).tolist()
                    urdf_text = URDF_TEMPLATE % (
                        2 * row_i,
                        2 * row_i + 1,
                        'box size=\"{0} {1} {2}\"'.format(*side_lengths),
                        'box size=\"{0} {1} {2}\"'.format(*side_lengths)
                    )

                    urdf_path = f'{args.urdf_dir}/urdfs/Box_{id}.urdf'
                    with open(urdf_path, 'w') as handle:
                        handle.write(urdf_text)
                elif args.output_mode == 'sim-pkl':
                    name = f'Primitive::Box_{id}'
                    props = np.array(string_to_list(com) + [float(mass), 0.2])
                    object_set['object_data']['object_names'].append(name)
                    object_set['object_data']['object_properties'].append(props)

    if args.output_mode == 'robot-pkl':
        pkl_filename = args.csv_file[:-4]+'_robot.pkl'
        print(f'Saving to {pkl_filename}.')
        with open(pkl_filename, 'wb') as f:
            pickle.dump(block_set, f)
    elif args.output_mode == 'sim-pkl':
        pkl_filename = args.csv_file[:-4]+'_sim.pkl'
        print(f'Saving to {pkl_filename}.')
        with open(pkl_filename, 'wb') as f:
            pickle.dump(object_set, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str)
    parser.add_argument('--n-blocks', type=int) # only need it mode == random
    parser.add_argument('--input-mode', choices=['csv', 'random'])
    parser.add_argument('--output-mode', choices=['robot-pkl', 'sim-pkl', 'urdf'])
    parser.add_argument('--urdf-dir', type=str)
    parser.add_argument('--block-min-dim', type=float, default=0.05)# only need it mode == random
    parser.add_argument('--block-max-dim', type=float, default=0.15)# only need it mode == random
    args = parser.parse_args()

    if args.input_mode == 'random':
        random_block_set(args)
    elif args.input_mode == 'csv':
        block_set_from_csv(args)
