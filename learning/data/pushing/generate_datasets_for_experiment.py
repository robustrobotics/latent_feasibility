import argparse
import os


# the equivalent function is defined in generate_dtasets_for_grasping.py
# makes more sense just to include it in this script.
def generate_push_data_for_obj(obj, config):
    # Call subroutines/helpers to run pybullet and get you your data!

    # store data here.
    obj_data = {
        # this is not definite. just giving you an idea
        'obj': obj,
        'physics': None,
        # other params you need here.
        # list of grasps, outcomes, helpful meshes, you name it.
    }

    return obj_data

def merge_into_gnp_format(list_of_object_sets):
    # don't imitate grasping/generate_datasets_for_experiment.py...
    # merge the object datasets together into the format specified by
    # process_and_save() and process_geometry() in create_gnp_data.py
    return {}


def main(args):
    # load up train objects make the set
    train_objs = [] # load in these from the object lists

    train_sets = []
    for train_obj, obj_config in train_objs:
        train_sets.append(generate_push_data_for_obj(train_obj, obj_config))

    train_set = merge_into_gnp_format(train_sets)
    # store the set, likely as a huge .pkl
    # if it's __too__ large, then you can split the dataset up into grouped/individual objects.

    # do the same thing as above for evaluation data for the GNP loop.



if __name__ == '__main__':
    # (1)
    parser = argparse.ArgumentParser()
    # define command line arguments
    # Consume object lists (filename/path)
        # Shapenet-net-sem, box-data <-- folders that contain the object meshes.
            # Missing (you must generate: physical properties)
            # Action data and outcomes.
        # object list tells  you which meshes you should use to generate physical properties + actions.
    # Configuration params: how many pushes per object? pushing surface properties?
    # Seed for random number generation (Seiji uses numpy's)
    # You may need others!

    # parser.add_arguments...

    # parse command line arguments
    args = parser.parse_args()

    # Set up data directory.
    # Good practice: save the running arguments (either as a .pkl or .json)
    # so you can remember your settings! A future you will thank you.

    # A personal practice of mine (you can refuse to go with it) is to define
    # all main script code in a function at the top.
    main(args)
