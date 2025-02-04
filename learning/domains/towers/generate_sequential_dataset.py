import argparse
import numpy as np
import pickle
from random import choices
from scipy.spatial.transform import Rotation as R

from block_utils import (
    all_rotations,
    Position,
    Quaternion,
    Pose,
    Object,
    get_rotated_block,
    World, Environment
)
from learning.domains.towers.tower_data import TowerDataset
from tower_planner import TowerPlanner

QUATERNIONS = np.array([Quaternion(*o.as_quat()) for o in all_rotations()])

def get_block_set(args):
    if args.block_set != "":
        vector_block_set = np.load(args.block_set)
        block_set = [
            Object.from_vector(v, name=f"obj_{i}")
            for i, v in enumerate(vector_block_set)
        ]
        args.block_set_size = vector_block_set.shape[0]
        print(f"Loaded {args.block_set_size}-block set from {args.block_set}")
        return block_set
    elif args.block_set_size > 0:
        print(f"Generating new {args.block_set_size}-block set.")
        return [Object.random(f"obj_{i}") for i in range(args.block_set_size)]
    else:
        print(f"No block set. Using random blocks.")
        return None


def get_num_towers_per_cat(args, multiple=8):
    num_per_cat = int(np.ceil(args.num_towers / float(multiple)))
    if num_per_cat * multiple != args.num_towers:
        print(f"num_towers should be a multiple of {multiple}.", end=" ")
        print(f"Rounding {args.num_towers} to {num_per_cat * multiple}.")
        args.num_towers = num_per_cat * multiple

    print(f"Generating {num_per_cat * 2} towers for each height.")
    return num_per_cat


def choose_blocks_in_tower(block_set, height):
    if block_set is None:
        return [Object.random(f"obj_{0}") for _ in range(blocks)]
    else:
        return np.random.choice(block_set, height, replace=False)


def choose_rotations_in_tower(height):
    idxs = np.random.choice(np.arange(QUATERNIONS.shape[0]), height, replace=True)
    return QUATERNIONS[idxs]


def build_tower(blocks, rotations, label, max_attempts=250, prerotate=False, fixed_positions=False):
    tp = TowerPlanner(stability_mode="contains")
    # pre-rotate each object to compute its dimensions in the tower
    dimensions = np.array([b.dimensions for b in blocks])
    rotated_dimensions = np.abs(R.from_quat(rotations).apply(dimensions))
    # get the maximum relative displacement of each block:
    # how far each block can be moved w/ losing contact w/ the block below
    max_displacements_xy = (
        rotated_dimensions[1:, :2] + rotated_dimensions[:-1, :2]
    ) / 2.0

    # calculate the z-position of each block in the tower
    cumulative_heights = np.cumsum(rotated_dimensions[:, 2])
    pos_z = rotated_dimensions[:, 2] / 2
    pos_z[1:] += cumulative_heights[:-1]

    for _ in range(max_attempts):
        # sample unscaled noise (clip bceause random normal can exceed -1, 1)
        noise_xy = np.clip(
            0.5 * np.random.randn(*max_displacements_xy.shape), -0.95, 0.95
        )
        # and scale the noise by the max allowed displacement
        if not fixed_positions:
            rel_xy = max_displacements_xy * noise_xy
        else:
            rel_xy = np.zeros(max_displacements_xy.shape)
            rel_xy[::2, 0] = 0.035
            rel_xy[1::2, 0] = -0.035
        # place the first block at the origin
        rel_xy = np.vstack([np.zeros([1, 2]), rel_xy])
        # and get the actual positions by cumsum of the relative positions
        pos_xy = np.cumsum(rel_xy, axis=0)
        # combine positions
        pos_xyz = np.hstack([pos_xy, pos_z[:, None]])

        # set the block positions and compute the pre-rotated blocks
        rotated_tower = []
        for b, p, q in zip(blocks, pos_xyz, rotations):
            b.set_pose(Pose(Position(*p), Quaternion(*q)))
            rotated_tower.append(get_rotated_block(b))

        # worlds = [World(rotated_tower), World(blocks)]
        # env = Environment(worlds, vis_sim=True, vis_frames=True)
        # env.step(vis_frames=True)
        # input('Next?')

        # env.disconnect()

        # check if the base is stable and the top block matches the label
        if tp.tower_is_constructable(rotated_tower[:-1]) and (
            label == tp.tower_is_constructable(rotated_tower)
        ):
            if prerotate:
                return rotated_tower
            else:
                return blocks

    return None


def get_tower(block_set, height, label, prerotate, fixed_positions):
    blocks = choose_blocks_in_tower(block_set, height)
    rotations = choose_rotations_in_tower(height)
    return build_tower(blocks, rotations, label, prerotate=prerotate, fixed_positions=fixed_positions)


def get_sub_dataset(block_set, height, num_towers_per_cat, prerotate, fixed_positions):
    block_ids = []
    towers = []
    labels = []

    for label in [0, 1]:
        count = 0
        while count < num_towers_per_cat:
            # generate a new tower
            label_string = "stable" if label else "unstable"
            print(f"{count}/{num_towers_per_cat}\t{label_string} {height}-block tower")
            tower = get_tower(block_set, height, label, prerotate, fixed_positions)
            if tower is None:
                continue
            
            # if we successfully generate a tower, save it
            block_ids.append([int(b.name.strip("obj_")) for b in tower])
            towers.append([b.vectorize() for b in tower])
            labels.append(label)
            count += 1

    sub_dataset = {"towers": np.array(towers), "labels": np.array(labels)}
    if block_set is not None:
        sub_dataset["block_ids"] = np.array(block_ids)

    return sub_dataset


def get_filename(num_towers, use_block_set, block_set_size, suffix, prerotate):
    # create a filename for the generated data based on the configuration
    block_set_string = (
        f"{block_set_size}block_set" if use_block_set else "random_blocks"
    )
    if prerotate:
        suffix += '_prerotated'
    
    return f"learning/data/{block_set_string}_(x{num_towers})_{suffix}_dict.pkl"


def main(args):
    block_set = get_block_set(args)
    num_towers_per_cat = get_num_towers_per_cat(args)
    dataset = {}

    for height in range(2, 6):
        dataset[f"{height}block"] = get_sub_dataset(
            block_set, height, num_towers_per_cat, args.prerotate, args.fixed_positions
        )

    filename = get_filename(
        args.num_towers, block_set != None, args.block_set_size, args.suffix, args.prerotate
    )
    print("Saving to", filename)
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)

    if args.save_dataset_object:
        td_filename = filename.replace("dict.pkl", "dataset.pkl")
        print("Saving to", td_filename)
        with open(td_filename, "wb") as f:
            pickle.dump(TowerDataset(dataset, augment=True), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix", type=str, default="sequential", help="Add to filename."
    )
    parser.add_argument(
        "--block-set-size",
        type=int,
        default=10,
        help="Only used if block-set not specified.",
    )
    parser.add_argument(
        "--block-set", type=str, default="", help="Path to .npy vectorized block-set."
    )
    parser.add_argument(
        "--num-towers", type=int, default=4000, help="Total number to generate."
    )
    parser.add_argument(
        "--save-dataset-object",
        action="store_true",
        help="Create and save TowerDataset.",
    )
    parser.add_argument(
        "--prerotate",
        action="store_true",
        help="Rotate the blocks before saving the dataset."
    )
    parser.add_argument(
        "--fixed-positions",
        action="store_true",
        help="Don't vary relative positions of the blocks."
    )

    args = parser.parse_args()
    main(args)
