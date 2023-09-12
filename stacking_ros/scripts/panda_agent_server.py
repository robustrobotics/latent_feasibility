#!/usr/bin/env python3.7
"""
Runs a PandaAgent as a server for multi-machine active learning
"""

import rospy
import argparse
from block_utils import Quaternion, Pose, Position
from agents.panda_agent import PandaAgent
from tamp.misc import load_eval_block

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-id', type=int, default=4)
    parser.add_argument('--real', default=False, action='store_true')
    parser.add_argument('--use-vision', default=False, action='store_true')
    parser.add_argument('--blocks-file', default='learning/domains/towers/final_block_set_10.pkl', type=str)
    args = parser.parse_args()

    # Load the blocks
    blocks = load_eval_block(
        blocks_fname=args.blocks_file,
        eval_block_id=args.block_id
    )
    block_init_xy_poses=[
        Pose(
            Position(0.4, 0.0, 0.0),
            Quaternion(0.0, 0.0, 0.0, 1.0)
        )
    ]

    # Create Panda agent and leave it running as ROS node
    agent = PandaAgent(
        blocks,
        block_init_xy_poses=block_init_xy_poses,
        use_vision=args.use_vision,
        real=args.real,
        use_action_server=False,
        use_learning_server=True
    )
    print("Panda agent server ready!")
    rospy.spin()
