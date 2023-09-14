# stacking_ros

This package provides a Task and Motion Planning (TAMP) server that offers ROS services to interact with the Panda stacking agent.

## Setup

First, ensure this package is in a Catkin workspace (e.g. `/catkin_ws/src/stacking/stacking_ros`).

Now, build the Catkin workspace

```
catkin_build -DCPYTHON_EXECUTABLE=$(which python3.7)
```

Ensure that this Catkin workspace is being sourced:

```
setup /catkin_ws/devel/setup.bash
```

To check that the package works correctly, try running some of these commands:

```
rospack find stacking_ros
rosmsg list | grep stacking
```

---

## Tower Simulation Usage
First, start up a Panda agent that communicates with the planning server. For example,

```
python3.7 -m run_towers --use-action-server --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

The `scripts/planning_server.py` file contains the planning server that will feed plans to the Panda agent. To run this, you have a few options.

### rosrun
```
rosrun stacking_ros planning_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

To do this, you will need to ensure the `stacking` repo is on the Python path. You can force this, e.g.
```
export PYTHONPATH=$PYTHONPATH:/catkin_ws/src/stacking
```

### Python
First, go to the top-level folder of the `stacking` repository. Then,

```
python3.7 stacking_ros/scripts/planning_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10

```

---

## Active Learning Usage
First, start the planning server with the block list of choice. Keep it handy because we need to use it for every separate call below.

```
rosrun stacking_ros planning_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

Then, start the Panda agent server with the same block list.

```
rosrun stacking_ros panda_agent_server.py --blocks-file learning/domains/towers/final_block_set.pkl --num-blocks 10
```

Finally, start active learning.

```
python3.7 -m learning.experiments.active_train_towers --exec-mode sim --use-panda-server --block-set-fname learning/domains/towers/final_block_set.pkl
```

## Grasping Usage
On Ted (RTK), start one nodes:

1. Franka Interface: `roslaunch franka_interface interface.launch`

On narrator (GPU), start two more nodes:

1. Panda Vision: `roslaunch panda_vision vision.launch`
2. Panda Agent Server: `rosrun stacking_ros panda_agent_server.py  --real --use-vision --blocks-file learning/domains/towers/grasping_block_set_robot.pkl --block-id <block_id>`

Continuing on narrator, run one of the following:

1. To open loop execute a plan: `python agents.panda_agent`
2. Fitting script: `python -m learning.experiments.active_fit_grasping_pf --exp-name real-robot-block0 --objects-fname learning/domains/towers/grasping_block_set_sim.pkl --pretrained-ensemble-exp-path learning/experiments/logs/grasp_gnp-boxv2-run1_train-20230406-110724  --strategy random --exec-mode real --use-progressive-priors --likelihood gnp --max-acquisitions 10 --eval-object-ix 16`
3. Eval script: `python -m learning.evaluate.evaluate_grasping --exec-mode real --exp-path real-robot-block0 --n-samples 100 --eval-tx 4 --task likely-grasp`
