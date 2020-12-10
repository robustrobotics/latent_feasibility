from collections import namedtuple
import argparse
import pickle

from planning.tree import Tree
from planning.problems import Tallest, Overhang, Deconstruct, NodeValue
from block_utils import Object
from learning.active.utils import ActiveExperimentLogger

def plan(timeout, blocks, problem, model):
    tree = Tree(init_value = NodeValue([], blocks))
    for t in range(timeout):
        
        best_node, best_node_value = tree.get_best_node()
        print(t, len(best_node_value.tower), tree.nodes[best_node].cost)
        new_values, costs, leaves = problem.sample_actions(best_node_value, model)
        for new_value, cost, leaf in zip(new_values, costs, leaves):
            tree.expand(new_value, cost, best_node, leaf)
    return tree.get_best_node(expand=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', 
                        choices=['tallest', 'overhang', 'deconstruct'], 
                        default='tallest',
                        help='planning problem/task to plan for')
    parser.add_argument('--block-set-fname', 
                        type=str, 
                        default='',
                        help='path to the block set file. if not set, random 5 blocks generated.')
    parser.add_argument('--exp-path', 
                        type=str, 
                        required=True)
    parser.add_argument('--timeout',
                        type=int,
                        default=1000,
                        help='max number of iterations to run planner for')
    parser.add_argument('--debug',
                        action='store_true',
                        help='set to run in debug mode')
    args = parser.parse_args()
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    n_blocks = 5
    tx = 99
    
    if args.block_set_fname is not '':
        with open(args.block_set_fname, 'rb') as f:
            block_set = pickle.load(f)
    else:
        block_set = [Object.random(f'obj_{ix}') for ix in range(n_blocks)]
        
    if args.problem == 'tallest':
        problem = Tallest()
    elif args.problem == 'overhang':
        problem = Overhang()
    elif args.problem == 'deconstruct':
        problem = Deconstruct()
        
    logger = ActiveExperimentLogger(args.exp_path)
    ensemble = logger.get_ensemble(tx)
    
    plan(args.timeout, block_set, problem, ensemble)