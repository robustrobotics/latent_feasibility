import os
import argparse
import torch
import numpy as np 
from datetime import datetime, timedelta
import math
from learning.models.push_np.attention_push_np import APNPDecoder
from learning.models.push_np.particle_filter import get_model 
from learning.domains.pushing.find_contact_points import run_sim 
from scipy.spatial.transform import Rotation as R
import pickle 

from pb_robot.planners.antipodalGraspPlanner import (
    GraspSimulationClient,
    GraspableBody,
)

""" 
Approach:
- Use previous outcomes to influence the prior belief on the distribution.
- Use the decoder to give probabilities for each outcome for a given probability.
- The "particle filter" will act as the prior for a given state.
- Use beliefs to assist in this process.
- Ensure the state represents the distance to the goal location.
"""

class POMCPOW:
    def __init__(self, model, dataset, num_particles, goal_loc=(100, 100, 0, 0), 
                 discount_factor=0.95, alpha=0.5, beta=0.5): 
        self.child = {}
        self.belief = {}  # Dictionary of form: {history: list of (state, probability)}
        self.value = {} 
        self.n = {}
        self.m = {} 
        self.discount_factor = discount_factor
        self.goal_loc = np.array(goal_loc)
        self.alpha = alpha  # Hyperparameter for action selection
        self.beta = beta    # Hyperparameter for observation handling
        self.dataset = dataset
        if torch.cuda.is_available(): 
            self.model = model.cuda() 
        else:
            self.model = model
        self.num_particles = num_particles 

    def get_belief(self, history, com): 
        differences = []
        actions = []
        angles = [np.array([0])]
        for i in range(0, len(history), 2):  # Corrected loop range
            if i == 0: 
                differences.append(self.goal_loc[:3] - history[i + 1][:3])
                actions.append(history[i])
            else: 
                differences.append(history[i + 1][:3] - history[i - 1][:3])
                actions.append(history[i]) 
            angles.append(np.array(history[i + 1][3])) 

        probability = np.log(1) 
        for i in range(len(differences)): 
            body_params = torch.cat((
                torch.from_numpy(com).cuda().float(), 
                torch.zeros(1).cuda(),
                torch.tensor([0.2, 0.07]).cuda()
            )) 
            sim_params = torch.cat((
                torch.from_numpy(actions[i]) * math.pi / 180, 
                torch.from_numpy(angles[i]), 
                torch.tensor([0.1, 0.0]).cuda()
            ))
            
            distributions = self.model(sim_params, None, None, None, body_params)[0] 
            probability += distributions[0][0].log_prob(torch.from_numpy(differences[i]).cuda()).sum()

        return torch.exp(probability).cpu().numpy()

    def search(self, history, time_seconds, depth=5):
        # Convert list to tuple for hashing
        history = tuple(history)
        start_time = datetime.now() 
        end_time = start_time + timedelta(seconds=time_seconds)

        particles = []
        for _ in range(self.num_particles): 
            if history == (): 
                # Uniform probability
                particles.append((np.concatenate((np.random.rand(2), (0, 0, 0, 0))), 1/self.num_particles)) 
            else: 
                # Sample based on belief
                location = (history[-1][2], history[-1][3], history[-1][4], history[-1][5])
                com = np.random.rand(2) 

                belief = self.get_belief(history, com) 
                particles.append((np.concatenate((com, (0, 0, 0, 0))), belief)) 

        while datetime.now() < end_time:
            total_prob = sum(prob for _, prob in particles)
            sample = np.random.rand() * total_prob
            current_prob = 0 
            selected_state = None
            for state, prob in particles:
                current_prob += prob 
                if current_prob > sample:
                    selected_state = state
                    break 
            if selected_state is None:
                selected_state = particles[-1][0]
                
            self.simulate(selected_state, history, depth)

        # After search, select the best action based on visit counts or values
        if history in self.child:
            best_action = max(
                self.child[history],
                key=lambda action: self.value.get(history + (action,), 0)
            )
            return best_action
        else:
            # If no actions have been explored, return a random action
            return 360 * np.random.rand()

    def action_prog_widen(self, history): 
        if history not in self.n: 
            self.n[history] = 0 
            self.value[history] = 0 
            self.child[history] = [] 

        max_actions = max(1, int(self.n[history] ** self.alpha)) 
        if len(self.child[history]) < max_actions: 
            new_action = 360 * np.random.rand() 
            if new_action not in self.child[history]:
                self.child[history].append(new_action)
                if (history + (new_action,)) not in self.n:
                    self.n[history + (new_action,)] = 0
                    self.value[history + (new_action,)] = 0
                return new_action 

        c = 2.0 
        best_value = float("-inf") 
        best_action = None  

        for action in self.child[history]:
            hist_a = history + (action,)
            if self.n[hist_a] == 0:
                return action
                
            # UCT formula: Q(s,a) + c * sqrt(log(N(s))/N(s,a))
            exploration = c * np.sqrt(math.log(self.n[history]) / self.n[hist_a])
            value = self.value[hist_a] + exploration
            
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action

    def simulate(self, state, history, depth):
        if depth == 0: 
            return 0 

        # Ensure history is a tuple
        history = tuple(history)

        action = self.action_prog_widen(history) 
        new_state, observation, reward, prob = self.do_singular_action(state, action) 
        hist_a = tuple(history + (action,))
        if hist_a not in self.value:
            self.value[hist_a] = 0
        if hist_a not in self.n:
            self.n[hist_a] = 0
        if hist_a not in self.child: 
            self.child[hist_a] = []
        hist_ao = tuple(hist_a + tuple(observation))
        # print(hist_ao)
        
        # Progressively widen over the observations
        if len(self.child[hist_a]) <= self.n[hist_a] ** self.beta:
            if hist_ao not in self.m:
                self.m[hist_ao] = 0
            self.m[hist_ao] += 1
        else: 
            probabilities = [self.m.get(hist_a + (o,), 0) for o in self.child[hist_a]]
            if sum(probabilities) == 0:
                observation = self.child[hist_a][np.random.choice(len(self.child[hist_a]))]
            else:
                observation = self.child[hist_a][np.random.choice(len(self.child[hist_a]), 
                                    p=np.array(probabilities) / sum(probabilities))]
            hist_ao = hist_a + (observation,) 

        # Initialize or update belief state
        if hist_ao not in self.belief:
            self.belief[hist_ao] = [(new_state, prob)]
        else:
            found = False
            for i in range(len(self.belief[hist_ao])):
                if np.array_equal(self.belief[hist_ao][i][0], new_state):
                    self.belief[hist_ao][i] = (new_state, self.belief[hist_ao][i][1] + prob)
                    found = True
                    break
            
            if not found:
                self.belief[hist_ao].append((new_state, prob))

        if hist_a not in self.child:
            self.child[hist_a] = []
            self.child[hist_a].append(observation)
            total = reward + self.discount_factor * self.rollout(new_state, hist_ao, depth - 1)
        else:
            # Check if the observation exists in child[hist_a] using array comparison
            observation_exists = any(np.array_equal(observation, obs) for obs in self.child[hist_a])
            if not observation_exists:
                self.child[hist_a].append(observation)
                total = reward + self.discount_factor * self.rollout(new_state, hist_ao, depth - 1)
            else:
                states = [state_prob[0] for state_prob in self.belief[hist_ao]]
                probabilities = [state_prob[1] for state_prob in self.belief[hist_ao]]
                probabilities = np.array(probabilities) / sum(probabilities)
                selected_state = states[np.random.choice(len(states), p=probabilities)]
                
                reward = self.do_singular_action(selected_state, action)[2]
                total = reward + self.discount_factor * self.simulate(selected_state, hist_ao, depth - 1)

        self.n[history] += 1
        self.n[hist_a] += 1
        self.value[hist_a] += (total - self.value[hist_a]) / self.n[hist_a]
        return total

    def rollout(self, state, history, depth):
        # Ensure history is a tuple
        history = tuple(history)
        
        if depth == 0:
            return 0

        action = np.random.rand() 
        
        new_state, observation, reward, _ = self.do_singular_action(state, action)
        
        future_value = self.rollout(new_state, history + (action, observation), depth - 1)
        
        return reward + self.discount_factor * future_value

    def get_action_list(self, depth): 
        step_size = 360 * (depth + 1) / 600
        return list(np.arange(0, 360, step_size))

    def do_singular_action(self, state, action): 
        with torch.no_grad(): 
            print(state, action)
            state_tensor = torch.tensor(state).cuda().float()
            action_tensor = torch.from_numpy(np.array([action])).float().cuda()
            
            body_params = torch.cat((
                state_tensor[0:2],
                torch.zeros(1).cuda(),
                torch.tensor([0.2, 0.07]).cuda()
            )).cuda()
            
            sim_params = torch.cat([
                action_tensor * math.pi / 180, 
                torch.tensor([0.1]).cuda(), 
                state_tensor[5:6]
            ]).cuda() 
            body_params = body_params.unsqueeze(0) 
            sim_params = sim_params.unsqueeze(0).unsqueeze(0)

            distributions = self.model(sim_params, None, None, None, body_params)[0]
            translation = distributions[0][0].mean[:3] 
            rotation = distributions[0][0].mean[3] 

            # Translation needs to be flipped back to world coordinates
            invert = {"final_position": translation, }
            inverted = self.dataset.inverse_transform(invert) 
            translation = inverted["final_position"] 

            # rotation = R.from_matrix(translation[0].mean[:3, :3]).as_euler('xyz', degrees=True)[2] 
            prob = torch.exp(distributions[0][0].log_prob(distributions[0][0].mean)) 

            new_state = (
                state[0], 
                state[1], 
                translation[0].item() + state[2], 
                translation[1].item() + state[3], 
                translation[2].item() + state[4], 
                ((rotation + state[5]) % (2 * math.pi)).item())

            observation = state[2:] 
            reward = -abs(new_state[2] - self.goal_loc[0]) - abs(new_state[3] - self.goal_loc[1])
            return new_state, observation, reward, prob.item()
    
    def planning_loop(self, com, iter=10, depth=3): 
        history = ()
        state = (com[0], com[1], 0, 0, 0, 0)  # (com_x, com_y, pos_x, pos_y, pos_z, angle)

        for _ in range(iter):
            best_action = self.search(history, 5, depth=depth) 
            state = (0, 0) + (history[-1] if len(history) > 0 else (0, 0, 0, 0)) 
            observation = real_simulate(state, best_action, self.dataset)
            print("HI ", state, best_action, observation)
            history = history + (best_action, observation)

        return history


def real_simulate(state, action, dataset): 
    inverse_dict = {
        "com": state[0:2] + (0,),
        "angle": action 
    }
    inversed = dataset.inverse_transform(inverse_dict)
    com = inversed["com"] 
    angle = inversed["angle"]
    body = GraspableBody("Primitive::Box_Test", com, 0.2, 0.07) 
    sim_client = GraspSimulationClient(body, False) 
    urdf = sim_client._get_object_urdf(body) 
    sim_client.disconnect() 

    transformation, contact_points, initial, _ = run_sim(urdf, angle, state[5], 0.1, 0, gui=False)  
    translation = transformation[:3, 3]
    rotation = R.from_matrix(transformation[:3, :3]).as_euler('xyz', degrees=True) 
    return translation + (rotation[-1])


def main(args): 
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset)
    instance_path = os.path.join(dataset_path, args.instance) 

    if not os.path.exists(instance_path): 
        os.makedirs(instance_path) 

    best_model_path = os.path.join(instance_path, 'best_model.pth') 
    if not os.path.exists(best_model_path): 
        model = get_model(args)  
    else: 
        model = APNPDecoder(args, point_net_encoding_size=64, point_cloud=True, d_latents=0)
        model.load_state_dict(torch.load(best_model_path))

    with open(os.path.join(instance_path, 'train_dataset.pkl'), 'rb') as f: 
        train_dataset = pickle.load(f)
    with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'rb') as f: 
        validation_dataset = pickle.load(f) 
    
    pomcpow = POMCPOW(model, validation_dataset, num_particles=100, alpha=0.5, beta=0.5)
    example_plan = pomcpow.planning_loop((0.5, 0.5), iter=10, depth=3) 
    for i in range(len(example_plan)): 
        if i % 2 == 0: 
            print("Action: ", example_plan[i]) 
        else: 
            print("Observation: ", example_plan[i]) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--instance', type=str, required=True) 
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-points', type=int, default=20)  # Set to 20 as per args in main
    parser.add_argument('--batch-size', type=int, default=32) 
    parser.add_argument('--guess-obj', action='store_true')

    args = parser.parse_args() 
    args.no_deterministic = True
    args.use_obj_prop = True
    args.guess_obj = False
    args.latent_samp = -1
    args.point_cloud = True
    args.no_contact = True 

    main(args)
