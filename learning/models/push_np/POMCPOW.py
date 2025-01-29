import os
import argparse
import torch
import numpy as np 
from datetime import datetime, timedelta
import math
from learning.models.push_np.attention_push_np import APNPDecoder
from learning.models.push_np.particle_filter import get_model 

""" 
How should we approach this?

We have that we need to use previous outcomes to influence the prior belief on the distribution. 
So what we can do is take our decoder and tell it to give probabilities for each outcome for a given probability.
That "particle filter" will then effectively be the prior for a given state. 

We can maybe use the beliefs to also help us out in this way.

TODO: We need to make sure that the state represents how much distance to the goal_loc
"""

class POMCPOW:
    def __init__(self, model, num_particles, goal_loc=(100, 100, 0, 0), discount_factor=0.95): 
        self.child = {}
        self.belief = {}  # Dictionary of form: {history: list of (state, probability)}
        self.value = {} 
        self.n = {}
        self.discount_factor = discount_factor
        self.goal_loc = np.array(goal_loc)
        if torch.cuda.is_available(): 
            self.model = model.cuda() 
        else:
            self.model = model

    def get_belief(self, history, com): 
        differences = []
        actions = []
        angles = [np.array([0])]
        for i in range(len(history), step=2): 
            if i == 0: 
                differences.append(self.goal_loc[:3] - history[i + 1][:3])
                actions.append(history[i])
            else: 
                differences.append(history[2 * i + 1][:3] - history[2 * i - 1][:3])
                actions.append(history[2 * i]) 
            angles.append(history[2 * i + 1][3]) 

        probability = np.log(1) 
        for i in range(len(differences)): 
            body_params = torch.cat((
                torch.from_numpy(com).cuda().float(), 
                torch.zeros(1).cuda(),
                torch.tensor([0.2, 0.07]).cuda()
            )) 
            action = torch.cat((torch.from_numpy(actions[i]).cuda().float(), to
        # Now we know differences is the list of ys 


    def search(self, history, time_seconds):
        # Convert list to tuple for hashing
        history = tuple(history)
        start_time = datetime.now() 
        end_time = start_time + timedelta(seconds=time_seconds)

        particles = []
        for _ in range(num_particles): 
            if history == []: 
                # Uniform probability
                particles.append((np.concatenate((np.random.rand(2), (0, 0, 0, 0))), 1/num_particles)) 
            else: 
                # Otherwise we say: 
                location = (history[-1][2], history[-1][3], history[-1][4], history[-1][5])
                com = np.random.rand(2) 



        while datetime.now() < end_time:
            if not history:  # Empty tuple check
                # No this is not what we want 
                state = np.concatenate((np.random.rand(2), (0, 0, 0, 0)))
            else: 
                total_prob = 0 
                for state, prob in self.belief[history]: # This is (action, observation...)
                    total_prob += prob 
                current_prob = 0 
                sample = np.random.rand() * total_prob
                selected_state = None
                for state, prob in self.belief[history]:
                    current_prob += prob 
                    if current_prob > sample:
                        selected_state = state
                        break 
                if selected_state is None:
                    selected_state = self.belief[history][-1][0]
                
            if history not in self.value:
                self.value[history] = 0
            if history not in self.n:
                self.n[history] = 0
                
            self.simulate(selected_state if history else state, history, depth=5)

    def simulate(self, state, history, depth):
        if depth == 0: 
            return 0 

        # Ensure history is a tuple
        history = tuple(history)

        if history not in self.child:
            self.child[history] = self.get_action_list(depth)
        if history not in self.value:
            self.value[history] = 0
        if history not in self.n:
            self.n[history] = 0

        action = np.random.choice(self.child[history])
        new_state, observation, reward, prob = self.do_singular_action(state, action) 

        # Create new history tuples
        hist_a = history + (action,)
        hist_ao = hist_a + (observation,)

        if hist_a not in self.value:
            self.value[hist_a] = 0
        if hist_a not in self.n:
            self.n[hist_a] = 0

        if hist_ao not in self.belief: 
            self.belief[hist_ao] = [(new_state, prob)] 
        else: 
            found = False
            for i in range(len(self.belief[hist_ao])):
                if self.belief[hist_ao][i][0] == new_state:
                    self.belief[hist_ao][i] = (new_state, self.belief[hist_ao][i][1] + prob) 
                    found = True
                    break 
            
            if not found: 
                self.belief[hist_ao].append((new_state, prob))

        if hist_a not in self.child or observation not in self.child[hist_a]:
            if hist_a not in self.child:
                self.child[hist_a] = []
            self.child[hist_a].append(observation)
            total = reward + self.discount_factor * self.rollout(new_state, hist_ao, depth - 1)
        else:
            states = [state_prob[0] for state_prob in self.belief[hist_ao]]
            probabilities = [state_prob[1] for state_prob in self.belief[hist_ao]]
            probabilities = np.array(probabilities) / sum(probabilities)
            selected_state = states[np.random.choice(len(states), p=probabilities)]
            
            reward = self.do_singular_action(selected_state, action)[2]
            total = reward + self.discount_factor * self.simulate(selected_state, hist_ao, depth - 1)

        self.n[history] = self.n[history] + 1
        self.n[hist_a] = self.n[hist_a] + 1
        self.value[hist_a] = self.value[hist_a] + (total - self.value[hist_a]) / self.n[hist_a]
        return total

    def rollout(self, state, history, depth):
        # Ensure history is a tuple
        history = tuple(history)
        
        if depth == 0:
            return 0

        possible_actions = self.get_action_list(depth)
        action = np.random.choice(possible_actions)
        
        new_state, observation, reward, _ = self.do_singular_action(state, action)
        
        future_value = self.rollout(new_state, history + (action, observation), depth - 1)
        
        return reward + self.discount_factor * future_value

    def get_action_list(self, depth): 
        step_size = 360 * (depth + 1) / 600
        return list(np.arange(0, 360, step_size))

    def do_singular_action(self, state, action): 
        with torch.no_grad(): 
            state_tensor = torch.from_numpy(np.array(state)).float().cuda()
            action_tensor = torch.from_numpy(np.array([action])).float().cuda()
            
            body_params = torch.cat((
                state_tensor[0:2],
                torch.zeros(1).cuda(),
                torch.tensor([0.2, 0.07]).cuda()
            )).cuda()
            
            sim_params = torch.cat((
                action_tensor * math.pi / 180,
                state_tensor[5:6],
                torch.tensor([0.1, 0.0]).cuda()
            )).cuda() 
            body_params = body_params.unsqueeze(0) 
            sim_params = sim_params.unsqueeze(0).unsqueeze(0)
            print(body_params.shape, sim_params.shape)

            distributions = self.model(sim_params, None, None, None, body_params)[0]
            translation = distributions.mean[:3] 
            rotation = distributions.mean[3] 
            prob = torch.exp(distributions.log_prob(distributions.mean)) 

            new_state = (state[0], state[1], translation[0].item() + state[2], 
                        translation[1].item() + state[3], translation[2].item() + state[4], 
                        rotation.item())
            observation = new_state[2:] 
            reward = -abs(new_state[0] - self.goal_loc[0]) - abs(new_state[1] - self.goal_loc[1]) 
            return new_state, observation, reward, prob.item()


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
    
    pomcpow = POMCPOW(model)
    pomcpow.search([], 2) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--instance', type=str, required=True) 
    parser.add_argument('--num-epochs', type=int, default=50)
    # parser.add_argument('--use-obj-prop', action='store_true') 
    # parser.add_argument('--use-full-trajectory', action='store_true') 
    parser.add_argument('--num-points', type=int, default=20)  # Set to 20 as per args in main
    parser.add_argument('--batch-size', type=int, default=32) 
    # parser.add_argument('--d-latents', type=int, default=5)
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--attention-encoding', type=int, default=512)
    # parser.add_argument('--learning-rate', type=float, default=1e-3)
    # parser.add_argument('--use-mixture' , action='store_true')
    # parser.add_argument('--regression', action='store_true')
    # parser.add_argument('--no-deterministic', action='store_true')
    # parser.add_argument('--latent-samp', type=int, default=-1) 
    parser.add_argument('--guess-obj', action='store_true')
    parser.add_argument('--no-contact', action='store_true')

    args = parser.parse_args() 
    args.no_deterministic = True
    args.use_obj_prop = True
    args.guess_obj = False
    args.latent_samp = -1
    args.point_cloud = True

    # args = parser.parse_args() 
    main(args)
