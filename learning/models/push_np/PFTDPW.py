import argparse
import os 
import torch
import pickle
import numpy as np
import time
import math 
import scipy as sp

from learning.models.push_np.attention_push_np import APNPDecoder
from learning.models.push_np.dataset import PushNPDataset, collate_fn
from learning.domains.pushing.find_contact_points import run_sim   
from learning.models.push_np.particle_filter import get_model 
from pb_robot.planners.antipodalGraspPlanner import (
    GraspSimulationClient,
    GraspableBody,
)

from scipy.spatial.transform import Rotation as R 
from scipy.special import logsumexp 




class PFTDPW:
    def __init__(self, model, dataset, true_com, search_depth=3, num_particles=500, goal_loc=(100, 100, 0, 0), discount_factor=0.8, alpha=0.5, beta=0.5, const=2.0): 
        self.c = {} # child
        self.q = {} # value
        self.n = {} # number of visits
        if torch.cuda.is_available(): 
            self.model = model.cuda() 
        else:
            self.model = model
        self.dataset = dataset
        self.search_depth = search_depth
        self.num_particles = num_particles
        self.goal_loc = np.array(goal_loc)
        self.discount_factor = discount_factor
        self.alpha = alpha  # Hyperparameter for action selection
        self.beta = beta    # Hyperparameter for observation handling 
        self.const = const 
        self.true_com = true_com

        self.particles = np.random.rand(num_particles, 2).tolist()  # Assuming 2D particles 


    def plan(self, b=None, total_time=10.0):
        if b is None: 
            b = (tuple([1 / self.num_particles for _ in range(self.num_particles)]), (0, 0, 0, 0)) 
        start_time = time.time()
        while time.time() - start_time < total_time:
            self.simulate(b, self.search_depth) 
        best_value = float("-inf")
        best_action = 0 
        for action in self.c[b]:
            value = self.q[b + (action,)] 

            if value > best_value: 
                best_value = value
                best_action = action
                # print("Value: ", value)
                # print("Action: ", action) 

        # print(self.n.values())
        
        return best_action 

    
    def do_singular_action(self, state, action, angle): 
        # print(state, action, angle) 
        with torch.no_grad(): 
            # Validate inputs
            if np.isnan(angle) or np.isinf(angle):
                # print("Warning: Invalid angle input in do_singular_action:", angle)
                angle = 0.0
            if np.isnan(action) or np.isinf(action):
                # print("Warning: Invalid action in do_singular_action:", action)
                action = 0.0

            # Prepare inputs for the model

            # print("ACTION: ", action) 
            if not isinstance(action, tuple):
                action = (action,) 
            state_tensor = torch.tensor(state).cuda().float()
            action_tensor = torch.from_numpy(np.array([action])).float().cuda()
            
            # invert = {"angle": action_tensor} 
            # inverted = self.dataset.inverse_transform(invert) 
            # action_tensor = inverted["angle"] 

            body_params = torch.cat((
                state_tensor[0:2],
                torch.zeros(1).cuda(),
                torch.tensor([0.2, 0.07]).cuda()
            )).cuda().float()
            # print ("action tensor:", action_tensor)
            
            # Ensure all tensors are 1D before concatenation
            # Fix tensor construction warning by using clone().detach()
            sim_params = torch.cat([
                action_tensor[0].clone().detach().cuda() / 360,  # Flatten to 1D
                torch.tensor([0.1]).cuda(),
                torch.tensor([angle]).cuda()
            ]).cuda().float()
            
            body_params = body_params.unsqueeze(0) 
            sim_params = sim_params.unsqueeze(0).unsqueeze(0)

            # Add validation checks
            if torch.isnan(body_params).any() or torch.isinf(body_params).any():
                # print("Warning: body_params contains NaN or Inf values:", body_params)
                return np.zeros(4), None
            if torch.isnan(sim_params).any() or torch.isinf(sim_params).any():
                # print("Warning: sim_params contains NaN or Inf values:", sim_params)
                return np.zeros(4), None

            # print("Simulating with body_params:", body_params, "and sim_params:", sim_params)
            distributions = self.model(sim_params, None, None, None, body_params)[0]
            sample = distributions[0][0].sample()
            translation = sample[:3].cpu().numpy()  # Convert to numpy immediately
            rotation = sample[3].cpu().numpy()
            # print(translation, rotation) 

            # Translation needs to be flipped back to world coordinates
            # invert = {"final_position": translation}
            # inverted = self.dataset.inverse_transform(invert) 
            # print (invert, inverted)
            # translation = inverted["final_position"] 

            # Concatenate as a sequence and ensure rotation is reshaped properly
            # print(translation, rotation) 
            return np.concatenate([translation, [rotation]]), distributions[0][0]


    def update(self, b, a, obs=None): 
        probs = b[0] 
        observation = b[1] 
        total = np.zeros(4)
        # print("OBSERVATIONS", b[1]) 
        # print("PROBS: ", probs)
        # print("OBSERVATION: ", observation) 
        # print("ACTION: ", a)
        # print("OBS: ", obs)

        # Only process particles with significant probabilities for prediction
        active_mask = np.array(probs) >= 1e-9
        if np.sum(active_mask) == 0:
            max_idx = np.argmax(probs)
            active_mask[max_idx] = True

        for i in range(self.num_particles):
            if active_mask[i]:
                com = self.particles[i] 
                result = np.array(self.do_singular_action(com, a, observation[3])[0])
                total += result * probs[i]

        debug = False
        if (obs is not None):
            total = obs
            debug = True

        # print("ACTION EFFECT: ", total) 
        # else:
        #     # Compute the new observation after the action when no observation is provided
        #     # Use the true center of mass to simulate the action
        #     true_com = self.true_com[0:2]
        #     # Use do_singular_action to predict the next state
        #     next_state, distribution = self.do_singular_action(true_com, a, observation[3])
            
        #     # The next_state represents the absolute new state, not the difference
        #     # We need to compute the difference between this predicted state and the current observation
        #     # This aligns with the simplified only_train_distrib method mentioned in the memory
            
        #     # For position components (first 3), calculate the difference
        #     # For rotation component (last one), use the predicted value directly
        #     position_diff = next_state[:3] - np.array(observation[:3])
        #     rotation = next_state[3]
            
        #     # Combine the position difference and rotation into the total observation
        #     total = np.concatenate([position_diff, [rotation]])
        #     # print("COMPUTED NEW OBSERVATION:", total)

        # Prevent division by zero in log computation
        probs = np.clip(probs, 1e-300, 1.0)  # Clip to small positive values
        log_probs = np.log(probs)
        new_probs = np.array(log_probs)
        
        # Convert observation to numpy array and add total to get the new observation
        observation_array = np.array(observation)
        new_observation = observation_array + total
        
        # Update all particles' probabilities using distributions
        for i in range(self.num_particles): 
            com = self.particles[i] 
            distribution = self.do_singular_action(com, a, observation[3])[1] 
            
            # Convert observation to tensor and validate/normalize
            obs_tensor = torch.from_numpy(total).cuda().float()
            
            # Get distribution parameters
            mean = distribution.mean
            std = distribution.stddev
            if debug:
                print("Mean: ", mean)
            
            # # Clip observation to be within reasonable bounds (e.g., ±5 standard deviations)
            # clipped_obs = torch.clamp(obs_tensor, 
            #                         min=mean - 5*std,
            #                         max=mean + 5*std)
            
            # Compute log probability with clipped observation
            # try:
            #     log_prob = distribution.log_prob(clipped_obs).sum().item()
            #     log_prob = np.clip(log_prob, -1e10, 0)  # Clip extreme negative values
            # except ValueError as e:
            #     log_prob = -1e10  # Very low log probability for invalid cases

            log_prob = distribution.log_prob(obs_tensor).sum().item()
            
            new_probs[i] = log_prob + np.log(probs[i])
            # print("Mean: ", mean, "New prob: ", np.exp(new_probs[i]))

        # Normalize probabilities to ensure they sum to 1
        # new_probs = logsumexp(new_probs)
        new_probs = np.exp(new_probs - logsumexp(new_probs))

        # Convert to tuple for hashability
        # print("NEW PROBS: ", new_probs)
        new_probs = tuple(new_probs)

        return (new_probs, tuple(new_observation)), -np.linalg.norm(new_observation[:2] - self.goal_loc[:2])


    def simulate(self, b, depth): 
        # print("Depth: ", depth)
        if depth == 0: 
            return 0 
        a = self.action_prog_widen(b) 
        if not isinstance(a, tuple): 
            a = (a, ) 
        if b + a not in self.n: 
            self.n[b + a] = 0 
            self.q[b + a] = 0 
            self.c[b + a] = [] 

        total = 0
        if len(self.c[b + a]) <= self.n[b + a] ** self.alpha: 
            # We generate a new state. Basically we do a pass of the particle filter.  
            new_b, r = self.update(b, a) 
            self.c[b + a].append((new_b, r)) 
            total = r + self.discount_factor * self.simulate(new_b, depth - 1) 
        else: 
            random_index = np.random.choice(len(self.c[b + a]), 1)[0]
            new_b, r = self.c[b + a][random_index]
            total = r + self.discount_factor * self.simulate(new_b, depth - 1) 
        
            
        self.n[b] += 1 
        self.n[b + a] += 1
        self.q[b + a] = self.q.get(b + a, 0) + (total - self.q.get(b + a, 0)) / self.n[b + a] 
        return total

    def rollout(self, b, depth): 
        if depth == 0:
            return 0 
        a = self.action_prog_widen(b)
        new_b, r = self.update(b, a)
        return r + self.discount_factor * self.rollout(new_b, depth - 1)
        
    def action_prog_widen(self, b): 
    
        if b not in self.n: 
            self.n[b] = 0 
        if b not in self.c:
            self.c[b] = [] 
        if len(self.c[b]) <= self.n[b] ** self.alpha:
            a = np.random.uniform(0, 1) 
            self.c[b].append(a) 
            return a 
        else: 
            # print(self.q.keys())
            # print("-------", self.n[b], self.c[b])
            values = [self.q[b + (c, )] + self.const * 
                      np.sqrt(np.log(self.n[b]) / self.n[b + (c, )]) 
                      for c in self.c[b]] 
            for c in self.c[b]: 
                # print("DEBUG: ", b, c, self.n[b + (c, )])
                if (self.n[b + (c, )] == 0): 
                    print("?????", b, c)
                    exit(0) 
            return self.c[b][np.argmax(values)] 

    def execute_planning_loop(self, initial_state, max_steps=50, planning_time=1.0, success_threshold=0.05):
        """
        Execute the complete planning and execution loop using real physics simulation.
        
        Args:
            initial_state: Initial state of the system (x, y, theta, phi)
            max_steps: Maximum number of planning steps
            planning_time: Time allocated for planning at each step (seconds)
            success_threshold: Distance threshold to consider goal reached
            
        Returns:
            list: History of states and actions
        """
        current_state = np.array(initial_state)
        history = {'states': [current_state.copy()], 'actions': []}
        belief = (tuple([1 / self.num_particles for _ in range(self.num_particles)]), (0, 0, 0, 0)) 
        
        for step in range(max_steps):
            # Check if goal is reached
            dist_to_goal = np.linalg.norm(current_state[:2] - self.goal_loc[:2])
            if dist_to_goal < success_threshold:
                # print(f"Goal reached at step {step} with distance {dist_to_goal:.3f}")
                break
                
            # Create belief state from current state
            # Plan next action
            # print("BELIEF", belief)
            action = self.plan(belief, total_time=planning_time)
            
            # Execute action using real physics simulation
            # print(np.array(self.true_com), current_state, action)
            # print(current_state) 
            next_state = real_simulate(np.concatenate([np.array(self.true_com[0:2]), current_state], axis=0), action, self.dataset)
            # Transform the next state to match the dataset's scaling
            next_state = np.array(next_state)
            print("Translation: ", next_state)
            
            # Transform the position and angle components
            transformed_position = self.transform_coordinates(next_state[:3], 'final_position')
            transformed_angle = self.transform_coordinates(next_state[3], 'angle')
            print("Transformed position: ", transformed_position)
            print("Transformed angle: ", transformed_angle)
            
            transformed_position = transformed_position + current_state[0:3] 
            # Combine into a transformed state
            transformed_next_state = np.concatenate([transformed_position, transformed_angle])
            print("TRANSFORMED NEXT STATE", transformed_next_state) 
            
            # Calculate the difference between transformed next state and current state
            difference = np.concatenate([transformed_next_state[:3] - current_state[:3], [transformed_next_state[3] - current_state[3]]])
            
            # Update current state to the transformed next state
            current_state = transformed_next_state
            
            # Store history
            history['states'].append(current_state.copy())
            history['actions'].append(action)
            print("DIFFERENCE: ", difference) 
            
            # print(f"Step {step}: Action={action:.2f}, Distance to goal={dist_to_goal:.3f}")
            # Gotta update belief now. 
            # print("Update: ", belief, action, difference) 
            belief = self.update(belief, action, difference)[0] 
            # print("Belief after: ", belief) 
            # print("Belief after: ", belief)

            
        return history


    def transform_coordinates(self, coordinates, field_name):
        """
        Transform coordinates using the same scaling that the dataset applies.
        
        Args:
            coordinates: The coordinates to transform
            field_name: The name of the field in the dataset (e.g., 'final_position', 'angle')
            
        Returns:
            The transformed coordinates
        """
        if field_name in self.dataset.minmax_scaled_fields:
            # Apply MinMaxScaler transformation
            min_vals = self.dataset.scalers.get(f"{field_name}_min")
            max_vals = self.dataset.scalers.get(f"{field_name}_max")
            
            if min_vals is None or max_vals is None:
                print(f"Warning: Scaler parameters for {field_name} not found.")
                return coordinates
                
            # Apply the transformation: (x - min) / (max - min)
            transformed = (coordinates - min_vals) / (max_vals - min_vals)
            return transformed
        elif field_name in self.dataset.standard_scaled_fields:
            # Apply StandardScaler transformation
            mean = self.dataset.scalers.get(f"{field_name}_mean")
            scale = self.dataset.scalers.get(f"{field_name}_scale")
            
            if mean is None or scale is None:
                print(f"Warning: Scaler parameters for {field_name} not found.")
                return coordinates
                
            # Apply the transformation: (x - mean) / scale
            transformed = (coordinates - mean) / scale
            return transformed
        else:
            # No transformation for this field
            return coordinates


def real_simulate(state, action, dataset): 
    inverse_dict = {
        "com": np.concatenate([state[0:2], [0,]]),
        "angle": action 
    }
    # print(dataset.standard_scaled_fields)
    inversed = dataset.inverse_transform(inverse_dict)
    com = inversed["com"] 
    angle = inversed["angle"]
    # angle = angle * math.pi / 180
    body = GraspableBody("Primitive::Box_Test", com, 0.2, 0.07) 
    sim_client = GraspSimulationClient(body, False) 
    urdf = sim_client._get_object_urdf(body) 
    sim_client.disconnect() 

    # print("REAL SIMULATE: ", urdf, angle, state[5])
    transformation, contact_points, initial, _ = run_sim(urdf, angle, state[5], 0.1, 0, gui=False)  
    # print(action, state, transformation) 
    translation = transformation[:3, 3]
    # translated_location = translation + state[2:5] 
    rotation = R.from_matrix(transformation[:3, :3]).as_euler('xyz', degrees=False) 
    # print("SIM RESULTS", translation, rotation[-1])
    return np.concatenate([translation, np.array([rotation[-1]])], axis=0)

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

    pftdpw = PFTDPW(model, validation_dataset, [0.5, 0.5, 0], num_particles=100, alpha=0.5, const=0.1, search_depth=3, goal_loc=[100, 100, 0.0, 0.0])
    example_plan = pftdpw.execute_planning_loop((0.0, 0.0, 0.0, 0.0))
    # example_plan = pftdpw.plan(None, total_time=10.0)
    # print(example_plan)
    # 

    # print(example_plan) 



if __name__ == "__main__": 
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