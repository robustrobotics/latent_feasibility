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
from learning.domains.pushing.virtual_tables import BoxTable

from scipy.spatial.transform import Rotation as R 
from scipy.special import logsumexp 

from tqdm import tqdm



class PFTDPW:
    def __init__(self, model, dataset, true_com, search_depth=3, num_particles=20, goal_loc=(100, 100, 0, 0), discount_factor=0.8, alpha=0.6, beta=0.5, const=2.0, args=None): 
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
        self.args = args  # Store args as an instance variable

        self.particles = np.random.rand(num_particles, 2).tolist()  # Assuming 2D particles 
        
        # Initialize visualization table as None, will be created when needed
        self.vis_table = None

    def plan(self, b=None, total_time=10.0):
        if b is None: 
            b = (tuple([1 / self.num_particles for _ in range(self.num_particles)]), (0, 0, 0, 0)) 
        
        # Clear maps for a fresh planning iteration
        self.c = {}  # Clear child map
        self.q = {}  # Clear value map
        self.n = {}  # Clear visit count map
        
        # Add counter for total searches
        search_count = 0
        
        start_time = time.time()
        while time.time() - start_time < total_time:
            self.simulate(b, self.search_depth)
            search_count += 1
            
        best_value = float("-inf")
        best_action = 0 
        for action in self.c[b]:
            value = self.q[b + (action,)] 
            if value > best_value: 
                best_value = value
                best_action = action
            # print(f"action: {action}, value: {value}, n: {self.n[b + (action,)]}")
        # print(self.n.values())
        
        print(f"Total searches performed: {search_count}")
        
        return best_action 

    
    def do_singular_action_old(self, state, action, angle): 
        with torch.no_grad(): 
            # Validate inputs
            if np.isnan(angle) or np.isinf(angle):
                angle = 0.0
            if np.isnan(action) or np.isinf(action):
                action = 0.0

            # Prepare inputs for the model
            if not isinstance(action, tuple):
                action = (action,) 
            state_tensor = torch.tensor(state).cuda().float()
            action_tensor = torch.from_numpy(np.array([action])).float().cuda()

            body_params = torch.cat((
                state_tensor[0:2],
                torch.zeros(1).cuda(),
                torch.tensor([0.2, 0.07]).cuda()
            )).cuda().float()
            
            # Ensure all tensors are 1D before concatenation
            # Fix tensor construction warning by using clone().detach()
            sim_params = torch.cat([
                action_tensor[0].clone().detach().cuda(),  # Flatten to 1D
                torch.tensor([0.1]).cuda(),
                torch.tensor([angle]).cuda()
            ]).cuda().float()
            
            # Shape correctly for the model: [B, N, D] where B=batch size, N=num samples, D=dimension
            body_params = body_params.unsqueeze(0).unsqueeze(0)  # [1, 1, 5]
            sim_params = sim_params.unsqueeze(0).unsqueeze(0)    # [1, 1, 3]

            # Add validation checks
            if torch.isnan(body_params).any() or torch.isinf(body_params).any():
                return np.zeros(4), None
            if torch.isnan(sim_params).any() or torch.isinf(sim_params).any():
                return np.zeros(4), None

            distributions = self.model(sim_params, None, None, None, body_params)[0]
            sample = distributions[0][0].sample()
            translation = sample[:3].cpu().numpy()  # Convert to numpy immediately
            rotation = sample[3].cpu().numpy()

            return np.concatenate([translation, [rotation]]), distributions[0][0]
        
    def do_batch_actions(self, batch_particles, batch_actions, batch_angles):
        """
        Performs a batch of actions on a batch of particles.

        Args:
            batch_particles (list): List of particles to update.
            batch_actions (list): List of actions to perform.
            batch_angles (list): List of angles for each action.

        Returns:
            tuple: (batch_results, batch_distributions)
                - batch_results: List of results for each action
                - batch_distributions: The batched distribution from the model
        """
        with torch.no_grad():
            # Convert inputs to tensors
            batch_particles = torch.tensor(batch_particles).float().cuda()
            batch_actions = torch.tensor(batch_actions).float().cuda()
            batch_angles = torch.tensor(batch_angles).float().cuda()
            # print(batch_particles.shape, batch_actions.shape, batch_angles.shape)
            if batch_actions.ndim == 1:
                batch_actions = batch_actions.unsqueeze(1)

            # Prepare input data for the model
            # Create state tensors
            state_tensors = batch_particles  # shape: [batch_size, 2]

            # Extract needed COM from states (first 2 components)
            com_tensors = state_tensors[:, 0:2]  # shape: [batch_size, 2]

            # Prepare body parameters - shape [batch_size, 5]
            body_params = torch.cat([
                com_tensors,                                          # COM x,y [batch_size, 2]
                torch.zeros(len(batch_particles), 1).cuda(),         # COM z [batch_size, 1]
                torch.ones(len(batch_particles), 1).cuda() * 0.2,    # length [batch_size, 1]
                torch.ones(len(batch_particles), 1).cuda() * 0.07    # width [batch_size, 1]
            ], dim=1).float()

            # Prepare action parameters
            action_values = batch_actions
            angle_values = batch_angles.unsqueeze(1)
            # print(action_values.shape, angle_values.shape)

            # Create simulation parameters - shape [batch_size, 3]
            sim_params = torch.cat([
                action_values,                           # Action [batch_size, 1]
                torch.ones(len(batch_particles), 1).cuda() * 0.1,  # Constant [batch_size, 1]
                angle_values                             # Angle [batch_size, 1]
            ], dim=1).float()

            # Reshape tensors to match model's expected input format
            # Model expects: target_x shape [batch_size, num_samples, feature_dim]
            # body_params = body_params.unsqueeze(1)   # [batch_size, 1, 5]
            sim_params = sim_params.unsqueeze(1)     # [batch_size, 1, 3]
            # print(body_params.shape, sim_params.shape)

            # Forward pass through model
            distributions, means, covs = self.model(sim_params, None, None, None, body_params)

            # Process results
            batch_results = []
            dists = []
            for i in range(len(batch_actions)):
                mean = means[i][0]
                cov = covs[i][0]
                # print(mean.shape, cov.shape)
                dist = torch.distributions.MultivariateNormal(mean, cov)
                sample = dist.sample()
                dists.append(dist)
                translation = sample[:3].cpu().numpy()
                rotation = sample[3].cpu().numpy()
                batch_results.append(np.concatenate([translation, [rotation]]))
            
            return batch_results, dists

    def update_old(self, b, a, obs=None): 
        probs = b[0] 
        observation = b[1] 
        total = np.zeros(4)
        
        # Only process particles with significant probabilities for prediction
        active_mask = np.array(probs) >= 1e-9
        if np.sum(active_mask) == 0:
            max_idx = np.argmax(probs)
            active_mask[max_idx] = True

        for i in range(self.num_particles):
            if active_mask[i]:
                com = self.particles[i] 
                result = np.array(self.do_singular_action(com, a, self.transform_coordinates(observation[3], 'output_angle')[0])[0])
                total += result * probs[i]

        debug = False
        if (obs is not None):
            total = obs
            debug = True

        # Prevent division by zero in log computation
        probs = np.clip(probs, 1e-300, 1.0)  # Clip to small positive values
        log_probs = np.log(probs)
        new_probs = np.array(log_probs)
        
        # Convert to real-world coordinates before adding
        # First convert the current observation to real-world coordinates
        real_obs_x = observation[0]
        real_obs_y = observation[1]
        real_obs_z = observation[2]
        real_obs_angle = observation[3]
        
        # Convert the total (action effect) to real-world coordinates
        real_total_x = self.inverse_transform_coordinates(total[0], 'final_position_x')
        real_total_y = self.inverse_transform_coordinates(total[1], 'final_position_y')
        real_total_z = self.inverse_transform_coordinates(total[2], 'final_position_z')
        real_total_angle = self.inverse_transform_coordinates(total[3], 'output_angle')
        
        # Add x and y in real-world coordinates
        real_new_x = real_obs_x + real_total_x
        real_new_y = real_obs_y + real_total_y
        
        # For z and angle, use the new values directly instead of adding
        real_new_z = real_total_z
        real_new_angle = real_total_angle
        
        # Convert back to model's normalized space
        new_x = self.transform_coordinates(real_new_x, 'final_position_x')
        new_y = self.transform_coordinates(real_new_y, 'final_position_y')
        new_z = self.transform_coordinates(real_new_z, 'final_position_z')
        new_angle = self.transform_coordinates(real_new_angle, 'output_angle')
        
        # Create the new observation in the model's normalized space
        new_observation = np.array([new_x, new_y, new_z, new_angle.item()])
        
        # Update all particles' probabilities using distributions
        for i in range(self.num_particles): 
            com = self.particles[i] 
            distribution = self.do_singular_action(com, a, observation[3])[1] 
            
            # Convert observation to tensor and validate/normalize
            obs_tensor = torch.from_numpy(total).cuda().float()
            
            # Get distribution parameters
            mean = distribution.mean
            std = distribution.stddev
            
            log_prob = distribution.log_prob(obs_tensor).sum().item()
            
            new_probs[i] = log_prob + np.log(probs[i])

        # Normalize probabilities to ensure they sum to 1
        new_probs = np.exp(new_probs - logsumexp(new_probs))

        # Convert to tuple for hashability
        new_probs = tuple(new_probs)
        
        # Use the real-world goal location directly (no need to transform)
        real_goal = np.array([self.goal_loc[0], self.goal_loc[1]])
        
        # Calculate reward as negative L2 distance in real-world coordinates
        reward = -np.linalg.norm(np.array([real_new_x, real_new_y]) - real_goal)
        
        # Return both model space observation and corresponding real-world coordinates
        # This allows downstream methods to use whichever representation they need
        return (new_probs, (real_new_x, real_new_y, real_new_z, real_new_angle.item())), reward

    def update(self, b, a, obs=None):
        """
        Optimized version of update that processes multiple particles in parallel to improve speed.
        Uses true batched operations for significant performance improvements.
        
        Args:
            b: Belief state, tuple of (particle probabilities, observation)
            a: Action to take
            obs: Optional observation override
            
        Returns:
            Updated belief state and reward
        """
        probs = np.array(b[0])
        observation = b[1]
        
        # Only process particles with significant probabilities for prediction
        active_mask = probs >= 1e-9
        if np.sum(active_mask) == 0:
            max_idx = np.argmax(probs)
            active_mask[max_idx] = True
        
        # Get transformed angle once for all particles
        transformed_angle = self.transform_coordinates(observation[3], 'output_angle')[0]
        
        # Extract active particles and their probabilities
        active_indices = np.where(active_mask)[0]
        active_particles = [self.particles[i] for i in active_indices]
        
        # Prepare batch actions and angles
        batch_actions = [a] * len(active_indices)
        batch_angles = [transformed_angle] * len(active_indices)
        
        # Run batch inference using real batched operations
        batch_results, batch_distributions = self.do_batch_actions(
            active_particles, 
            batch_actions, 
            batch_angles
        )
        
        # Handle provided observation
        if obs is not None:
            total = obs
        else:
            # Compute weighted sum of results
            total = np.zeros(4)
            for i, idx in enumerate(active_indices):
                total += batch_results[i] * probs[idx]
        
        # Prevent division by zero in log computation
        probs = np.clip(probs, 1e-300, 1.0)
        log_probs = np.log(probs)
        new_probs = np.array(log_probs)
        
        # Real-world coordinate conversion
        real_obs_x = observation[0]
        real_obs_y = observation[1]
        real_obs_z = observation[2]
        real_obs_angle = observation[3]
        
        # Convert the total (action effect) to real-world coordinates
        real_total_x = self.inverse_transform_coordinates(total[0], 'final_position_x')
        real_total_y = self.inverse_transform_coordinates(total[1], 'final_position_y')
        real_total_z = self.inverse_transform_coordinates(total[2], 'final_position_z')
        real_total_angle = self.inverse_transform_coordinates(total[3], 'output_angle')
        
        # Calculate new state in real-world coordinates
        real_new_x = real_obs_x + real_total_x
        real_new_y = real_obs_y + real_total_y
        real_new_z = real_total_z
        real_new_angle = real_total_angle
        
        # Convert back to model's normalized space
        new_x = self.transform_coordinates(real_new_x, 'final_position_x')
        new_y = self.transform_coordinates(real_new_y, 'final_position_y')
        new_z = self.transform_coordinates(real_new_z, 'final_position_z')
        new_angle = self.transform_coordinates(real_new_angle, 'output_angle')
        
        # Create the new observation in the model's normalized space
        new_observation = np.array([new_x, new_y, new_z, new_angle.item()])
        
        # Update all particles' probabilities using distributions in batch
        # Convert observation to tensor once
        obs_tensor = torch.from_numpy(total).cuda().float()
        
        # Calculate log probabilities for all particles with batch operations
        for i, idx in enumerate(active_indices):
            distribution = batch_distributions[i]
            log_prob = distribution.log_prob(obs_tensor).sum().item()
            new_probs[idx] = log_prob + log_probs[idx]
            
        # For inactive particles, set probability to very low value
        inactive_indices = np.where(~active_mask)[0]
        if len(inactive_indices) > 0:
            new_probs[inactive_indices] = -1e10  # Very low log probability
            
        # Normalize probabilities to ensure they sum to 1
        new_probs = np.exp(new_probs - logsumexp(new_probs))
        
        # Convert to tuple for hashability
        new_probs = tuple(new_probs)
        
        # Calculate reward as negative L2 distance to goal
        real_goal = np.array([self.goal_loc[0], self.goal_loc[1]])
        reward = -np.linalg.norm(np.array([real_new_x, real_new_y]) - real_goal)
        # print(real_new_x, real_new_y, real_goal)
        # print(real_new_x, real_new_y)
        
        return (new_probs, (real_new_x, real_new_y, real_new_z, real_new_angle.item())), reward


    def simulate(self, b, depth): 
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
            total = r + self.discount_factor * self.rollout(new_b, depth - 1) 
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

        a = np.random.uniform(0, 1)

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
            values = [self.q[b + (c, )] + self.const * 
                     np.sqrt(np.log(self.n[b]) / self.n[b + (c, )]) 
                     for c in self.c[b]] 
            for c in self.c[b]: 
                if (self.n[b + (c, )] == 0): 
                    exit(0) 
            return self.c[b][np.argmax(values)] 

    def execute_planning_loop(self, initial_state=None, max_steps=50, planning_time=1.0, success_threshold=0.05, visualize=False, show_visualization=False, save_visualization_path=None):
        """
        Execute the complete planning and execution loop using real physics simulation.
        
        Args:
            initial_state: Initial state of the system (x, y, theta, phi). If None, use (0,0,0,0)
            max_steps: Maximum number of planning steps
            planning_time: Time allocated for planning at each step (seconds)
            success_threshold: Distance threshold to consider goal reached
            visualize: Whether to visualize the planning results
            show_visualization: Whether to show the visualization interactively
            save_visualization_path: Path to save the visualization image (if None, uses timestamp)
            
        Returns:
            list: History of states and actions
        """
        # Ensure starting pose is (0,0,0,0) if not specified
        if initial_state is None:
            initial_state = (0.0, 0.0, 0.0, 0.0)
            
        # Initialize states in both real-world and transformed coordinates
        current_state_real = np.array(initial_state)
        current_state_transformed = np.array(initial_state)
        
        # Print initial position
        print(f"Initial position: ({current_state_real[0]:.4f}, {current_state_real[1]:.4f}), orientation: {current_state_real[3]:.4f}")
        
        # Keep track of history in real-world coordinates for visualization
        history = {'states': [current_state_real.copy()], 'actions': [], 'transformations': []}
        
        # Initialize belief state
        belief = (tuple([1 / self.num_particles for _ in range(self.num_particles)]), (initial_state[0], initial_state[1],initial_state[2], initial_state[3])) 
        
        # Initialize visualization table if needed and not already initialized
        if visualize and self.vis_table is None:
            self.initialize_visualization_table()
        
        for step in range(max_steps):
            # Check if goal is reached using real-world coordinates
            dist_to_goal = np.linalg.norm(current_state_real[:2] - self.goal_loc[:2])
            if dist_to_goal < success_threshold:
                print(f"Goal reached at step {step}! Final position: ({current_state_real[0]:.4f}, {current_state_real[1]:.4f})")
                print(f"Distance to goal: {dist_to_goal:.4f}")
                break
                
            # Plan next action using the belief state
            action = self.plan(belief, total_time=planning_time)
            
            # Convert action to radians using inverse_transform, then to degrees for display
            inverse_dict = {"angle": action}
            inversed = self.dataset.inverse_transform(inverse_dict)
            angle_radians = inversed["angle"]
            
            # Extract scalar value from the array for printing and storage
            angle_degrees_scalar = float(np.degrees(angle_radians))
            
            # Execute action using real physics simulation to get real-world next state
            # The state parameter for real_simulate combines the true COM and the current state
            sim_result = real_simulate(
                np.concatenate([np.array(self.true_com[0:2]), current_state_real], axis=0), 
                action, 
                self.dataset
            )
            sim_result = np.array(sim_result)
            
            # Convert from world coordinates to object-local coordinates for visualization
            current_ori = current_state_real[3]
            world_to_object_rotation = np.array([
                [np.cos(current_ori), np.sin(current_ori)],
                [-np.sin(current_ori), np.cos(current_ori)]
            ])
            
            # Convert world-frame translation to object-frame translation
            translation_local = world_to_object_rotation @ sim_result[:2]
            
            # Create the adjusted transformation with object-local coordinates
            adjusted_sim_result = np.zeros_like(sim_result)
            adjusted_sim_result[:2] = translation_local
            adjusted_sim_result[2] = sim_result[2]  # Z-translation unchanged
            adjusted_sim_result[3] = sim_result[3] - current_ori  # Relative rotation
            
            # Store the local-frame transformation for visualization
            history['transformations'].append(adjusted_sim_result.copy())

            # Update the current state using the world-frame results
            next_state_real = np.array([
                current_state_real[0] + sim_result[0],  # Add world x translation
                current_state_real[1] + sim_result[1],  # Add world y translation
                current_state_real[2] + sim_result[2],  # Add z translation
                sim_result[3]                          # Use the new absolute rotation
            ])
            
            # Print the push action and resulting position
            print(f"Step {step+1}: Push at {angle_degrees_scalar:.2f}° → Position: ({next_state_real[0]:.4f}, {next_state_real[1]:.4f}), orientation: {next_state_real[3]:.4f}")
            
            # Transform the simulation result for the model
            transformed_position = self.transform_coordinates(sim_result[:3], 'final_position')
            transformed_angle = self.transform_coordinates(sim_result[3], 'output_angle')
            
            # Ensure both arrays are 1D before concatenation
            position_part = current_state_transformed[:3] + transformed_position
            angle_part = np.atleast_1d(transformed_angle).flatten()  # Ensure it's a 1D array
            
            # Calculate the next transformed state
            next_state_transformed = np.concatenate([position_part, angle_part])
            
            # Calculate the difference between transformed states for the belief update
            pos_diff = transformed_position
            angle_diff = np.atleast_1d(transformed_angle - current_state_transformed[3]).flatten()
            
            difference = np.concatenate([pos_diff, angle_diff])
            
            # Update the current states
            current_state_real = next_state_real
            current_state_transformed = next_state_transformed
            
            # Store history in real-world coordinates
            history['states'].append(current_state_real.copy())
            history['actions'].append(angle_degrees_scalar)
            
            # Update belief state using the transformed difference
            updated_state, _ = self.update(belief, action, difference)
            # Just use the first two components for the belief update
            belief = updated_state

            # print(f"belief: {belief[0]}")
        
        # If we didn't reach the goal, print the final position
        if np.linalg.norm(current_state_real[:2] - self.goal_loc[:2]) >= success_threshold:
            print(f"Planning ended. Final position: ({current_state_real[0]:.4f}, {current_state_real[1]:.4f})")
            print(f"Distance to goal: {np.linalg.norm(current_state_real[:2] - self.goal_loc[:2]):.4f}")
            
        # Automatically visualize the planning results if requested
        if visualize:
            # Visualize the planning history using real-world coordinates
            self.visualize_planning(
                history=history,
                show=show_visualization,
                save_path=save_visualization_path)
        
        return history


    def transform_coordinates(self, coordinates, field_name):
        """
        Transform coordinates using the same scaling that the dataset applies.
        
        Args:
            coordinates: The coordinates to transform
            field_name: The name of the field in the dataset (e.g., 'final_position', 'angle')
                        Can also be individual components like 'final_position_x'
            
        Returns:
            The transformed coordinates
        """
        if field_name == 'output_angle': 
            return np.array([coordinates,] )
        # Handle individual components (e.g., 'final_position_x')
        if field_name.endswith('_x') or field_name.endswith('_y') or field_name.endswith('_z'):
            # Extract base field name (e.g., 'final_position')
            base_field = field_name.rsplit('_', 1)[0]
            # Extract component index
            component = field_name.rsplit('_', 1)[1]
            component_idx = {'x': 0, 'y': 1, 'z': 2}.get(component, 0)
            
            # Check if we have a single value or array
            if np.isscalar(coordinates):
                # For individual components, create a 3D array with the value at the right position
                coords_array = np.zeros(3)
                coords_array[component_idx] = coordinates
                
                # Transform the full array
                if base_field in self.dataset.minmax_scaled_fields:
                    # Apply MinMaxScaler transformation
                    min_vals = self.dataset.scalers.get(f"{base_field}_min")
                    max_vals = self.dataset.scalers.get(f"{base_field}_max")
                    
                    if min_vals is None or max_vals is None:
                        return coordinates
                        
                    # Apply the transformation: (x - min) / (max - min)
                    transformed = (coords_array - min_vals) / (max_vals - min_vals)
                    return transformed[component_idx]
                elif base_field in self.dataset.standard_scaled_fields:
                    # Apply StandardScaler transformation
                    mean = self.dataset.scalers.get(f"{base_field}_mean")
                    scale = self.dataset.scalers.get(f"{base_field}_scale")
                    
                    if mean is None or scale is None:
                        return coordinates
                        
                    # Apply the transformation: (x - mean) / scale
                    transformed = (coords_array - mean) / scale
                    return transformed[component_idx]
                else:
                    # If we already have an array, extract the component after transformation
                    transformed = self.transform_coordinates(coordinates, base_field)
                    if len(transformed.shape) > 0 and transformed.shape[0] > component_idx:
                        return transformed[component_idx]
                return transformed
                
        # Original logic for full fields
        if field_name in self.dataset.minmax_scaled_fields:
            # Apply MinMaxScaler transformation
            min_vals = self.dataset.scalers.get(f"{field_name}_min")
            max_vals = self.dataset.scalers.get(f"{field_name}_max")
            
            if min_vals is None or max_vals is None:
                return coordinates
                
            # Apply the transformation: (x - min) / (max - min)
            transformed = (coordinates - min_vals) / (max_vals - min_vals)
            return transformed
        elif field_name in self.dataset.standard_scaled_fields:
            # Apply StandardScaler transformation
            mean = self.dataset.scalers.get(f"{field_name}_mean")
            scale = self.dataset.scalers.get(f"{field_name}_scale")
            
            if mean is None or scale is None:
                return coordinates
                
            # Apply the transformation: (x - mean) / scale
            transformed = (coordinates - mean) / scale
            return transformed
        else:
            # No transformation for this field
            return coordinates
            
    def inverse_transform_coordinates(self, coordinates, field_name):
        """
        Inverse transform coordinates from the model's normalized space back to real-world coordinates.
        
        Args:
            coordinates: The normalized coordinates to inverse transform
            field_name: The name of the field in the dataset (e.g., 'final_position', 'angle')
                        Can also be individual components like 'final_position_x'
            
        Returns:
            The real-world coordinates
        """
        # Handle individual components (e.g., 'final_position_x')
        if field_name == 'output_angle': 
            return coordinates
        if field_name.endswith('_x') or field_name.endswith('_y') or field_name.endswith('_z'):
            # Extract base field name (e.g., 'final_position')
            base_field = field_name.rsplit('_', 1)[0]
            # Extract component index
            component = field_name.rsplit('_', 1)[1]
            component_idx = {'x': 0, 'y': 1, 'z': 2}.get(component, 0)
            
            # Check if we have a single value or array
            if np.isscalar(coordinates):
                # For individual components, create a 3D array with the value at the right position
                coords_array = np.zeros(3)
                coords_array[component_idx] = coordinates
                
                # Transform the full array
                if base_field in self.dataset.minmax_scaled_fields:
                    # Apply inverse MinMaxScaler transformation
                    min_vals = self.dataset.scalers.get(f"{base_field}_min")
                    max_vals = self.dataset.scalers.get(f"{base_field}_max")
                    
                    if min_vals is None or max_vals is None:
                        return coordinates
                        
                    # Apply the inverse transformation: x * (max - min) + min
                    real_world = coords_array * (max_vals - min_vals) + min_vals
                    return real_world[component_idx]
                elif base_field in self.dataset.standard_scaled_fields:
                    # Apply inverse StandardScaler transformation
                    mean = self.dataset.scalers.get(f"{base_field}_mean")
                    scale = self.dataset.scalers.get(f"{base_field}_scale")
                    
                    if mean is None or scale is None:
                        return coordinates
                        
                    # Apply the inverse transformation: x * scale + mean
                    real_world = coords_array * scale + mean
                    return real_world[component_idx]
                else:
                    # No transformation for this field
                    return coordinates
            else:
                # If we already have an array, extract the component after transformation
                real_world = self.inverse_transform_coordinates(coordinates, base_field)
                if len(real_world.shape) > 0 and real_world.shape[0] > component_idx:
                    return real_world[component_idx]
                return real_world
                
        # Original logic for full fields
        if field_name in self.dataset.minmax_scaled_fields:
            # Apply inverse MinMaxScaler transformation
            min_vals = self.dataset.scalers.get(f"{field_name}_min")
            max_vals = self.dataset.scalers.get(f"{field_name}_max")
            
            if min_vals is None or max_vals is None:
                return coordinates
                
            # Apply the inverse transformation: x * (max - min) + min
            real_world = coordinates * (max_vals - min_vals) + min_vals
            return real_world
        elif field_name in self.dataset.standard_scaled_fields:
            # Apply inverse StandardScaler transformation
            mean = self.dataset.scalers.get(f"{field_name}_mean")
            scale = self.dataset.scalers.get(f"{field_name}_scale")
            
            if mean is None or scale is None:
                return coordinates
                
            # Apply the inverse transformation: x * scale + mean
            real_world = coordinates * scale + mean
            return real_world
        else:
            # No transformation for this field
            return coordinates
            
    def initialize_visualization_table(self, block_width=0.2, block_length=0.2, table_length=10.0, table_width=10.0):
        """
        Initialize a BoxTable for visualization purposes.
        
        Args:
            block_width: Width of the block
            block_length: Length of the block
            table_length: Length of the table
            table_width: Width of the table
            
        Returns:
            The initialized BoxTable instance
        """
        # Use goal location to determine table size, making the table just large enough to contain the goal
        # Add some margin to ensure the goal is well within the table
        if hasattr(self, 'goal_loc') and self.goal_loc is not None:
            # Make sure the table is at least as large as the goal location plus some margin
            table_length = max(self.goal_loc[0] * 1.5, 5.0)  
            table_width = max(self.goal_loc[1] * 1.5, 5.0)
        
        # Create a BoxTable with the updated constructor
        goal_x = self.goal_loc[0] if hasattr(self, 'goal_loc') and self.goal_loc is not None else 3.0
        goal_y = self.goal_loc[1] if hasattr(self, 'goal_loc') and self.goal_loc is not None else 3.0
        
        self.vis_table = BoxTable(
            block_width=block_width,
            block_length=block_length,
            block_com_relative_to_centroid=np.array([0, 0]),
            goal_loc_x=goal_x,
            goal_loc_y=goal_y,
            table_length=table_length,
            table_width=table_width
        )
        
        return self.vis_table
    
    def visualize_planning(self, history=None, show=False, save_path=None, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, res=0.01):
        """
        Visualize the planning and execution process using the BoxTable visualization.
        
        Args:
            history: Dictionary containing 'states', 'actions', and 'transformations' from execute_planning_loop
            show: Whether to show the plot interactively (default is False)
            save_path: Path to save the visualization image (if None, uses timestamp)
            min_x, max_x, min_y, max_y: Bounds for the visualization
            res: Resolution for the table visualization grid
            
        Returns:
            None
        """
        # Initialize the visualization table if not already done
        if self.vis_table is None:
            self.initialize_visualization_table()
            
        # Reset the table to clear any previous trajectories
        self.vis_table.reset()
        
        # Set visualization boundaries based on table size
        if hasattr(self.vis_table, 'table_length') and hasattr(self.vis_table, 'table_width'):
            max_x = self.vis_table.table_length
            max_y = self.vis_table.table_width
        
        if history is not None and 'transformations' in history and len(history['transformations']) > 0:
            # Apply each transformation as a separate trajectory
            for transform in history['transformations']:
                # The transformations are already in object-local coordinates
                dx, dy, dz, dtheta = transform
                
                # Create a transformation matrix in SE(2) format
                pose = np.eye(3)
                
                # Apply the translation (in object-local frame)
                pose[0, 2] = dx
                pose[1, 2] = dy
                
                # Apply the rotation (relative to current orientation)
                pose[0, 0] = np.cos(dtheta)
                pose[0, 1] = -np.sin(dtheta)
                pose[1, 0] = np.sin(dtheta)
                pose[1, 1] = np.cos(dtheta)
                
                # Apply the transformation to the visualization
                self.vis_table.apply_push_trajectory([pose])
            
        # Visualize the table
        if save_path is None:
            # Generate a unique filename with timestamp if none provided
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = f'planning_visualization_{timestamp}.png'
            
        # Create visualization and save it
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        
        # Create visualization
        self.vis_table.visualize(show=show, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, res=res)
        
        # Save the figure
        plt.savefig(save_path)
        if not show:
            plt.close()
                
        return self.vis_table


def real_simulate(state, action, dataset): 
    inverse_dict = {
        "com": np.concatenate([state[0:2], [0,]]),
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
    rotation = R.from_matrix(transformation[:3, :3]).as_euler('xyz', degrees=False) 
    return np.concatenate([translation, np.array([(rotation[-1] + 2 * np.pi) % (2 * np.pi)])], axis=0)

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

    # Create PFTDPW instance with appropriate parameters
    pftdpw = PFTDPW(model, validation_dataset, [0.5, 0.5, 0.3], num_particles=20, alpha=0.5, const=0.1, search_depth=3, goal_loc=[10, 10,  0.0, 0.0], args=args)
    
    # Determine visualization settings from command-line arguments
    visualize = not hasattr(args, 'no_visualization') or not args.no_visualization
    show_visualization = False  # Always set to False to never show interactively
    
    # Determine output directory for visualizations
    vis_output_dir = instance_path
    if hasattr(args, 'vis_output_dir') and args.vis_output_dir:
        vis_output_dir = args.vis_output_dir
        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)
    
    # Generate a timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Initialize the visualization table if visualization is enabled
    if visualize:
        # Initialize the visualization table using goal location to determine size
        pftdpw.initialize_visualization_table(
            block_width=0.07,
            block_length=0.2
        )
    
    # Set the path for saving the visualization
    save_visualization_path = None
    if visualize:
        save_visualization_path = os.path.join(vis_output_dir, f'planning_visualization_{timestamp}.png')
    
    # Execute planning with visualization settings
    example_plan = pftdpw.execute_planning_loop(
        initial_state=(1.0, 1.0, 0.0, 0.0),
        max_steps=args.max_steps,
        planning_time=1.0,
        success_threshold=0.05,
        visualize=visualize,
        show_visualization=show_visualization,
        save_visualization_path=save_visualization_path
    )
    
    # Additional visualization with custom parameters if requested
    if visualize and hasattr(args, 'additional_visualization') and args.additional_visualization:
        detailed_vis_path = os.path.join(vis_output_dir, f'detailed_visualization_{timestamp}.png')
        pftdpw.visualize_planning(
            history=example_plan,
            show=show_visualization,
            save_path=detailed_vis_path,
            res=0.005  # Higher resolution for detailed visualization
        )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--instance', type=str, required=True) 
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-points', type=int, default=20)  # Set to 20 as per args in main
    parser.add_argument('--batch-size', type=int, default=32) 
    parser.add_argument('--guess-obj', action='store_true')
    parser.add_argument('--no-visualization', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--vis-output-dir', type=str, default=None)
    parser.add_argument('--additional-visualization', action='store_true')
    parser.add_argument('--max-steps', type=int, default=20)

    args = parser.parse_args() 
    args.no_deterministic = True
    args.use_obj_prop = True
    args.guess_obj = False
    args.latent_samp = -1
    args.point_cloud = True
    args.no_contact = True 

    main(args)