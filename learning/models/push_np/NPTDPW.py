from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import pickle
import argparse
from scipy.spatial.transform import Rotation as R  # Required for real_simulate
from learning.models.push_np.attention_push_np import AttentionPushNP
from learning.models.push_np.dataset import PushNPDataset, collate_fn
from learning.domains.pushing.virtual_tables import BoxTable, RingTable, BeamTable
from learning.domains.pushing.find_contact_points import run_sim  # Required for real_simulate
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody  # Required for real_simulate
from learning.models.push_np.train_APNP import train  # Import the training function
from learning.models.push_np.PFTDPW import real_simulate  # Import real_simulate function for physics simulation

class NPTDPW:
    def __init__(self, model, dataset, true_com, search_depth=1, goal_loc=(10, 10, 0, 0), discount_factor=0.8, alpha=0.8, beta=0.5, const=5.0, args=None, table_type='box', success_threshold=0.2):
        self.c = {} # child
        self.q = {} # value
        self.n = {} # number of visits
        self.success_threshold = success_threshold  # Distance threshold for goal success
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.dataset = dataset
        self.search_depth = search_depth
        self.goal_loc = np.array(goal_loc)
        self.discount_factor = discount_factor
        self.alpha = alpha  # Hyperparameter for action selection
        self.beta = beta    # Hyperparameter for observation handling
        self.const = const
        self.true_com = true_com
        self.args = args
        
        # Initialize visualization table as None, will be created when needed
        # Set the table type for visualization ('box' or 'ring')
        self.table_type = table_type
        self.vis_table = self.initialize_visualization_table() 
        
        # Initialize statistics tracking
        self.stats = {
            # Planning statistics
            'planning_times': [],          # Time spent planning at each step
            'action_counts': {},           # Frequency of actions chosen
            'visits_per_step': [],         # Number of nodes visited in each planning step
            'depth_reached': [],           # Maximum depth reached in planning
            
            # Execution statistics
            'step_durations': [],          # Time taken for each step (planning + execution)
            'distances_to_goal': [],       # Distance to goal at each step
            'state_trajectory': [],        # Complete state trajectory
            'action_trajectory': [],       # Complete action trajectory
            'cumulative_translation': 0.0, # Total distance traveled
            'cumulative_rotation': 0.0,    # Total rotation performed
            
            # Success metrics
            'final_distance': None,        # Final distance to goal
            'success': False,              # Whether goal was reached
            'success_step': None,          # Step at which goal was reached
            'total_steps': 0,              # Total steps executed
            'total_planning_time': 0.0,    # Total time spent in planning
            'total_execution_time': 0.0,   # Total time spent in execution
            'fell_off_table': False,       # Whether block fell off table
            'total_reward': 0.0,           # Cumulative reward during planning
            
            # Prediction accuracy metrics
            'prediction_errors': [],       # Differences between predicted and actual outcomes
            'model_confidence': []         # Model's reported confidence in predictions
        }

    def plan(self, b=None, total_time=10.0):
        if b is None:
            # Initialize with empty history
            b = {
                'states': [],  # List of past states
                'actions': [], # List of past actions
                'outcomes': [] # List of outcomes (next states) from each action
            }
        
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
        b_key = self._belief_to_key(b)
        for action in self.c[b_key]:
            # Ensure action is hashable
            if isinstance(action, np.ndarray):
                action_hash = tuple(action.tolist())
            elif not isinstance(action, tuple):
                action_hash = (action,)
            else:
                action_hash = action
                
            # Create a hashable key for dictionary operations
            hash_key = b_key + action_hash
            value = self.q[hash_key]
            
            # Get mean of distribution for this action
            next_b, _ = self.update(b, action)
            next_state = next_b['states'][-1]
            # print(f"Action: {np.degrees(action)}, Value: {value}, n: {self.n[b_key + action_hash]}, Next: {next_state[:2]}")
            if value > best_value: 
                best_value = value
                best_action = action
            # print(best_value, best_action)
        # print(f"Best action: {np.degrees(best_action)}, Value: {best_value}") 
        # Total searches performed (silenced)
        
        return best_action

    def update(self, b, a):
        """
        Update belief state based on action and observation, handling coordinate transformations.
        
        Args:
            b: Current belief state
            a: Action to take (in real-world coordinates)
            obs: Optional observation override
            
        Returns:
            Updated belief state and reward in real-world coordinates
        """
        # print("WTF", a)
        # Convert current state to tensor
        if not isinstance(a, tuple):
            a = (a,)
        current_state_real = b['states'][-1] if b['states'] else np.ndarray([1, 1, 0, 0])
        # print("CURRENT STATE REAL", current_state_real)
        
        # Transform the current state to model's normalized space - collect each component separately to avoid mixed types
        pos_x = self.transform_coordinates(current_state_real[0], 'final_position_x')
        pos_y = self.transform_coordinates(current_state_real[1], 'final_position_y')
        pos_z = self.transform_coordinates(current_state_real[2], 'final_position_z')
        angle_transformed = self.transform_coordinates(current_state_real[3], 'output_angle')[0]  # Extract scalar from array
        
        # Store as separate components to avoid array with mixed types
        current_state_transformed = np.array([pos_x, pos_y, pos_z, angle_transformed])
        
        # Use the original action angle (no normalization for push angles)
        action_transformed = a[0]
        
        # Convert angle from [0, 2π] to [-π, π] range for better angle representation
        if current_state_transformed[3] > np.pi:
            current_state_transformed[3] = current_state_transformed[3] - 2 * np.pi
        
        # Prepare input tensors for the model (in normalized space)
        angle = torch.tensor([self.transform_coordinates(action_transformed, 'angle')]).float().unsqueeze(0)
        push_velocities = torch.zeros_like(angle)  # shape [1, 1, 1]
        initials = torch.tensor([[current_state_transformed[3].tolist()]]).float()  # shape [1, 1, 3]
        # print(a, angle)
        
        # Stack tensors for current action
        current_target_xs = torch.cat([
            angle,  # [1, 1, 1]
            push_velocities,  # [1, 1, 1]
            initials.unsqueeze(0)  # [1, 1, 1]
        ], dim=2)  # shape [1, 1, 3]
        
        # print(current_target_xs) 
        # Handle past actions and observations if they exist (all in normalized space)
        if b['actions'] and b['outcomes']:
            # Get past states and transform them to normalized space
            past_states_normalized = []
            for state in b['states'][:-1]:  # Skip the current state
                # Process each component separately to avoid mixed types
                angle = self.transform_coordinates(state[3], 'output_angle')
                # Only need position components for past states
                past_states_normalized.append([float(angle)])  
                
            # Get past outcomes and transform them to normalized space
            past_outcomes_normalized = []
            past_rotations_normalized = []
            for outcome in b['outcomes']:
                # Process each component separately
                x = self.transform_coordinates(outcome[0], 'final_position_x')
                y = self.transform_coordinates(outcome[1], 'final_position_y')
                z = self.transform_coordinates(outcome[2], 'final_position_z')
                angle = self.transform_coordinates(outcome[3], 'output_angle')
                # Extract scalar value from angle if it's an array
                if hasattr(angle, '__len__') and len(angle) == 1:
                    angle = angle[0]
                
                past_outcomes_normalized.append([float(x), float(y), float(z)])  # Position components
                past_rotations_normalized.append(float(angle))   # Rotation component
                
            # Transform past actions to normalized space and ensure they're scalar
            past_angles_normalized = []
            for act in b['actions']:
                angle = self.transform_coordinates(act[0], 'angle')
                # Extract scalar value if it's an array
                if hasattr(angle, '__len__') and len(angle) == 1:
                    angle = angle[0]
                past_angles_normalized.append(float(angle))
            
            # Convert to tensors with explicit float type
            past_angles = torch.tensor(past_angles_normalized).float().unsqueeze(0).unsqueeze(2)  # shape [1, N, 1]
            past_velocities = torch.zeros_like(past_angles)  # shape [1, N, 1]
            past_initials = torch.tensor(past_states_normalized).float().unsqueeze(0)
            past_finals = torch.tensor(past_outcomes_normalized).float().unsqueeze(0)  # shape [1, N, 3]
            past_rotations = torch.tensor(past_rotations_normalized).float().unsqueeze(0).unsqueeze(2)  # shape [1, N, 1]
            
            # Stack context tensors
            context_xs = torch.cat([
                past_angles,  # [1, N, 1]
                past_velocities,  # [1, N, 1]
                past_initials,  # [1, N, 1]
            ], dim=2)  # shape [1, N, 3]
            context_ys = torch.cat((past_finals, past_rotations), dim=2)  # shape [1, N, 4]
            
            target_xs = current_target_xs
        else:
            # No past actions, use zero tensors for context and current action for target
            context_xs = torch.zeros(1, 1, 3)  # Empty context with just angle, velocity, and position
            context_ys = torch.zeros(1, 1, 4)  # Empty context outputs
            target_xs = current_target_xs  # Just use the current action as target
        
        if torch.cuda.is_available():
            target_xs = target_xs.cuda()
            context_xs = context_xs.cuda()
            context_ys = context_ys.cuda()
        
        # Get prediction from neural process
        with torch.no_grad():
            # print("MODEL INPUT SHAPE", context_xs.shape, context_ys.shape, target_xs.shape)
            total_loss, bce_loss, kl_loss, mu, sigma, distance, dist = self.model(
                context_xs, context_ys, target_xs, None, None, None, mode="test"
            )
            
            # Extract the predicted next state from mu in normalized space
            # print(mu, sigma)
            # print(mu.shape, sigma.shape)
            next_state_normalized = dist.sample().cpu().numpy()[0][0]
        
        # The neural network predicts transformations/deltas, not absolute states
        # Convert predicted transformation back to real-world coordinates
        predicted_transform = np.array([
            self.inverse_transform_coordinates(next_state_normalized[0], 'final_position_x'),
            self.inverse_transform_coordinates(next_state_normalized[1], 'final_position_y'),
            self.inverse_transform_coordinates(next_state_normalized[2], 'final_position_z'),
            self.inverse_transform_coordinates(next_state_normalized[3], 'output_angle')
        ])
        
        # Ensure the output angle is in the [-π, π] range
        predicted_transform[3] = (predicted_transform[3] + np.pi) % (2 * np.pi) - np.pi
        
        # Extract the current state from belief history
        if len(b['states']) > 0:
            current_state_real = np.array(b['states'][-1])
        else:
            # Use initial state if no history yet
            current_state_real = np.array(b['initial_state']) if 'initial_state' in b else np.zeros(4)
            
        # Apply the predicted transformation to get the next state
        next_state_real = np.array([
            current_state_real[0] + predicted_transform[0],  # Add predicted x delta
            current_state_real[1] + predicted_transform[1],  # Add predicted y delta
            current_state_real[2] + predicted_transform[2],  # Add predicted z delta
            predicted_transform[3]                           # Use predicted absolute rotation (already in [-π, π])
        ])
        
        # In NPTDPW we just use the model prediction directly, no observation parameter needed
        # This is different from PFTDPW which updates beliefs using observations
        
        # Create new belief state with updated history in real-world coordinates
        # Convert action to a hashable format (not numpy array)
        hashable_action = tuple(float(x) if isinstance(x, (np.number, np.ndarray)) else x for x in a)
        
        new_b = {
            'states': b['states'] + [next_state_real.tolist() if isinstance(next_state_real, np.ndarray) else next_state_real],
            'actions': b['actions'] + [hashable_action],
            'outcomes': b['outcomes'] + [predicted_transform.tolist() if isinstance(predicted_transform, np.ndarray) else predicted_transform]
        }
        
        # Calculate distance to goal in real-world coordinates
        distance_to_goal = np.linalg.norm(next_state_real[:2] - self.goal_loc[:2])
        
        # Calculate reward as negative distance scaled by a factor
        distance_reward = -distance_to_goal * 10
        
        # Add a success bonus if the block is within a threshold distance from the goal
        # Using a default threshold of 0.2 (can be overridden by success_threshold parameter)
        success_threshold = 0.2 if not hasattr(self, 'success_threshold') else self.success_threshold
        if distance_to_goal < success_threshold:
            # Add large positive bonus for being close to the goal
            distance_reward += 100000.0  # Strong positive reward for success
        
        # Calculate table boundary loss if visualization is enabled
        table_loss = 0.0
        if hasattr(self, 'vis_table') and self.vis_table is not None:
            # Check if the new position is off the table
            pos = np.array([next_state_real[0], next_state_real[1]])
            table_value = self.vis_table.is_on_table_fn(pos)
            table_loss = 1.0 if table_value < 0 else 0.0
            
            # Apply a large penalty for going off the table
            distance_reward -= 100.0 * table_loss
        
        # Return the updated belief and the reward
        return new_b, distance_reward

    def simulate(self, b, depth):
        # print(depth)
        if depth == 0:
            return 0
        # print(depth)
        # print(b)
            
        a = self.action_prog_widen(b)
        # Ensure action is hashable and convert any numpy values to regular Python types
        if isinstance(a, np.ndarray):
            a = tuple(float(x) if isinstance(x, np.number) else x for x in a.tolist())
        elif isinstance(a, np.number):
            a = (float(a),)
        elif not isinstance(a, tuple):
            a = (a,)
        
        b_key = self._belief_to_key(b)
        # Create a hashable key for dictionary operations
        hash_key = b_key + a
        if hash_key not in self.n:
            self.n[hash_key] = 0
            self.q[hash_key] = 0
            self.c[hash_key] = []

        total = 0
        if len(self.c[hash_key]) <= self.n[hash_key] ** self.alpha:
            # Generate a new state using neural process
            new_b, r = self.update(b, a)
            
            self.c[hash_key].append((new_b, r))
            total = r + self.discount_factor * self.rollout(new_b, depth - 1)
        else:
            random_index = np.random.choice(len(self.c[hash_key]), 1)[0]
            new_b, r = self.c[hash_key][random_index]
            total = r + self.discount_factor * self.simulate(new_b, depth - 1)
        
        self.n[b_key] += 1
        self.n[hash_key] += 1
        self.q[hash_key] = self.q.get(hash_key, 0) + (total - self.q.get(hash_key, 0)) / self.n[hash_key]
        return total

    def rollout(self, b, depth):
        if depth == 0:
            return 0

        a = np.random.uniform(0, 1)
        a = (a, )
        new_b, r = self.update(b, a)
        return r + self.discount_factor * self.rollout(new_b, depth - 1)

    def _belief_to_key(self, b):
        """Convert belief state dictionary to a hashable tuple."""
        # Process states - ensure all elements are hashable
        states_list = []
        for s in b['states']:
            if isinstance(s, np.ndarray):
                states_list.append(tuple(s.tolist()))
            else:
                states_list.append(tuple(s))
        states = tuple(states_list)
        
        # Process actions - ensure all elements are hashable
        actions_list = []
        for a in b['actions']:
            if isinstance(a, np.ndarray):
                actions_list.append(tuple(a.tolist()))
            elif isinstance(a, (list, tuple)):
                actions_list.append(tuple(a))
            else:
                actions_list.append((a,))
        actions = tuple(actions_list)
        
        # Process outcomes - ensure all elements are hashable
        outcomes_list = []
        for o in b['outcomes']:
            if isinstance(o, np.ndarray):
                outcomes_list.append(tuple(o.tolist()))
            else:
                outcomes_list.append(tuple(o))
        outcomes = tuple(outcomes_list)
        
        return (states, actions, outcomes)

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
            scale = self.dataset.scalers.get(f"{base_field}_scale")
            
            if mean is None or scale is None:
                return coordinates
                
            # Apply the inverse transformation: x * scale + mean
            real_world = coordinates * scale + mean
            return real_world
        else:
            # No transformation for this field
            return coordinates
            
    def action_prog_widen(self, b):
        b_key = self._belief_to_key(b)
        if b_key not in self.n:
            self.n[b_key] = 0
        if b_key not in self.c:
            self.c[b_key] = []
        if len(self.c[b_key]) <= self.n[b_key] ** self.alpha:
            # Generate a random action in normalized space
            a_normalized = np.random.uniform(0, 1)
            
            # Convert to real-world coordinates for storage
            a_real = self.inverse_transform_coordinates(a_normalized, 'angle')
            
            # Convert numpy arrays to regular Python float to ensure hashability
            if isinstance(a_real, np.ndarray):
                a_real = float(a_real.item()) if a_real.size == 1 else tuple(float(x) for x in a_real.tolist())
            elif isinstance(a_real, np.number):
                a_real = float(a_real)
                
            # Store as a tuple for consistency
            a_real_tuple = (a_real,) if not isinstance(a_real, tuple) else a_real
            self.c[b_key].append(a_real_tuple)
            return a_real_tuple
        else:
            values = []
            for c in self.c[b_key]:
                # Ensure action is hashable
                if isinstance(c, np.ndarray):
                    c_hash = tuple(c.tolist())
                elif not isinstance(c, tuple):
                    c_hash = (c,)
                else:
                    c_hash = c
                
                # Create a hashable key for dictionary operations
                hash_key = b_key + c_hash
                values.append(self.q[hash_key] + self.const * 
                              np.sqrt(np.log(self.n[b_key]) / self.n[hash_key]))
            
            return self.c[b_key][np.argmax(values)]

    def initialize_visualization_table(self, block_width=0.07, block_length=0.2, table_length=3.0, table_width=3.0):
        """
        Initialize a table for visualization purposes with smaller dimensions to create a shorter planning horizon.
        
        Args:
            block_width: Width of the block
            block_length: Length of the block
            table_length: Length of the table (default reduced to 3.0)
            table_width: Width of the table (default reduced to 3.0)
            
        Returns:
            The initialized table instance
        """
        # Use standardized table dimensions that match PFTDPW for consistent behavior
        # These are proportional to the object dimensions
        
        # Create the appropriate table type based on the table_type parameter
        if self.table_type == 'ring':
            # RingTable with smaller dimensions for shorter problem horizon
            self.vis_table = RingTable(
                block_width=block_width,
                block_length=block_length,
                block_com_relative_to_centroid=np.array([0, 0]),
                ring_center=np.array([table_length, table_width]),  # Center of the ring
                inner_rad=table_width/3,     # Inner radius - slightly bigger proportion
                outer_rad=table_width/1.25,   # Outer radius - slightly bigger proportion
                platform_rad=table_width/4   # Platform radius - slightly bigger proportion
                # Note: track_middle_rad_disp is calculated internally by RingTable
            )
            # print("GOAL: ", self.vis_table.goal_pos)
            # Update goal location from the table
            self.goal_loc = np.append(self.vis_table.goal_pos, [0.0, 0.0])  # Add z and angle components
        elif self.table_type == 'beam':
            # BeamTable with smaller dimensions for shorter problem horizon
            self.vis_table = BeamTable(
                block_width=block_width,
                block_length=block_length,
                block_com_relative_to_centroid=np.array([0, 0]),
                table_length=table_length,      # Length of the beam
                narrow_width=table_width/2.5,   # Width of the beam - slightly wider proportion
                platform_rad=table_width/2.5    # Size of end platforms - slightly larger proportion
            )
            # Update goal location from the table
            self.goal_loc = np.append(self.vis_table.goal_pos, [0.0, 0.0])  # Add z and angle components
        else:  # Default to box table
            self.vis_table = BoxTable(
                block_width=block_width,
                block_length=block_length,
                block_com_relative_to_centroid=np.array([0, 0]),
                goal_loc_x=table_length*0.8,    # Goal position proportional to table size
                goal_loc_y=table_width*0.8,     # Goal position proportional to table size 
                table_length=table_length,
                table_width=table_width
            )
            # Update goal location from the table
            self.goal_loc = np.append(self.vis_table.goal_pos, [0.0, 0.0])  # Add z and angle components
        
        return self.vis_table
        
    def visualize_planning(self, history=None, show=False, save_path=None, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, res=0.01):
        """
        Visualize the planning and execution process using the visualization table.
        
        Args:
            history: Dictionary containing 'states', 'actions', and 'transformations' from execute_planning_loop
            show: Whether to show the plot interactively (default is False)
            save_path: Path to save the visualization image (if None, uses timestamp)
            min_x, max_x, min_y, max_y: Bounds for the visualization
            res: Resolution for the table visualization grid
            
        Returns:
            The visualization table instance
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
                # Extract the transformation components
                dx, dy, dz, dtheta = transform
                
                # Create a transformation matrix in SE(2) format
                pose = np.eye(3)
                
                # Apply the translation
                pose[0, 2] = dx
                pose[1, 2] = dy
                
                # Apply the rotation - dtheta should already be in radians
                pose[0, 0] = np.cos(dtheta)
                pose[0, 1] = -np.sin(dtheta)
                pose[1, 0] = np.sin(dtheta)
                pose[1, 1] = np.cos(dtheta)
                
                # Apply the transformation to the visualization
                self.vis_table.apply_push_trajectory([pose])
        
        # Visualize the table
        if save_path is None:
            # Generate a unique filename with timestamp if none provided
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = f'planning_visualization_{timestamp}.png'
            
        # Create visualization and save it
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create visualization
        self.vis_table.visualize(show=show, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, res=res)
        
        # Save the figure
        plt.savefig(save_path, dpi=300)
        
        if not show:
            plt.close()
                
        return self.vis_table
        
    def execute_planning_loop(self, initial_state=None, max_steps=50, planning_time=1.0, success_threshold=0.2, visualize=False, show_visualization=False, save_visualization_path=None):
        """
        Execute the complete planning and execution loop.
        
        Args:
            initial_state: Initial state of the system (x, y, theta, phi) in real-world coordinates. If None, use (0,0,0,0)
            max_steps: Maximum number of planning steps
            planning_time: Time allocated for planning at each step (seconds)
            success_threshold: Distance threshold to consider goal reached (in real-world coordinates)
            visualize: Whether to visualize the planning results
            show_visualization: Whether to show the visualization interactively
            save_visualization_path: Path to save the visualization image (if None, uses timestamp)
            
        Returns:
            dict: History of states, actions, and outcomes in real-world coordinates
        """
        # Reset statistics for new execution
        self.stats = {
            # Planning statistics
            'planning_times': [],          # Time spent planning at each step
            'action_counts': {},           # Frequency of actions chosen
            'visits_per_step': [],         # Number of nodes visited in each planning step
            'depth_reached': [],           # Maximum depth reached in planning
            
            # Execution statistics
            'step_durations': [],          # Time taken for each step (planning + execution)
            'distances_to_goal': [],       # Distance to goal at each step
            'state_trajectory': [],        # Complete state trajectory
            'action_trajectory': [],       # Complete action trajectory
            'cumulative_translation': 0.0, # Total distance traveled
            'cumulative_rotation': 0.0,    # Total rotation performed
            
            # Success metrics
            'final_distance': None,        # Final distance to goal
            'success': False,              # Whether goal was reached
            'success_step': None,          # Step at which goal was reached
            'total_steps': 0,              # Total steps executed
            'total_planning_time': 0.0,    # Total time spent in planning
            'total_execution_time': 0.0,   # Total time spent in execution
            'fell_off_table': False,       # Whether block fell off table
            'total_reward': 0.0,           # Cumulative reward during planning
            
            # Prediction accuracy metrics
            'prediction_errors': [],       # Differences between predicted and actual outcomes
            'model_confidence': []         # Model's reported confidence in predictions
        }
        
        # Start tracking execution time
        execution_start_time = time.time()
        
        # Initialize visualization table first (if requested)
        if visualize and not hasattr(self, 'vis_table'):
            self.initialize_visualization_table(
                block_width=0.07, 
                block_length=0.2,
                table_length=3.0,
                table_width=3.0
            )
        
        # Always use the table's start position if we have a table, regardless of whether the user provided an initial state
        if hasattr(self, 'vis_table') and self.vis_table is not None:
            # Get the start position from the table's init_pose attribute (SE(2) matrix)
            # The last column of init_pose contains the x,y coordinates (first two elements)
            start_x = float(self.vis_table.init_pose[0, 2])  # x-coordinate from transformation matrix
            start_y = float(self.vis_table.init_pose[1, 2])  # y-coordinate from transformation matrix
            initial_state = (start_x, start_y, 0.0, 0.0)
            # print(f"Using table's start position: {initial_state}")
        elif initial_state is None:
            # Only use the default if we don't have a table and no initial state was provided
            initial_state = (0.0, 0.0, 0.0, 0.0)
            
        # Initialize the state in real-world coordinates
        current_state_real = np.array(initial_state)
            
        # Create initial belief state in real-world coordinates
        belief = {
            'states': [current_state_real.tolist()],
            'actions': [],
            'outcomes': []
        }
        
        # Visualization table has already been initialized above

        # Initialize a history dictionary to track all states and actions in real-world coordinates
        history = {
            'states': [current_state_real.copy()],
            'actions': [],
            'transformations': [], # For tracking relative movements between states
            'outcomes': []  # Full outcome states after each push
        }
        
        # Record initial distance to goal
        initial_dist = np.linalg.norm(current_state_real[:2] - self.goal_loc[:2])
        self.stats['distances_to_goal'].append(initial_dist)
        
        for step in range(max_steps):
            # Start timing the step
            step_start_time = time.time()
            
            # Check if goal is reached using real-world coordinates
            dist_to_goal = np.linalg.norm(current_state_real[:2] - self.goal_loc[:2])
            self.stats['distances_to_goal'].append(dist_to_goal)
            
            if dist_to_goal < success_threshold:
                # Goal reached - print clear success message
                # print(f"\nSUCCESS! Goal reached at step {step+1} with distance {dist_to_goal:.3f} < threshold {success_threshold}")
                # print(f"Final position: x={current_state_real[0]:.3f}, y={current_state_real[1]:.3f}")
                
                self.stats['success'] = True
                self.stats['success_step'] = step
                break
            
            # Plan next action using the belief state
            # The action returned is in real-world coordinates
            plan_start_time = time.time()
            action = self.plan(belief, total_time=planning_time)
            plan_duration = time.time() - plan_start_time
            self.stats['planning_times'].append(plan_duration)
            self.stats['total_planning_time'] += plan_duration
            # Keep the original action format without modifying the angle range
            
            # DEBUGGING: Get the model's prediction for this action before executing it
            try:
                # Create a temporary copy of the belief for model prediction
                temp_belief = belief.copy()
                # Call update to get the model's prediction (but don't apply it yet)
                temp_belief, _ = self.update(temp_belief, action if isinstance(action, tuple) else (action,))
                # Get the predicted next state
                model_next_state = temp_belief['states'][-1] if 'states' in temp_belief and len(temp_belief['states']) > 0 else None
                
                if model_next_state is not None:
                    # Calculate the predicted delta if we have a valid next state
                    current_state = belief['states'][-1] if len(belief['states']) > 0 else belief['initial_state']
                    pred_delta = np.array([
                        model_next_state[0] - current_state[0],  # x delta
                        model_next_state[1] - current_state[1],  # y delta
                        model_next_state[2] - current_state[2],  # z delta
                    ])
                    
                    # Calculate angle delta in the shortest direction
                    current_angle = (current_state[3] + np.pi) % (2 * np.pi) - np.pi
                    model_angle = (model_next_state[3] + np.pi) % (2 * np.pi) - np.pi
                    angle_delta = model_angle - current_angle
                    if angle_delta > np.pi:
                        angle_delta -= 2 * np.pi
                    elif angle_delta < -np.pi:
                        angle_delta += 2 * np.pi
                    
                    pred_delta = np.append(pred_delta, angle_delta)
                    
                    # Print model prediction
                    # Model prediction (silenced)
            except Exception as e:
                pass  # Could not get model prediction (silenced)
            
            # Convert action to proper format
            if not isinstance(action, tuple):
                action = (action,)
            
            # Keep the action value in radians for internal use
            if isinstance(action, tuple) and len(action) == 1:
                action_value = action[0]
            else:
                action_value = action
                
            if isinstance(action_value, (float, int)):
                # Store the action in the history in radians
                history['actions'].append(float(action_value))
                # Calculate degrees only for display purposes
                action_degrees = float(np.degrees(action_value))
            else:
                # If it's not a simple angle, store the raw action
                history['actions'].append(action)
                action_degrees = None
                
            # Execute action using real physics simulation
            # Execute action using real physics simulation - we've already printed the model prediction earlier
            
            # Perform the simulation with the chosen action to get the next state
            current_state_real_raw = np.array(current_state_real)
            # Prepare state with COM information if needed
            sim_state = np.concatenate([np.array(self.true_com[0:2]), current_state_real_raw], axis=0) if hasattr(self, 'true_com') else current_state_real_raw
            # Use imported real_simulate function instead of class method
            # Format action as a tuple or list since real_simulate expects an indexable object
            action_tuple = (action_value,) if isinstance(action_value, (int, float)) else action_value
            sim_result = real_simulate(sim_state, action_tuple, self.dataset, real=True)
            # print(sim_result)
            sim_result = np.array(sim_result)
            
            # Update the current state using the world-frame results from real simulation
            next_state_real = np.array([
                current_state_real[0] + sim_result[0],  # Add world x translation
                current_state_real[1] + sim_result[1],  # Add world y translation
                current_state_real[2] + sim_result[2],  # Add z translation
                sim_result[3]                          # Use the new absolute rotation
            ])
            
            # Check if block fell off table
            # Define table boundaries (assuming table is centered at origin with dimensions table_length x table_width)
            table_length = 10.0  # Default value, should match table initialization
            table_width = 10.0   # Default value, should match table initialization
            
            if hasattr(self.vis_table, 'table_length') and hasattr(self.vis_table, 'table_width'):
                table_length = self.vis_table.table_length
                table_width = self.vis_table.table_width
            
            # Check if block fell off table
            # print("NEXT STATE REAL", next_state_real)
            # Ensure vis_table is initialized
            if not hasattr(self, 'vis_table') or self.vis_table is None:
                self.initialize_visualization_table(
                    block_width=0.07,
                    block_length=0.2,
                    table_length=3.0,
                    table_width=3.0
                )
                
            # Check if block is on the table
            if self.vis_table.is_on_table_fn(next_state_real[:2]) < 0.0: 
                # Block fell off table message (silenced)
                self.stats['fell_off_table'] = True
                # print("FELL")
                break  # End planning loop if block falls off table
            
            # Update our belief state by applying the action
            # In NPTDPW we don't pass observations to update, unlike PFTDPW
            
            # Convert world-frame translation to object-local coordinates for visualization
            current_ori = current_state_real[3]
            # Ensure current_ori is in [-π, π] range for consistent calculations
            current_ori = (current_ori + np.pi) % (2 * np.pi) - np.pi
            
            world_to_object_rotation = np.array([
                [np.cos(current_ori), np.sin(current_ori)],
                [-np.sin(current_ori), np.cos(current_ori)]
            ])
            
            # Convert world-frame translation to object-local coordinates
            # Convert world-frame translation to object-frame translation
            translation_local = world_to_object_rotation @ sim_result[:2]
            
            # Create the adjusted transformation with object-local coordinates for visualization
            adjusted_sim_result = np.zeros_like(sim_result)
            adjusted_sim_result[:2] = translation_local
            adjusted_sim_result[2] = sim_result[2]  # Z-translation unchanged
            adjusted_sim_result[3] = sim_result[3] - current_ori  # Relative rotation
            
            # Store both the local-frame transformation for visualization
            history['transformations'].append(adjusted_sim_result.copy())
            
            # Calculate the world-frame state delta for tracking and reference
            # Ensure angles are in [-π, π] for proper delta calculation
            current_angle = (current_state_real[3] + np.pi) % (2 * np.pi) - np.pi
            result_angle = (sim_result[3] + np.pi) % (2 * np.pi) - np.pi
            
            # Calculate angle delta in the shortest direction
            angle_delta = result_angle - current_angle
            # Normalize to [-π, π]
            if angle_delta > np.pi:
                angle_delta -= 2 * np.pi
            elif angle_delta < -np.pi:
                angle_delta += 2 * np.pi
                
            state_delta = np.array([
                sim_result[0],  # x delta from simulation
                sim_result[1],  # y delta from simulation
                sim_result[2],  # z delta from simulation
                angle_delta     # orientation delta in [-π, π] range
            ])
            
            # Calculate translation and rotation for statistics
            translation_distance = np.linalg.norm(sim_result[:2])  # Euclidean distance traveled in this step
            rotation_amount = abs(angle_delta)  # Absolute rotation amount
            
            # Update cumulative statistics
            self.stats['cumulative_translation'] += translation_distance
            self.stats['cumulative_rotation'] += rotation_amount
            
            # Calculate reward for this step (negative distance to goal)
            current_dist = np.linalg.norm(next_state_real[:2] - self.goal_loc[:2])
            reward = -current_dist  # Negative distance as reward
            self.stats['total_reward'] += reward
            
            # Store the outcome state and update states history with cumulative effect
            history['outcomes'].append(next_state_real.copy())
            history['states'].append(next_state_real.copy())
            belief['outcomes'].append(sim_result.copy())
            belief['states'].append(next_state_real.copy())
            belief['actions'].append(action) 
            
            # Update our accumulated state - this is crucial for composing multiple pushes
            current_state_real = next_state_real.copy()
            
            # Record state for statistics
            self.stats['state_trajectory'].append(current_state_real.copy())
            self.stats['action_trajectory'].append(action_degrees if action_degrees is not None else action)
            
            # Print information about the push effect and the resulting position
            # Convert angle to degrees only for display
            
            # Print current location and distance to goal after each push
            # print(f"Step {step+1} location: x={current_state_real[0]:.3f}, y={current_state_real[1]:.3f}, θ={np.degrees(current_state_real[3]):.1f}°")
            # print(f"Distance to goal: {current_dist:.3f} units | Goal: ({self.goal_loc[0]:.3f}, {self.goal_loc[1]:.3f})")
            
            # Update action count statistics
            action_key = f"{action_degrees:.1f}°" if action_degrees is not None else str(action)
            if action_key in self.stats['action_counts']:
                self.stats['action_counts'][action_key] += 1
            else:
                self.stats['action_counts'][action_key] = 1
                
            # Calculate step duration and store it
            step_duration = time.time() - step_start_time
            self.stats['step_durations'].append(step_duration)
            
            # Increment total steps counter
            self.stats['total_steps'] += 1
            
            # Create new belief state with updated orientation
            next_ori = (current_state_real[3] + np.pi) % (2 * np.pi) - np.pi
            new_belief = (current_state_real[0], current_state_real[1], current_state_real[2], next_ori)
            
            # Update our belief state for the next planning iteration
            belief = {'states': [new_belief], 'initial_state': new_belief, 'actions': [], 'outcomes': []}
        
        # Record final statistics
        execution_end_time = time.time()
        self.stats['total_execution_time'] = execution_end_time - execution_start_time
        
        # Final distance to goal
        final_dist = np.linalg.norm(current_state_real[:2] - self.goal_loc[:2])
        self.stats['final_distance'] = final_dist
        
        # If we didn't reach the goal, print the final position
        if final_dist >= success_threshold:
            pass  # Planning ended message (silenced)
        
        # Print summary of key metrics
        # Execution summary is now handled by generate_experiment_data.py
        
        # Automatically visualize the planning results if requested
        if visualize:
            # Visualize the planning history
            self.visualize_planning(
                history=history,
                show=show_visualization,
                save_path=save_visualization_path)
        
        # Add statistics to history for return
        history['stats'] = self.stats
        
        return history

def main(args):
    # Set up paths
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset)
    instance_path = os.path.join(dataset_path, args.instance)

    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    # Load datasets
    with open(os.path.join(instance_path, 'train_dataset.pkl'), 'rb') as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'rb') as f:
        validation_dataset = pickle.load(f)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Get the model using the training function
    if os.path.exists(os.path.join(instance_path, 'args.pkl')):
        with open(os.path.join(instance_path, 'args.pkl'), 'rb') as f:
            train_args = pickle.load(f)
    else:
        train_args = args
    model = AttentionPushNP(train_args)
    if not os.path.exists(os.path.join(instance_path, 'best_model.pth')):
        model = train(model, args, train_loader, val_loader)
    else: 
        model.load_state_dict(torch.load(os.path.join(instance_path, 'best_model.pth')))
    # Create NPTDPW instance with appropriate parameters
    # Using success_threshold of 0.2 for consistent behavior with PFTDPW
    success_threshold = 0.2
    
    nptdpw = NPTDPW(
        model=model,
        dataset=validation_dataset,
        true_com=[0.3, 0.5, 0],
        alpha=0.5,
        beta=0.8,
        const=0.1,
        search_depth=2,
        goal_loc=[2.4, 2.4, 0.0, 0.0],  # Set goal at 80% of table dimensions (3.0 * 0.8 = 2.4)
        args=args,
        table_type=args.table_type,  # Pass the table type from command line args
        success_threshold=success_threshold,  # Pass the success threshold for reward bonuses
        discount_factor=0.6
    )

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
        nptdpw.initialize_visualization_table(
            block_width=0.07,
            block_length=0.2,
            table_length=3.0,  # Standardized table length for consistent behavior with PFTDPW
            table_width=3.0,   # Standardized table width for consistent behavior with PFTDPW
        )

    # Set the path for saving the visualization
    save_visualization_path = None
    if visualize:
        save_visualization_path = os.path.join(vis_output_dir, f'planning_visualization_{timestamp}.png')

    # Execute planning with visualization settings
    example_plan = nptdpw.execute_planning_loop(
        initial_state=None,  # Set to None to use the table's start position
        max_steps=args.max_steps,
        planning_time=5.0,    # Increased planning time to match PFTDPW
        success_threshold=success_threshold,  # Use the same success threshold for consistency
        visualize=visualize,
        show_visualization=show_visualization,
        save_visualization_path=save_visualization_path
    )

    # Additional visualization with custom parameters if requested
    if visualize and hasattr(args, 'additional_visualization') and args.additional_visualization:
        detailed_vis_path = os.path.join(vis_output_dir, f'detailed_visualization_{timestamp}.png')
        nptdpw.visualize_planning(
            history=example_plan,
            show=show_visualization,
            save_path=detailed_vis_path,
            res=0.005  # Higher resolution for detailed visualization
        )

def get_model(args):
    """Initialize and return a new AttentionPushNP model with the given arguments."""
    return AttentionPushNP(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--table-type', type=str, choices=['box', 'ring', 'beam'], default='box',
                        help='Type of table to use for visualization')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-points', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--guess-obj', action='store_true')
    parser.add_argument('--no-visualization', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--vis-output-dir', type=str, default=None)
    parser.add_argument('--additional-visualization', action='store_true')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--use-full-trajectory', action='store_true', help='Use full trajectory information instead of just contact points')
    parser.add_argument('--no-pointnet', action='store_true')
    parser.add_argument('--no-deterministic', action='store_true')
    parser.add_argument('--use-mixture', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--d-latents', type=int, default=5)
    parser.add_argument('--attention-encoding', type=int, default=512)
    parser.add_argument('--no-kl', action='store_true')
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--use-regression-model', action='store_true')
    parser.add_argument('--regression-model', type=str, default=None)
    parser.add_argument('--learning-rate', type=float, default=1e-3)

    args = parser.parse_args()
    args.no_deterministic = True
    args.use_obj_prop = True
    args.guess_obj = False
    args.latent_samp = -1
    args.point_cloud = True
    args.no_contact = True
    args.no_pointnet = True

    main(args)
