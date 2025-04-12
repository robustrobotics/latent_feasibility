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
from learning.domains.pushing.virtual_tables import BoxTable
from learning.domains.pushing.find_contact_points import run_sim  # Required for real_simulate
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody  # Required for real_simulate
from learning.models.push_np.train_APNP import train  # Import the training function
from learning.models.push_np.PFTDPW import real_simulate  # Import real_simulate function for physics simulation

class NPTDPW:
    def __init__(self, model, dataset, true_com, search_depth=3, goal_loc=(100, 100, 0, 0), discount_factor=0.8, alpha=0.6, beta=0.5, const=2.0, args=None):
        self.c = {} # child
        self.q = {} # value
        self.n = {} # number of visits
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
        self.vis_table = None

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
            if value > best_value: 
                best_value = value
                best_action = action
        
        print(f"Total searches performed: {search_count}")
        
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
        # Convert current state to tensor
        if not isinstance(a, tuple):
            a = (a,)
        current_state_real = b['states'][-1] if b['states'] else np.zeros(4)
        
        # Transform the current state to model's normalized space - collect each component separately to avoid mixed types
        pos_x = self.transform_coordinates(current_state_real[0], 'final_position_x')
        pos_y = self.transform_coordinates(current_state_real[1], 'final_position_y')
        pos_z = self.transform_coordinates(current_state_real[2], 'final_position_z')
        angle_transformed = self.transform_coordinates(current_state_real[3], 'output_angle')[0]  # Extract scalar from array
        
        # Store as separate components to avoid array with mixed types
        current_state_transformed = np.array([pos_x, pos_y, pos_z, angle_transformed])
        
        # Transform the action angle to normalized space
        # The input action is in real-world coordinates and needs to be transformed
        transformed_action = self.transform_coordinates(a[0], 'angle')
        # Extract scalar value if it's in an array
        if hasattr(transformed_action, '__len__') and len(transformed_action) == 1:
            transformed_action = transformed_action[0]
        
        # Prepare input tensors for the model (in normalized space)
        angle = torch.tensor([[transformed_action]]).float().unsqueeze(0)
        push_velocities = torch.tensor([[1.0]]).float().unsqueeze(0)  # shape [1, 1, 1]
        initials = torch.tensor([[current_state_transformed[:3].tolist()]]).float()  # shape [1, 1, 3]
        
        # Stack tensors for current action
        current_target_xs = torch.cat([
            angle,  # [1, 1, 1]
            push_velocities,  # [1, 1, 1]
            initials  # [1, 1, 3]
        ], dim=2)  # shape [1, 1, 5]
        
        # Handle past actions and observations if they exist (all in normalized space)
        if b['actions'] and b['outcomes']:
            # Get past states and transform them to normalized space
            past_states_normalized = []
            for state in b['states'][:-1]:  # Skip the current state
                # Process each component separately to avoid mixed types
                x = self.transform_coordinates(state[0], 'final_position_x')
                y = self.transform_coordinates(state[1], 'final_position_y')
                z = self.transform_coordinates(state[2], 'final_position_z')
                # Only need position components for past states
                past_states_normalized.append([float(x), float(y), float(z)])  
                
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
            past_velocities = torch.ones_like(past_angles)  # shape [1, N, 1]
            past_initials = torch.tensor(past_states_normalized).float().unsqueeze(0)  # shape [1, N, 3]
            past_finals = torch.tensor(past_outcomes_normalized).float().unsqueeze(0)  # shape [1, N, 3]
            past_rotations = torch.tensor(past_rotations_normalized).float().unsqueeze(0).unsqueeze(2)  # shape [1, N, 1]
            
            # Stack context tensors
            context_xs = torch.cat([
                past_angles,  # [1, N, 1]
                past_velocities,  # [1, N, 1]
                past_initials  # [1, N, 3]
            ], dim=2)  # shape [1, N, 5]
            context_ys = torch.cat((past_finals, past_rotations), dim=2)  # shape [1, N, 4]
            
            # Combine past actions and current action for target_xs
            target_xs = torch.cat((context_xs, current_target_xs), dim=1)  # shape [1, N+1, 5]
        else:
            # No past actions, use zero tensors for context and current action for target
            context_xs = torch.zeros(1, 1, 5)  # Empty context with just angle, velocity, and position
            context_ys = torch.zeros(1, 1, 4)  # Empty context outputs
            target_xs = current_target_xs  # Just use the current action as target
        
        if torch.cuda.is_available():
            target_xs = target_xs.cuda()
            context_xs = context_xs.cuda()
            context_ys = context_ys.cuda()
        
        # Get prediction from neural process
        with torch.no_grad():
            total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy = self.model(
                context_xs, context_ys, target_xs, None, None, None, mode="test"
            )
            
            # Extract the predicted next state from mu in normalized space
            next_state_normalized = mu[0, -1].cpu().numpy()  # First batch, last sample (current action)
        
        # The neural network predicts transformations/deltas, not absolute states
        # Convert predicted transformation back to real-world coordinates
        predicted_transform = np.array([
            self.inverse_transform_coordinates(next_state_normalized[0], 'final_position_x'),
            self.inverse_transform_coordinates(next_state_normalized[1], 'final_position_y'),
            self.inverse_transform_coordinates(next_state_normalized[2], 'final_position_z'),
            self.inverse_transform_coordinates(next_state_normalized[3], 'output_angle')
        ])
        
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
            predicted_transform[3]                           # Use predicted absolute rotation
        ])
        
        # In NPTDPW we just use the model prediction directly, no observation parameter needed
        # This is different from PFTDPW which updates beliefs using observations
        
        # Create new belief state with updated history in real-world coordinates
        # Convert action to a hashable format (not numpy array)
        hashable_action = tuple(float(x) if isinstance(x, (np.number, np.ndarray)) else x for x in a)
        
        new_b = {
            'states': b['states'] + [current_state_real.tolist() if isinstance(current_state_real, np.ndarray) else current_state_real],
            'actions': b['actions'] + [hashable_action],
            'outcomes': b['outcomes'] + [next_state_real.tolist() if isinstance(next_state_real, np.ndarray) else next_state_real]
        }
        
        # Calculate reward as negative L2 distance to goal in real-world coordinates
        distance_reward = -np.linalg.norm(next_state_real[:2] - self.goal_loc[:2])
        
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
        if depth == 0:
            return 0
            
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
        # Use goal location to determine table size
        if hasattr(self, 'goal_loc') and self.goal_loc is not None:
            table_length = max(self.goal_loc[0] * 1.5, 5.0)  
            table_width = max(self.goal_loc[1] * 1.5, 5.0)
        
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
        
        # If we have actions, add push direction arrows to the visualization
        if history is not None and 'states' in history and 'actions' in history and len(history['actions']) > 0:
            # Draw arrows for each push action
            for i, (state, action) in enumerate(zip(history['states'][:-1], history['actions'])):
                # The action should be in radians
                if isinstance(action, (float, int)):
                    # Calculate arrow endpoint using trigonometry
                    arrow_length = 0.3  # visual length of arrow
                    dx = arrow_length * np.cos(action)
                    dy = arrow_length * np.sin(action)
                    
                    # Draw arrow from state position
                    ax.arrow(state[0], state[1], dx, dy, head_width=0.1, head_length=0.1, 
                             fc='blue', ec='blue', alpha=0.7)
                    
                    # Add step number label
                    ax.text(state[0], state[1], f"{i+1}", fontsize=10, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='circle'))
        
        # Add title with information about the planning sequence
        if history is not None:
            num_steps = len(history['actions'])
            plt.title(f"Planning Trajectory: {num_steps} steps", fontsize=14)
        
        # Save the figure
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
        
        if not show:
            plt.close()
                
        return self.vis_table
        
    def execute_planning_loop(self, initial_state=None, max_steps=50, planning_time=1.0, success_threshold=0.05, visualize=False, show_visualization=False, save_visualization_path=None):
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
        # Ensure starting pose is (0,0,0,0) if not specified
        if initial_state is None:
            initial_state = (0.0, 0.0, 0.0, 0.0)
        
        # Initialize the state in real-world coordinates
        current_state_real = np.array(initial_state)
            
        # Create initial belief state in real-world coordinates
        belief = {
            'states': [current_state_real.tolist()],
            'actions': [],
            'outcomes': []
        }
        
        # Initialize visualization table if needed
        if visualize and self.vis_table is None:
            self.initialize_visualization_table()
        
        # Initialize a history dictionary to track all states and actions in real-world coordinates
        history = {
            'states': [current_state_real.copy()],
            'actions': [],
            'transformations': [], # For tracking relative movements between states
            'outcomes': []  # Full outcome states after each push
        }
        
        for step in range(max_steps):
            # Check if goal is reached using real-world coordinates
            dist_to_goal = np.linalg.norm(current_state_real[:2] - self.goal_loc[:2])
            if dist_to_goal < success_threshold:
                print(f"Goal reached at step {step}! Final position: ({current_state_real[0]:.4f}, {current_state_real[1]:.4f})")
                print(f"Distance to goal: {dist_to_goal:.4f}")
                break
            
            # Plan next action using the belief state
            # The action returned is already in real-world coordinates
            action = self.plan(belief, total_time=planning_time)
            
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
            # Prepare state for real_simulate - concat COM and current state
            sim_state = np.concatenate([np.array(self.true_com[0:2]), current_state_real], axis=0)
            
            # Execute action using real physics simulation to get real-world next state
            sim_result = real_simulate(sim_state, action, self.dataset)
            sim_result = np.array(sim_result)
            
            # Update the current state using the world-frame results from real simulation
            next_state_real = np.array([
                current_state_real[0] + sim_result[0],  # Add world x translation
                current_state_real[1] + sim_result[1],  # Add world y translation
                current_state_real[2] + sim_result[2],  # Add z translation
                sim_result[3]                          # Use the new absolute rotation
            ])
            
            # Update our belief state by applying the action
            # In NPTDPW we don't pass observations to update, unlike PFTDPW
            new_belief, reward = self.update(belief, action)
            
            # Calculate the exact state delta for tracking and visualization
            # For rotation, we need to handle the fact that sim_result[3] is absolute rotation
            state_delta = np.array([
                sim_result[0],  # x delta from simulation
                sim_result[1],  # y delta from simulation
                sim_result[2],  # z delta from simulation
                sim_result[3] - current_state_real[3]  # orientation delta (absolute rotation minus current)
            ])
            
            # Store the state transformation for visualization and analysis
            history['transformations'].append(state_delta.copy())
            history['outcomes'].append(next_state_real.copy())  # Store the full outcome state
            history['states'].append(next_state_real.copy())    # Update states history with cumulative effect
            
            # Update our accumulated state - this is crucial for composing multiple pushes
            current_state_real = next_state_real.copy()
            
            # Print information about the push effect and the resulting position
            # Convert angle to degrees only for display
            print(f"Step {step+1}: Push at {action_degrees:.2f}° → ")
            print(f"  Effect: Δx: {state_delta[0]:.4f}, Δy: {state_delta[1]:.4f}, Δz: {state_delta[2]:.4f}, Δθ: {state_delta[3]:.4f}")
            orientation_degrees = np.degrees(current_state_real[3])
            print(f"  New Position: ({current_state_real[0]:.4f}, {current_state_real[1]:.4f}), orientation: {current_state_real[3]:.4f} rad ({orientation_degrees:.2f}°)")
            
            # Update belief with the new state for the next planning iteration
            belief = new_belief
            
        
        # If we didn't reach the goal, print the final position
        if np.linalg.norm(current_state_real[:2] - self.goal_loc[:2]) >= success_threshold:
            print(f"Planning ended. Final position: ({current_state_real[0]:.4f}, {current_state_real[1]:.4f})")
            print(f"Distance to goal: {np.linalg.norm(current_state_real[:2] - self.goal_loc[:2]):.4f}")
        
        # Automatically visualize the planning results if requested
        if visualize:
            # Visualize the planning history
            self.visualize_planning(
                history=history,
                show=show_visualization,
                save_path=save_visualization_path)
        
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
    nptdpw = NPTDPW(
        model=model,
        dataset=validation_dataset,
        true_com=[0.5, 0.5, 0.3],
        alpha=0.5,
        const=0.1,
        search_depth=3,
        goal_loc=[10, 10, 0.0, 0.0],
        args=args
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
            block_length=0.2
        )

    # Set the path for saving the visualization
    save_visualization_path = None
    if visualize:
        save_visualization_path = os.path.join(vis_output_dir, f'planning_visualization_{timestamp}.png')

    # Execute planning with visualization settings
    example_plan = nptdpw.execute_planning_loop(
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

    main(args)
