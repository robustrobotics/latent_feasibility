#!/usr/bin/env python3

import argparse
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import time
from tqdm.auto import tqdm

# Add the parent directory to sys.path
sys.path.append('/home/urop/code/latent_feasibility')

# Import our models
from learning.models.push_np.NPTDPW import NPTDPW
from learning.models.push_np.PFTDPW import PFTDPW
from learning.models.push_np.attention_push_np import AttentionPushNP
from learning.models.push_np.particle_filter import get_model
from learning.models.push_np.attention_push_np import APNPDecoder


class ExperimentRunner:
    """Framework for running and comparing experiments with NPTDPW and PFTDPW models."""
    
    def __init__(self, args):
        """Initialize the experiment runner with command-line arguments."""
        self.args = args
        self.results = {
            'nptdpw': [],
            'pftdpw': []
        }
        self.metrics = {
            'nptdpw': {},
            'pftdpw': {}
        }
        
        # Setup dataset paths
        self.dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset)
        
        # Setup separate instance paths for each model type
        self.nptdpw_instance_path = os.path.join(self.dataset_path, args.nptdpw_instance)
        self.pftdpw_instance_path = os.path.join(self.dataset_path, args.pftdpw_instance)
        
        # Use NPTDPW instance as default for output and common operations
        self.output_dir = os.path.join(self.nptdpw_instance_path, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Load datasets
        self._load_datasets()
        
        # Initialize models
        self.nptdpw_model = None
        self.pftdpw_model = None
        
    def _load_datasets(self):
        """Load training and validation datasets for both models."""
        # Load NPTDPW datasets
        print(f"Loading NPTDPW datasets from {self.nptdpw_instance_path}...")
        
        # Create the instance directory if it doesn't exist
        if not os.path.exists(self.nptdpw_instance_path):
            os.makedirs(self.nptdpw_instance_path)
            print(f"Warning: Created new directory {self.nptdpw_instance_path}")
        
        # Load NPTDPW validation dataset
        nptdpw_val_dataset_path = os.path.join(self.nptdpw_instance_path, 'validation_dataset.pkl')
        if os.path.exists(nptdpw_val_dataset_path):
            with open(nptdpw_val_dataset_path, 'rb') as f:
                self.nptdpw_validation_dataset = pickle.load(f)
            print(f"Loaded NPTDPW validation dataset with {len(self.nptdpw_validation_dataset)} samples")
        else:
            self.nptdpw_validation_dataset = None
            print(f"Warning: NPTDPW validation dataset not found at {nptdpw_val_dataset_path}")
            
        # Load PFTDPW datasets
        print(f"Loading PFTDPW datasets from {self.pftdpw_instance_path}...")
        
        # Create the instance directory if it doesn't exist
        if not os.path.exists(self.pftdpw_instance_path):
            os.makedirs(self.pftdpw_instance_path)
            print(f"Warning: Created new directory {self.pftdpw_instance_path}")
        
        # Load PFTDPW validation dataset
        pftdpw_val_dataset_path = os.path.join(self.pftdpw_instance_path, 'validation_dataset.pkl')
        if os.path.exists(pftdpw_val_dataset_path):
            with open(pftdpw_val_dataset_path, 'rb') as f:
                self.pftdpw_validation_dataset = pickle.load(f)
            print(f"Loaded PFTDPW validation dataset with {len(self.pftdpw_validation_dataset)} samples")
        else:
            self.pftdpw_validation_dataset = None
            print(f"Warning: PFTDPW validation dataset not found at {pftdpw_val_dataset_path}")
            
        # For backward compatibility, also set the generic validation_dataset
        self.validation_dataset = self.nptdpw_validation_dataset
    
    def initialize_nptdpw(self):
        """Initialize the NPTDPW model."""
        # print("Initializing NPTDPW model...")
        
        # Check if model weights exist
        best_model_path = os.path.join(self.nptdpw_instance_path, 'best_model.pth')
        model_args_path = os.path.join(self.nptdpw_instance_path, 'args.pkl')
        
        # Load model arguments if they exist
        if os.path.exists(model_args_path):
            with open(model_args_path, 'rb') as f:
                model_args = pickle.load(f)
        else:
            model_args = self.args
        
        # Create or load the model - NPTDPW uses the full AttentionPushNP model
        model = AttentionPushNP(model_args)
        if os.path.exists(best_model_path):
            # print(f"Loading NPTDPW model weights from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            # print(f"Warning: NPTDPW model weights not found at {best_model_path}")
            pass
        
        # Create NPTDPW instance
        self.nptdpw = NPTDPW(
            model=model,
            dataset=self.nptdpw_validation_dataset,
            true_com=self.args.true_com,
            search_depth=self.args.nptdpw_search_depth,
            goal_loc=self.args.goal_loc,
            alpha=self.args.nptdpw_alpha,
            beta=self.args.nptdpw_beta,
            const=self.args.nptdpw_const,
            args=self.args,
            table_type=self.args.table_type
        )
        
        # Initialize visualization table if needed
        if self.args.visualize:
            self.nptdpw.initialize_visualization_table(
                block_width=self.args.block_width,
                block_length=self.args.block_length
            )
    
    def initialize_pftdpw(self):
        """Initialize the PFTDPW model."""
        # print("Initializing PFTDPW model...")
        
        # Get paths to model weights and arguments
        best_model_path = os.path.join(self.pftdpw_instance_path, 'best_model.pth')
        model_args_path = os.path.join(self.pftdpw_instance_path, 'args.pkl')
        
        # Load model arguments if they exist
        if os.path.exists(model_args_path):
            # print(f"Loading PFTDPW model args from {model_args_path}")
            with open(model_args_path, 'rb') as f:
                model_args = pickle.load(f)
        else:
            # print(f"Warning: PFTDPW model args not found at {model_args_path}, using default args")
            model_args = self.args
            pass
            
        # Add required attributes that might be missing
        required_attrs = {
            'no_pointnet': False,
            'no_deterministic': False,
            'd_latents': 256,
            'use_mixture': False,
            'use_full_trajectory': False,
            'use_obj_prop': True,
            'attention_encoding': 256,
            'dropout': 0.1,
            'latent_samp': -1,
            'point_cloud': True,
            'guess_obj': False
        }
        
        # Set missing attributes
        # for attr, default_value in required_attrs.items():
        #     if not hasattr(model_args, attr):
        #         print(f"Adding missing attribute '{attr}' with default value {default_value}")
        #         setattr(model_args, attr, default_value)
        
        # Import the decoder component directly
        
        # Create a decoder with the specified arguments
        model = APNPDecoder(
            args=model_args, 
            d_latents=0, 
            point_cloud=True, 
            point_net_encoding_size=64
        ) 

        model.load_state_dict(torch.load(best_model_path))

        
        # Create PFTDPW instance
        self.pftdpw = PFTDPW(
            model=model,
            dataset=self.pftdpw_validation_dataset,
            true_com=self.args.true_com,
            search_depth=self.args.pftdpw_search_depth,
            num_particles=self.args.num_particles,
            goal_loc=self.args.goal_loc,
            alpha=self.args.pftdpw_alpha,
            beta=self.args.pftdpw_beta,
            const=self.args.pftdpw_const,
            args=self.args,
            table_type=self.args.table_type
        )
        
        # Initialize visualization table if needed
        if self.args.visualize:
            self.pftdpw.initialize_visualization_table(
                block_width=self.args.block_width,
                block_length=self.args.block_length
            )
    
    def run_nptdpw_experiment(self):
        """Run experiment with NPTDPW model."""
        # print("Running NPTDPW experiment...")
        
        # Ensure model is initialized
        if not hasattr(self, 'nptdpw'):
            self.initialize_nptdpw()
        
        # Setup visualization paths
        vis_path = None
        if self.args.visualize:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            vis_path = os.path.join(self.output_dir, f'nptdpw_planning_{timestamp}.png')
        
        # Run planning loop
        start_time = time.time()
        result = self.nptdpw.execute_planning_loop(
            initial_state=self.args.initial_state,
            max_steps=self.args.max_steps,
            planning_time=self.args.nptdpw_planning_time,
            success_threshold=self.args.success_threshold,
            visualize=self.args.visualize,
            show_visualization=False,
            save_visualization_path=vis_path
        )
        end_time = time.time()
        
        # Add basic statistics if NPTDPW doesn't have its own stats
        if 'stats' not in result:
            # Calculate final distance to goal
            final_distance = np.linalg.norm(
                np.array(result['states'][-1][:2]) - np.array(self.args.goal_loc[:2])
            ) if result['states'] else None
            
            # Calculate reward as negative distance to goal
            total_reward = -final_distance if final_distance is not None else 0.0
            
            # Check if the block fell off the table (roughly estimated)
            table_length = 10.0  # Default value
            table_width = 10.0   # Default value
            fell_off_table = False
            
            if result['states'] and len(result['states']) > 0:
                final_x = result['states'][-1][0]
                final_y = result['states'][-1][1]
                if final_x < 0 or final_x > table_length or final_y < 0 or final_y > table_width:
                    fell_off_table = True
            
            result['stats'] = {
                'total_execution_time': end_time - start_time,
                'final_distance': final_distance,
                'success': final_distance < self.args.success_threshold if final_distance is not None else False,
                'total_steps': len(result['states']) - 1 if 'states' in result else 0,
                'fell_off_table': fell_off_table,
                'total_reward': total_reward
            }
        
        # Calculate metrics
        self.metrics['nptdpw'] = {
            'execution_time': result['stats']['total_execution_time'] if 'total_execution_time' in result['stats'] else end_time - start_time,
            'final_state': result['states'][-1] if 'states' in result and result['states'] else None,
            'total_steps': result['stats']['total_steps'] if 'total_steps' in result['stats'] else len(result['states']) - 1 if 'states' in result else 0,
            'success': result['stats']['success'] if 'success' in result['stats'] else False,
            'final_distance': result['stats']['final_distance'] if 'final_distance' in result['stats'] else None,
            'fell_off_table': result['stats'].get('fell_off_table', False),
            'total_reward': result['stats'].get('total_reward', 0.0),
            'visualization_path': vis_path
        }
        
        # Save detailed results
        self.results['nptdpw'] = result
        
        return result
    
    def run_pftdpw_experiment(self):
        """Run experiment with PFTDPW model."""
        # print("Running PFTDPW experiment...")
        
        # Ensure model is initialized
        if not hasattr(self, 'pftdpw'):
            self.initialize_pftdpw()
        
        # Setup visualization paths
        vis_path = None
        if self.args.visualize:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            vis_path = os.path.join(self.output_dir, f'pftdpw_planning_{timestamp}.png')
        
        # Run planning loop
        start_time = time.time()
        result = self.pftdpw.execute_planning_loop(
            initial_state=self.args.initial_state,
            max_steps=self.args.max_steps,
            planning_time=self.args.pftdpw_planning_time,
            success_threshold=self.args.success_threshold,
            visualize=self.args.visualize,
            show_visualization=False,
            save_visualization_path=vis_path
        )
        end_time = time.time()
        
        # Extract comprehensive statistics
        stats = result['stats'] if 'stats' in result else {}
        
        # Calculate metrics using the comprehensive statistics
        self.metrics['pftdpw'] = {
            'execution_time': stats.get('total_execution_time', end_time - start_time),
            'final_state': result['states'][-1] if 'states' in result and result['states'] else None,
            'total_steps': stats.get('total_steps', len(result['states']) - 1 if 'states' in result else 0),
            'success': stats.get('success', False),
            'final_distance': stats.get('final_distance', None),
            'visualization_path': vis_path,
            'planning_time': stats.get('total_planning_time', 0.0),
            'cumulative_translation': stats.get('cumulative_translation', 0.0),
            'cumulative_rotation': stats.get('cumulative_rotation', 0.0),
            'avg_step_duration': np.mean(stats.get('step_durations', [0])) if stats.get('step_durations') else 0.0,
            'action_distribution': stats.get('action_counts', {}),
            'success_step': stats.get('success_step', None),
            'avg_particle_variance': np.mean(stats.get('particle_variance', [0])) if stats.get('particle_variance') else 0.0,
            'avg_effective_particles': np.mean(stats.get('effective_particles', [0])) if stats.get('effective_particles') else 0.0,
            'resampling_events': stats.get('resampling_events', 0)
        }
        
        # Save detailed results
        self.results['pftdpw'] = result
        
        return result
    
    def run_comparison(self):
        """Run both models and compare results."""
        # print("Running comparison experiment...")
        
        # Run both experiments
        pftdpw_result = self.run_pftdpw_experiment()
        nptdpw_result = self.run_nptdpw_experiment()
        
        # Compare and analyze results
        self._analyze_and_save_results()
        
        return {
            'nptdpw': nptdpw_result,
            'pftdpw': pftdpw_result,
            'metrics': self.metrics
        }
    
    def _analyze_and_save_results(self):
        """Analyze and save the experiment results."""
        # Extract metrics from the results
        nptdpw_stats = self.results.get('nptdpw', {}).get('stats', {})
        pftdpw_stats = self.results.get('pftdpw', {}).get('stats', {})
        
        # Success metrics
        nptdpw_success = nptdpw_stats.get('success', False)
        pftdpw_success = pftdpw_stats.get('success', False)
        
        # Distance metrics
        nptdpw_distance = nptdpw_stats.get('final_distance', float('inf'))
        pftdpw_distance = pftdpw_stats.get('final_distance', float('inf'))
        nptdpw_distance_cm = nptdpw_distance * 100.0
        pftdpw_distance_cm = pftdpw_distance * 100.0
        
        # Failure metrics
        nptdpw_fell = nptdpw_stats.get('fell_off_table', False)
        pftdpw_fell = pftdpw_stats.get('fell_off_table', False)
        
        # Reward metrics
        nptdpw_reward = nptdpw_stats.get('total_reward', 0.0)
        pftdpw_reward = pftdpw_stats.get('total_reward', 0.0)
        
        # Planning efficiency metrics
        nptdpw_steps = nptdpw_stats.get('total_steps', 0)
        pftdpw_steps = pftdpw_stats.get('total_steps', 0)
        nptdpw_time = nptdpw_stats.get('total_planning_time', 0.0)
        pftdpw_time = pftdpw_stats.get('total_planning_time', 0.0)
        
        # Print summary
        print("\n====== EXPERIMENT RESULTS SUMMARY ======\n")
        
        # Print success rate
        print("1. SUCCESS RATE:")
        print(f"NPTDPW: {nptdpw_success}")
        print(f"PFTDPW: {pftdpw_success}\n")
        
        # Print final distance
        print("2. FINAL DISPLACEMENT FROM TARGET (cm):")
        print(f"NPTDPW: {nptdpw_distance_cm:.2f} cm")
        print(f"PFTDPW: {pftdpw_distance_cm:.2f} cm\n")
        
        # Print failure rate
        print("3. FAILURE RATE (FELL OFF TABLE):")
        print(f"NPTDPW: {nptdpw_fell}")
        print(f"PFTDPW: {pftdpw_fell}\n")
        
        # Print total reward
        print("4. TOTAL REWARD:")
        print(f"NPTDPW: {nptdpw_reward:.2f}")
        print(f"PFTDPW: {pftdpw_reward:.2f}\n")
        
        # Print planning efficiency
        print("5. PLANNING EFFICIENCY:")
        nptdpw_time = nptdpw_stats.get('total_planning_time', 0.0)
        pftdpw_time = pftdpw_stats.get('total_planning_time', 0.0)
        print(f"NPTDPW: {nptdpw_steps} steps, {nptdpw_time:.2f} seconds planning time")
        print(f"PFTDPW: {pftdpw_steps} steps, {pftdpw_time:.2f} seconds planning time\n")
        
        print("======================================\n")
        
        if self.args.save_results:
            # Save the results to the output directory
            output_file = os.path.join(self.output_dir, 'experiment_results.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'results': self.results,
                    'metrics': {
                        'nptdpw': {
                            'success': nptdpw_success,
                            'final_distance': nptdpw_distance,
                            'fell_off_table': nptdpw_fell,
                            'total_reward': nptdpw_reward,
                            'total_steps': nptdpw_steps,
                            'total_planning_time': nptdpw_time
                        },
                        'pftdpw': {
                            'success': pftdpw_success,
                            'final_distance': pftdpw_distance,
                            'fell_off_table': pftdpw_fell,
                            'total_reward': pftdpw_reward,
                            'total_steps': pftdpw_steps,
                            'total_planning_time': pftdpw_time
                        }
                    }
                }, f)
            print(f"Results saved to {output_file}")
            
            # Save the paper figure format data
            self._save_paper_figure_data()
            
        # Create visualizations
        self._create_visualizations()
        
    def _save_paper_figure_data(self):
        """Save data in the format expected by the paper figure generation notebook."""
        try:
            # Create a dictionary for each model with the format expected by the figure notebook
            nptdpw_stats = self.results.get('nptdpw', {}).get('stats', {})
            pftdpw_stats = self.results.get('pftdpw', {}).get('stats', {})
            
            # Extract state history and planning time history
            nptdpw_history = self.results.get('nptdpw', {}).get('history', {})
            pftdpw_history = self.results.get('pftdpw', {}).get('history', {})
            
            # Extract planning times per step
            nptdpw_planning_times = nptdpw_stats.get('planning_times', [])
            pftdpw_planning_times = pftdpw_stats.get('planning_times', [])
            
            # Extract rewards per step
            nptdpw_rewards = nptdpw_stats.get('rewards', [])
            pftdpw_rewards = pftdpw_stats.get('rewards', [])
            
            # Extract distances to goal per step
            nptdpw_distances = nptdpw_stats.get('distances_to_goal', [])
            pftdpw_distances = pftdpw_stats.get('distances_to_goal', [])
            
            # Create cumulative planning times
            nptdpw_cumulative_times = []
            total_time = 0.0
            for t in nptdpw_planning_times:
                total_time += t
                nptdpw_cumulative_times.append(total_time)
                
            pftdpw_cumulative_times = []
            total_time = 0.0
            for t in pftdpw_planning_times:
                total_time += t
                pftdpw_cumulative_times.append(total_time)
            
            # Format data in the way expected by the figure notebook
            nptdpw_figure_data = {
                'compute time': nptdpw_cumulative_times,
                'reward': np.array(nptdpw_rewards) if nptdpw_rewards else np.array([]),
                'distance': np.array(nptdpw_distances) if nptdpw_distances else np.array([]),
                'plan length': np.array(range(1, len(nptdpw_planning_times) + 1)) if nptdpw_planning_times else np.array([])
            }
            
            pftdpw_figure_data = {
                'compute time': pftdpw_cumulative_times,
                'reward': np.array(pftdpw_rewards) if pftdpw_rewards else np.array([]),
                'distance': np.array(pftdpw_distances) if pftdpw_distances else np.array([]),
                'plan length': np.array(range(1, len(pftdpw_planning_times) + 1)) if pftdpw_planning_times else np.array([])
            }
            
            # Save the figure data
            figure_data_file = os.path.join(self.output_dir, 'figure_data.pkl')
            with open(figure_data_file, 'wb') as f:
                pickle.dump({
                    'nptdpw': nptdpw_figure_data,
                    'pftdpw': pftdpw_figure_data
                }, f)
            print(f"Paper figure data saved to {figure_data_file}")
            
        except Exception as e:
            print(f"Error saving paper figure data: {e}")
    
    def _create_visualizations(self):
        """Create visualizations of the experiment results."""
        if not hasattr(self, 'results') or 'nptdpw' not in self.results or 'pftdpw' not in self.results:
            print("No results available for visualization.")
            return
            
        # Create a directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Create visualizations
        self._visualize_distance_to_goal(vis_dir)
        self._visualize_key_metrics_comparison(vis_dir)
        self._visualize_action_distribution(vis_dir)
    
    def _visualize_distance_to_goal(self, output_dir):
        """Create a visualization of distance to goal over time."""
        try:
            # Get the distance to goal data from the results
            nptdpw_distances = []
            pftdpw_distances = []
            
            # Extract distances from NPTDPW results if available
            if self.results['nptdpw'] and 'stats' in self.results['nptdpw'] and 'distances_to_goal' in self.results['nptdpw']['stats']:
                nptdpw_distances = self.results['nptdpw']['stats']['distances_to_goal']
                
            # Extract distances from PFTDPW results if available
            if self.results['pftdpw'] and 'stats' in self.results['pftdpw'] and 'distances_to_goal' in self.results['pftdpw']['stats']:
                pftdpw_distances = self.results['pftdpw']['stats']['distances_to_goal']
            
            # Check if we have data to visualize
            if not nptdpw_distances and not pftdpw_distances:
                print("No distance data available for visualization.")
                return
                
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot NPTDPW distances if available
            if nptdpw_distances:
                plt.plot(range(len(nptdpw_distances)), nptdpw_distances, 'b-', label='NPTDPW')
                
            # Plot PFTDPW distances if available
            if pftdpw_distances:
                plt.plot(range(len(pftdpw_distances)), pftdpw_distances, 'r-', label='PFTDPW')
            
            # Add success threshold line
            plt.axhline(y=self.args.success_threshold, color='g', linestyle='--', label=f'Success Threshold ({self.args.success_threshold})')
                
            # Add labels and title
            plt.xlabel('Planning Step')
            plt.ylabel('Distance to Goal (m)')
            plt.title('Distance to Goal Over Time')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'distance_to_goal.png'))
            plt.close()
            
            print(f"Distance to goal visualization saved to {output_dir}/distance_to_goal.png")
        except Exception as e:
            print(f"Error creating distance to goal visualization: {e}")
    
    def _visualize_key_metrics_comparison(self, output_dir):
        """Create a bar chart comparing key metrics between NPTDPW and PFTDPW."""
        try:
            # Get stats for both models
            nptdpw_stats = self.results['nptdpw'].get('stats', {}) if self.results['nptdpw'] else {}
            pftdpw_stats = self.results['pftdpw'].get('stats', {}) if self.results['pftdpw'] else {}
            
            # Extract key metrics
            metrics = {
                'Final Distance (cm)': [
                    nptdpw_stats.get('final_distance', 0) * 100,  # Convert to cm
                    pftdpw_stats.get('final_distance', 0) * 100   # Convert to cm
                ],
                'Total Reward': [
                    nptdpw_stats.get('total_reward', 0),
                    pftdpw_stats.get('total_reward', 0)
                ],
                'Planning Time (s)': [
                    nptdpw_stats.get('total_planning_time', 0),
                    pftdpw_stats.get('total_planning_time', 0)
                ],
                'Total Steps': [
                    nptdpw_stats.get('total_steps', 0),
                    pftdpw_stats.get('total_steps', 0)
                ]
            }
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(metrics))
            width = 0.35
            
            # Plot bars
            rects1 = ax.bar(x - width/2, [metrics[m][0] for m in metrics], width, label='NPTDPW')
            rects2 = ax.bar(x + width/2, [metrics[m][1] for m in metrics], width, label='PFTDPW')
            
            # Add labels and title
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title('Key Performance Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(list(metrics.keys()))
            ax.legend()
            
            # Add value labels on bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            # Add indicators for success and failure
            y_pos = max([max(v) for v in metrics.values()]) * 1.1
            if nptdpw_stats.get('success', False):
                ax.annotate('SUCCESS', xy=(0 - width/2, y_pos), ha='center', va='bottom', color='green', fontweight='bold')
            if nptdpw_stats.get('fell_off_table', False):
                ax.annotate('FELL OFF', xy=(0 - width/2, y_pos), ha='center', va='bottom', color='red', fontweight='bold')
                
            if pftdpw_stats.get('success', False):
                ax.annotate('SUCCESS', xy=(0 + width/2, y_pos), ha='center', va='bottom', color='green', fontweight='bold')
            if pftdpw_stats.get('fell_off_table', False):
                ax.annotate('FELL OFF', xy=(0 + width/2, y_pos), ha='center', va='bottom', color='red', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
            plt.close()
            
            print(f"Key metrics comparison visualization saved to {output_dir}/metrics_comparison.png")
        except Exception as e:
            print(f"Error creating key metrics comparison visualization: {e}")
            
    def _visualize_action_distribution(self, output_dir):
        """Create a visualization of action distributions for both planners."""
        try:
            # Get action counts for both models
            nptdpw_stats = self.results['nptdpw'].get('stats', {}) if self.results['nptdpw'] else {}
            pftdpw_stats = self.results['pftdpw'].get('stats', {}) if self.results['pftdpw'] else {}
            
            nptdpw_actions = nptdpw_stats.get('action_counts', {})
            pftdpw_actions = pftdpw_stats.get('action_counts', {})
            
            if not nptdpw_actions and not pftdpw_actions:
                print("No action distribution data available for visualization.")
                return
                
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot NPTDPW action distribution
            if nptdpw_actions:
                labels = list(nptdpw_actions.keys())
                values = list(nptdpw_actions.values())
                ax1.bar(labels, values)
                ax1.set_title('NPTDPW Action Distribution')
                ax1.set_xlabel('Action')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
            else:
                ax1.text(0.5, 0.5, 'No NPTDPW action data', ha='center', va='center')
                ax1.set_title('NPTDPW Action Distribution')
            
            # Plot PFTDPW action distribution
            if pftdpw_actions:
                labels = list(pftdpw_actions.keys())
                values = list(pftdpw_actions.values())
                ax2.bar(labels, values)
                ax2.set_title('PFTDPW Action Distribution')
                ax2.set_xlabel('Action')
                ax2.set_ylabel('Count')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'No PFTDPW action data', ha='center', va='center')
                ax2.set_title('PFTDPW Action Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'action_distribution.png'))
            plt.close()
            
            print(f"Action distribution visualization saved to {output_dir}/action_distribution.png")
        except Exception as e:
            print(f"Error creating action distribution visualization: {e}")


def run_grid_experiments(args, planning_times=[1.0, 5.0, 10.0, 15.0], particle_counts=[20], num_trials=5):
    """Run a grid of experiments with different planning times and particle counts.
    
    Args:
        args: Command line arguments
        planning_times: List of planning times to evaluate
        particle_counts: List of particle counts to evaluate (only affects PFTDPW)
        num_trials: Number of trials to run for each configuration
    
    Returns:
        Dictionary containing results for paper figures
    """
    # Comment out print statements and use tqdm for progress tracking
    # print(f"Running {num_trials} trials for each configuration:")
    # print(f"- Planning times: {planning_times}")
    # print(f"- Particle counts: {particle_counts}")
    
    # Initialize results dictionaries
    # Store results by model and configuration
    nptdpw_data = {}
    pftdpw_data = {}
    
    # Also organize results by configuration for easy comparison
    results_by_config = {}
    
    # Store NPTDPW results for each planning time
    nptdpw_results_by_planning_time = {}
    
    # Lists to store trial data for backward compatibility
    nptdpw_trials = []
    pftdpw_trials = []
    for planning_time in planning_times:
        for particle_count in particle_counts:
            config_key = f"pt_{planning_time}_pc_{particle_count}"
            results_by_config[config_key] = {
                'planning_time': planning_time,
                'particle_count': particle_count,
                'nptdpw_trials': [],
                'pftdpw_trials': []
            }
    
    # Store original parameter values
    original_nptdpw_time = args.nptdpw_planning_time
    original_pftdpw_time = args.pftdpw_planning_time
    original_num_particles = args.num_particles
    
    # Ensure we're saving results
    args.save_results = True
    
    # Store original COM
    original_com = args.true_com.copy() if isinstance(args.true_com, list) else args.true_com
    
    # Iterate through trials (to average over) with tqdm progress bar
    for trial in tqdm(range(num_trials), desc="Trials", position=0):
        # Comment out print statements
        # print(f"\nStarting trial {trial+1}/{num_trials}...")
        
        # Randomly sample a center of mass for this trial
        # Keep z-coordinate the same, randomize x,y within reasonable bounds
        if isinstance(original_com, list) and len(original_com) >= 3:
            z_com = original_com[2] if len(original_com) > 2 else 0.0
            args.true_com = [
                np.random.uniform(0.2, 0.8),  # x between 0.2 and 0.8
                np.random.uniform(0.2, 0.8),  # y between 0.2 and 0.8
                z_com                         # keep original z
            ]
            # print(f"Using randomly sampled COM: {args.true_com}")
        
        # First, run NPTDPW for each planning time (since it doesn't use particles)
        # print("\nRunning NPTDPW for all planning times (independent of particle counts)...")
        for planning_time in tqdm(planning_times, desc="NPTDPW planning times", position=1, leave=False):
            # print(f"Running NPTDPW with planning time {planning_time}s...")
            
            # Update planning time for NPTDPW
            args.nptdpw_planning_time = planning_time
            
            # Create a runner for this specific configuration
            # Note: Particle count doesn't matter for NPTDPW, but we'll use the default
            runner = ExperimentRunner(args)
            
            # Run NPTDPW and store results
            nptdpw_result = runner.run_nptdpw_experiment()
            nptdpw_results_by_planning_time[planning_time] = nptdpw_result
            
        # Now run PFTDPW for each combination of planning time and particle count
        for particle_count in tqdm(particle_counts, desc="PFTDPW particle counts", position=1, leave=False):
            # print(f"\nUsing particle count: {particle_count} for PFTDPW")
            # Set particle count
            args.num_particles = particle_count
                
            trial_nptdpw_data = {
                'compute time': [],
                'reward': [],
                'distance': [],
                'plan length': [],
                'success': [],
                'particles': particle_count  # Store particle count for reference
            }
            
            trial_pftdpw_data = {
                'compute time': [],
                'reward': [],
                'distance': [],
                'plan length': [],
                'success': [],
                'particles': particle_count  # Store particle count for reference
            }
            
            # Run PFTDPW for all planning times with this particle count
            for planning_time in tqdm(planning_times, desc="PFTDPW planning times", position=2, leave=False):
                # print(f"Running PFTDPW with planning time {planning_time}s...")
                
                # Update planning time for PFTDPW
                args.pftdpw_planning_time = planning_time
                
                # Create a runner for this specific configuration
                runner = ExperimentRunner(args)
                
                # Run PFTDPW and get results
                pftdpw_result = runner.run_pftdpw_experiment()
                
                # Get the stored NPTDPW result for this planning time
                nptdpw_result = nptdpw_results_by_planning_time[planning_time]
                
                # Extract key metrics
                nptdpw_stats = nptdpw_result.get('stats', {})
                pftdpw_stats = pftdpw_result.get('stats', {})
                
                # Add data for this planning time
                # Create the proper config key for this specific planning time and particle count
                config_key = f"pt_{planning_time}_pc_{particle_count}"
                
                # Need to make sure the data structure exists for this configuration
                if config_key not in results_by_config:
                    results_by_config[config_key] = {
                        'planning_time': planning_time,
                        'particle_count': particle_count,
                        'nptdpw_trials': [],
                        'pftdpw_trials': []
                    }
                
                # Store per-planning time, per-particle count results
                # This creates a separate entry for each unique configuration
                current_nptdpw_result = {
                    'compute time': planning_time,
                    'reward': nptdpw_stats.get('total_reward', 0.0),
                    'distance': nptdpw_stats.get('final_distance', float('inf')),
                    'plan length': nptdpw_stats.get('total_steps', 0),
                    'success': 1 if nptdpw_stats.get('success', False) else 0,
                    'particle_count': particle_count,
                    'trial': trial,
                    'map': args.table_type
                }
                
                current_pftdpw_result = {
                    'compute time': planning_time,
                    'reward': pftdpw_stats.get('total_reward', 0.0),
                    'distance': pftdpw_stats.get('final_distance', float('inf')),
                    'plan length': pftdpw_stats.get('total_steps', 0),
                    'success': 1 if pftdpw_stats.get('success', False) else 0,
                    'particle_count': particle_count,
                    'trial': trial,
                    'map': args.table_type
                }
                
                # Add each individual result to the configuration's trial list
                results_by_config[config_key]['nptdpw_trials'].append(current_nptdpw_result)
                results_by_config[config_key]['pftdpw_trials'].append(current_pftdpw_result)
                
                # Add current data point to the trial data collection
                trial_nptdpw_data['compute time'].append(planning_time)
                trial_nptdpw_data['reward'].append(nptdpw_stats.get('total_reward', 0.0))
                trial_nptdpw_data['distance'].append(nptdpw_stats.get('final_distance', float('inf')))
                trial_nptdpw_data['plan length'].append(nptdpw_stats.get('total_steps', 0))
                trial_nptdpw_data['success'].append(1 if nptdpw_stats.get('success', False) else 0)
                
                trial_pftdpw_data['compute time'].append(planning_time)
                trial_pftdpw_data['reward'].append(pftdpw_stats.get('total_reward', 0.0))
                trial_pftdpw_data['distance'].append(pftdpw_stats.get('final_distance', float('inf')))
                trial_pftdpw_data['plan length'].append(pftdpw_stats.get('total_steps', 0))
                trial_pftdpw_data['success'].append(1 if pftdpw_stats.get('success', False) else 0)
            
            # Convert lists to numpy arrays for the completed trial
            for key in trial_nptdpw_data:
                if key != 'compute time' and key != 'particles':
                    trial_nptdpw_data[key] = np.array(trial_nptdpw_data[key])
                    
            for key in trial_pftdpw_data:
                if key != 'compute time' and key != 'particles':
                    trial_pftdpw_data[key] = np.array(trial_pftdpw_data[key])
            
            # We've already added the individual planning time results in the inner loop
            # Now just add the completed trial data to the global lists for backward compatibility
            nptdpw_trials.append(trial_nptdpw_data)
            pftdpw_trials.append(trial_pftdpw_data)
    
    # Restore original parameters
    args.nptdpw_planning_time = original_nptdpw_time
    args.pftdpw_planning_time = original_pftdpw_time
    args.num_particles = original_num_particles
    
    # Save the multi-trial data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('learning/data/pushing', args.dataset, 'paper_figures', f'trials_{timestamp}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the overall results (backward compatibility)
    overall_file = os.path.join(output_dir, 'planning_time_trials.pkl')
    with open(overall_file, 'wb') as f:
        pickle.dump({
            'nptdpw_trials': nptdpw_trials,
            'pftdpw_trials': pftdpw_trials,
            'planning_times': planning_times,
            'particle_counts': particle_counts,
            'num_trials': num_trials
        }, f)
    
    # Save per-configuration results (for more detailed analysis)
    config_file = os.path.join(output_dir, 'grid_experiment_results.pkl')
    with open(config_file, 'wb') as f:
        pickle.dump({
            'results_by_config': results_by_config,
            'planning_times': planning_times,
            'particle_counts': particle_counts,
            'num_trials': num_trials,
            'original_com': original_com
        }, f)
        
    print("\nGrid experiments completed:")
    print(f"- Overall results saved to: {overall_file}")
    print(f"- Configuration results saved to: {config_file}")
    
    return {
        'nptdpw_trials': nptdpw_trials,
        'pftdpw_trials': pftdpw_trials,
        'results_by_config': results_by_config
    }

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description="Run comparison experiments between NPTDPW and PFTDPW models")
    
    # Dataset and instance arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--nptdpw-instance', type=str, required=True, help='Instance to load for NPTDPW')
    parser.add_argument('--pftdpw-instance', type=str, required=True, help='Instance to load for PFTDPW')
    
    # For backward compatibility - if only --instance is provided, use it for both models
    parser.add_argument('--instance', type=str, help='Instance to load for both models (backward compatibility)')
    
    # General experiment parameters
    parser.add_argument('--initial-state', type=tuple, default=(1.0, 1.0, 0.0, 0.0), 
                        help='Initial state as (x, y, angle_sin, angle_cos)')
    parser.add_argument('--goal-loc', type=list, default=[3.0, 3.0, 0.0, 0.0], 
                        help='Goal location as [x, y, angle_sin, angle_cos]')
    parser.add_argument('--max-steps', type=int, default=30, 
                        help='Maximum number of planning steps')
    parser.add_argument('--success-threshold', type=float, default=0.10, 
                        help='Distance threshold for success')
    parser.add_argument('--true-com', type=list, default=[0.7, 0.3, 0.0], 
                        help='True center of mass as [x, y, z]')
    parser.add_argument('--visualize', action='store_true', 
                        help='Enable visualization')
    parser.add_argument('--block-width', type=float, default=0.07, 
                        help='Block width for visualization')
    parser.add_argument('--block-length', type=float, default=0.2, 
                        help='Block length for visualization')
    
    # NPTDPW specific parameters
    parser.add_argument('--nptdpw-search-depth', type=int, default=2, 
                        help='Search depth for NPTDPW')
    parser.add_argument('--nptdpw-alpha', type=float, default=0.5, 
                        help='Alpha parameter for NPTDPW')
    parser.add_argument('--nptdpw-beta', type=float, default=0.5, 
                        help='Beta parameter for NPTDPW')
    parser.add_argument('--nptdpw-const', type=float, default=0.1, 
                        help='Const parameter for NPTDPW')
    parser.add_argument('--nptdpw-planning-time', type=float, default=3.0, 
                        help='Planning time per step for NPTDPW')
    
    # PFTDPW specific parameters
    parser.add_argument('--pftdpw-search-depth', type=int, default=2, 
                        help='Search depth for PFTDPW')
    parser.add_argument('--num-particles', type=int, default=20, 
                        help='Number of particles for PFTDPW')
    parser.add_argument('--pftdpw-alpha', type=float, default=0.5, 
                        help='Alpha parameter for PFTDPW')
    parser.add_argument('--pftdpw-beta', type=float, default=0.5, 
                        help='Beta parameter for PFTDPW')
    parser.add_argument('--pftdpw-const', type=float, default=0.1, 
                        help='Const parameter for PFTDPW')
    parser.add_argument('--pftdpw-planning-time', type=float, default=1.0, 
                        help='Planning time per step for PFTDPW')
    
    # Add parameters required by the models but that will have default values set
    parser.add_argument('--no-contact', action='store_true', help='Disable contact information')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--no-deterministic', action='store_true', help='Disable deterministic encoder')
    parser.add_argument('--d-latents', type=int, default=8, help='Dimension of latent space')
    parser.add_argument('--attention-encoding', type=int, default=512, help='Dimension of attention encoding')
    parser.add_argument('--use-full-trajectory', action='store_true', help='Use full trajectory in model')
    parser.add_argument('--use-obj-prop', action='store_true', help='Use object properties')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--no-pointnet', action='store_true', help='Disable PointNet')
    parser.add_argument('--use-mixture', action='store_true', help='Use mixture of Gaussians')
    parser.add_argument('--save_results', action='store_true', help='Save results to file')
    
    # Add visualization arguments
    # parser.add_argument('--visualize', action='store_true', help='Enable visualization during experiments')
    parser.add_argument('--show-plot', action='store_true', help='Show the visualization plots (vs. just saving them)')
    parser.add_argument('--vis-resolution', type=float, default=0.01, help='Resolution for visualization grid')
    parser.add_argument('--table-type', type=str, choices=['box', 'ring', 'beam'], default='box',
                      help='Type of table to use for visualization (box, ring, or beam)')
    
    # Run specific test scenario
    parser.add_argument('--run-mode', type=str, choices=['nptdpw', 'pftdpw', 'both'], default='both',
                        help='Which model(s) to run')
    
    # Grid experiment options
    parser.add_argument('--run-grid-experiments', action='store_true',
                      help='Run a grid of experiments with different planning times and particle counts')
    parser.add_argument('--planning-times', type=str, default='1.0,5.0,10.0,15.0',
                      help='Comma-separated list of planning times to evaluate')
    parser.add_argument('--particle-counts', type=str, default='20',
                      help='Comma-separated list of particle counts to evaluate (only affects PFTDPW)')
    parser.add_argument('--num-trials', type=int, default=5,
                      help='Number of trials to run for each configuration')
    
    # For backward compatibility
    parser.add_argument('--run-planning-time-experiments', action='store_true',
                      help='Alias for --run-grid-experiments')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert string tuple to actual tuple if needed
    if isinstance(args.initial_state, str):
        args.initial_state = eval(args.initial_state)
    
    # Run grid experiments if requested (including backward compatibility with planning time experiments)
    if args.run_grid_experiments or args.run_planning_time_experiments:
        planning_times = [float(t) for t in args.planning_times.split(',')]
        particle_counts = [int(p) for p in args.particle_counts.split(',')]
        
        print("Starting grid experiments with:")
        print(f"- Planning times: {planning_times}")
        print(f"- Particle counts: {particle_counts}")
        print(f"- Trials per configuration: {args.num_trials}")
        print(f"- Total combinations: {len(planning_times) * len(particle_counts) * args.num_trials}")
        
        # Note: PFTDPW visualization will work as configured in the execute_planning_loop method
        # considering we've added visualization in a previous session
        
        run_grid_experiments(args, 
                           planning_times=planning_times, 
                           particle_counts=particle_counts, 
                           num_trials=args.num_trials)
    else:
        # Run regular experiment
        runner = ExperimentRunner(args)
        
        if args.run_mode == 'nptdpw':
            runner.run_nptdpw_experiment()
        elif args.run_mode == 'pftdpw':
            runner.run_pftdpw_experiment()
        else:  # 'both'
            runner.run_comparison()


if __name__ == "__main__":
    main()
