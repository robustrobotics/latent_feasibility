#!/usr/bin/env python3

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from typing import Dict, List, Any, Optional

def load_experiment_data(file_path: str) -> Dict:
    """Load experiment data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_planning_time_experiment(data_file: str, output_dir: Optional[str] = None):
    """Plot results from planning time experiment.
    
    Args:
        data_file: Path to the planning_time_trials.pkl file
        output_dir: Directory to save plots, defaults to same directory as data_file
    """
    # Load data
    print(f"Loading data from {data_file}")
    data = load_experiment_data(data_file)
    
    # Extract data
    nptdpw_trials = data['nptdpw_trials']
    pftdpw_trials = data['pftdpw_trials']
    planning_times = data.get('planning_times', [])
    particle_counts = data.get('particle_counts', [20])  # Default particle count if not present
    num_trials = data.get('num_trials', 1)
    
    # Print debugging information
    print(f"Found {num_trials} trials with planning times: {planning_times}")
    print(f"Found particle counts: {particle_counts}")
    
    # If planning_times is empty, extract them from the data
    if not planning_times:
        # Extract unique planning times from trial data
        all_times = set()
        for trial in nptdpw_trials:
            if 'compute time' in trial:
                all_times.update(trial['compute time'])
        for trial in pftdpw_trials:
            if 'compute time' in trial:
                all_times.update(trial['compute time'])
        planning_times = sorted(all_times)
        print(f"Extracted planning times from trial data: {planning_times}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(data_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics vs planning time
    metrics = ['reward', 'distance', 'plan length', 'success']
    
    # Markers and colors for different configurations
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Group NPTDPW data (doesn't vary by particle count, but we'll include for consistency)
        nptdpw_by_particles = {}
        nptdpw_by_particles[0] = {'values': {}, 'label': 'NPTDPW'}
        
        # Group PFTDPW data by particle count
        pftdpw_by_particles = {}
        
        # Initialize the data structures
        for time in planning_times:
            nptdpw_by_particles[0]['values'][time] = []

        # Collect NPTDPW values
        for trial in nptdpw_trials:
            for i, time in enumerate(trial['compute time']):
                if metric in trial and i < len(trial[metric]):
                    nptdpw_by_particles[0]['values'][time].append(trial[metric][i])
        
        # Collect PFTDPW values grouped by particle count
        for trial in pftdpw_trials:
            # Get particle count for this trial
            particle_count = trial.get('particles', 20)  # Default to 20 if not specified
            
            # Initialize if we haven't seen this particle count before
            if particle_count not in pftdpw_by_particles:
                pftdpw_by_particles[particle_count] = {
                    'values': {},
                    'label': f'PFTDPW (p={particle_count})'
                }
                for time in planning_times:
                    pftdpw_by_particles[particle_count]['values'][time] = []
            
            # Add the data points for this trial
            for i, time in enumerate(trial['compute time']):
                if metric in trial and i < len(trial[metric]):
                    pftdpw_by_particles[particle_count]['values'][time].append(trial[metric][i])
        
        # Plot one line for NPTDPW
        config = nptdpw_by_particles[0]
        means = [np.mean(config['values'][t]) if config['values'][t] else np.nan for t in planning_times]
        stds = [np.std(config['values'][t]) if len(config['values'][t]) > 1 else 0 for t in planning_times]
        plt.errorbar(planning_times, means, yerr=stds, marker='o', color=colors[0],
                     label=config['label'], capsize=5)
                     
        # Plot one line for each PFTDPW particle count
        for i, (p_count, config) in enumerate(sorted(pftdpw_by_particles.items())):
            means = [np.mean(config['values'][t]) if config['values'][t] else np.nan for t in planning_times]
            stds = [np.std(config['values'][t]) if len(config['values'][t]) > 1 else 0 for t in planning_times]
            plt.errorbar(planning_times, means, yerr=stds, 
                         marker=markers[min(i+1, len(markers)-1)], 
                         color=colors[min(i+1, len(colors)-1)],
                         label=config['label'], capsize=5)
        
        plt.xlabel('Planning Time (s)')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Planning Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{metric}_vs_time.png')
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        plt.close()
    
    # Plot combined metrics
    plt.figure(figsize=(12, 10))
    
    # Markers and colors for different configurations
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5))]  # Different line styles
    
    # Create 2x2 subplot grid for 4 metrics
    for subplot_idx, metric in enumerate(metrics):
        plt.subplot(2, 2, subplot_idx+1)
        
        # Group NPTDPW data (doesn't vary by particle count)
        nptdpw_by_particles = {}
        nptdpw_by_particles[0] = {'values': {}, 'label': 'NPTDPW'}
        
        # Group PFTDPW data by particle count
        pftdpw_by_particles = {}
        
        # Initialize the data structures
        for time in planning_times:
            nptdpw_by_particles[0]['values'][time] = []

        # Collect NPTDPW values
        for trial in nptdpw_trials:
            for i, time in enumerate(trial['compute time']):
                if metric in trial and i < len(trial[metric]):
                    nptdpw_by_particles[0]['values'][time].append(trial[metric][i])
        
        # Collect PFTDPW values grouped by particle count
        for trial in pftdpw_trials:
            particle_count = trial.get('particles', 20)  # Default to 20 if not specified
            
            if particle_count not in pftdpw_by_particles:
                pftdpw_by_particles[particle_count] = {
                    'values': {},
                    'label': f'PFTDPW (p={particle_count})'
                }
                for time in planning_times:
                    pftdpw_by_particles[particle_count]['values'][time] = []
            
            for i, time in enumerate(trial['compute time']):
                if metric in trial and i < len(trial[metric]):
                    pftdpw_by_particles[particle_count]['values'][time].append(trial[metric][i])
        
        # Plot one line for NPTDPW
        config = nptdpw_by_particles[0]
        means = [np.mean(config['values'][t]) if config['values'][t] else np.nan for t in planning_times]
        plt.plot(planning_times, means, marker='o', color=colors[0], linestyle='-',
                 label=config['label'])
                     
        # Plot one line for each PFTDPW particle count
        for i, (p_count, config) in enumerate(sorted(pftdpw_by_particles.items())):
            means = [np.mean(config['values'][t]) if config['values'][t] else np.nan for t in planning_times]
            plt.plot(planning_times, means, 
                     marker=markers[min(i+1, len(markers)-1)], 
                     color=colors[min(i+1, len(colors)-1)],
                     linestyle=linestyles[min(i, len(linestyles)-1)],
                     label=config['label'])
        
        plt.xlabel('Planning Time (s)')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Planning Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    # Save combined figure
    output_file = os.path.join(output_dir, 'combined_metrics.png')
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()

def plot_figure_data(data_file: str, output_dir: Optional[str] = None):
    """Plot results from a single experiment figure data file.
    
    Args:
        data_file: Path to the figure_data.pkl file
        output_dir: Directory to save plots, defaults to same directory as data_file
    """
    # Load data
    print(f"Loading data from {data_file}")
    data = load_experiment_data(data_file)
    
    # Extract data
    nptdpw_data = data['nptdpw']
    pftdpw_data = data['pftdpw']
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(data_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics over compute time
    metrics = ['reward', 'distance', 'plan length']
    for metric in metrics:
        if metric in nptdpw_data and metric in pftdpw_data:
            plt.figure(figsize=(10, 6))
            
            # Plot NPTDPW data
            nptdpw_times = nptdpw_data['compute time']
            nptdpw_values = nptdpw_data[metric]
            plt.plot(nptdpw_times, nptdpw_values, 'o-', label='NPTDPW')
            
            # Plot PFTDPW data
            pftdpw_times = pftdpw_data['compute time']
            pftdpw_values = pftdpw_data[metric]
            plt.plot(pftdpw_times, pftdpw_values, 's-', label='PFTDPW')
            
            plt.xlabel('Compute Time (s)')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} vs Compute Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            output_file = os.path.join(output_dir, f'{metric}_vs_time.png')
            plt.savefig(output_file)
            print(f"Saved {output_file}")
            plt.close()
    
    # Plot state trajectories (if available)
    if 'states' in nptdpw_data and 'states' in pftdpw_data:
        plt.figure(figsize=(10, 6))
        
        nptdpw_states = nptdpw_data['states']
        pftdpw_states = pftdpw_data['states']
        
        # Extract x, y coordinates if states are arrays with at least 2 elements
        if len(nptdpw_states) > 0 and isinstance(nptdpw_states[0], (list, np.ndarray)) and len(nptdpw_states[0]) >= 2:
            nptdpw_x = [state[0] for state in nptdpw_states]
            nptdpw_y = [state[1] for state in nptdpw_states]
            plt.plot(nptdpw_x, nptdpw_y, 'o-', label='NPTDPW Trajectory')
        
        if len(pftdpw_states) > 0 and isinstance(pftdpw_states[0], (list, np.ndarray)) and len(pftdpw_states[0]) >= 2:
            pftdpw_x = [state[0] for state in pftdpw_states]
            pftdpw_y = [state[1] for state in pftdpw_states]
            plt.plot(pftdpw_x, pftdpw_y, 's-', label='PFTDPW Trajectory')
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Block Trajectories')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # Save figure
        output_file = os.path.join(output_dir, 'trajectories.png')
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        plt.close()

def find_latest_experiment_data(base_dir: str, file_pattern: str = '*planning_time_trials.pkl') -> str:
    """Find the latest experiment data file based on directory timestamp."""
    # List all directories in the base_dir that match the format trials_YYYYMMDD_HHMMSS
    dirs = glob.glob(os.path.join(base_dir, 'paper_figures', 'trials_*'))
    
    if not dirs:
        raise FileNotFoundError(f"No experiment directories found in {base_dir}")
    
    # Sort by directory name (which contains timestamp)
    latest_dir = sorted(dirs)[-1]
    
    # Find the file in the latest directory
    files = glob.glob(os.path.join(latest_dir, file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} found in {latest_dir}")
    
    return files[0]

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument('--data-file', type=str, help='Path to experiment data file')
    parser.add_argument('--dataset', type=str, help='Dataset name to find latest experiment')
    parser.add_argument('--output-dir', type=str, help='Directory to save plots')
    parser.add_argument('--planning-time-experiments', action='store_true', 
                        help='Plot planning time experiment results')
    parser.add_argument('--single-experiment', action='store_true',
                        help='Plot single experiment figure data')
    
    args = parser.parse_args()
    
    # If data file not specified, try to find latest based on dataset
    if args.data_file is None and args.dataset:
        try:
            base_dir = os.path.join('learning/data/pushing', args.dataset)
            if args.planning_time_experiments:
                args.data_file = find_latest_experiment_data(base_dir, '*planning_time_trials.pkl')
            elif args.single_experiment:
                args.data_file = find_latest_experiment_data(base_dir, '*figure_data.pkl')
            else:
                # Try planning time trials first, then figure data
                try:
                    args.data_file = find_latest_experiment_data(base_dir, '*planning_time_trials.pkl')
                    args.planning_time_experiments = True
                except FileNotFoundError:
                    args.data_file = find_latest_experiment_data(base_dir, '*figure_data.pkl')
                    args.single_experiment = True
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    if args.data_file is None:
        print("Error: No data file specified. Use --data-file or --dataset")
        return
    
    # Determine which plot function to use based on file name or args
    if args.planning_time_experiments or 'planning_time_trials.pkl' in args.data_file:
        plot_planning_time_experiment(args.data_file, args.output_dir)
    elif args.single_experiment or 'figure_data.pkl' in args.data_file:
        plot_figure_data(args.data_file, args.output_dir)
    else:
        print(f"Error: Could not determine plot type for file {args.data_file}")
        print("Specify --planning-time-experiments or --single-experiment")

if __name__ == "__main__":
    main()
