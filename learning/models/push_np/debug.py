import argparse
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import torch
import math

# Import the necessary modules from your codebase
from learning.models.push_np.PFTDPW import PFTDPW, real_simulate
from learning.models.push_np.dataset import PushNPDataset
from learning.models.push_np.particle_filter import get_model
from learning.models.push_np.attention_push_np import APNPDecoder

def debug_do_singular_action(args, dataset):
    # Load dataset for transformation parameters
    
    # Load the model

    if os.path.exists(os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'best_model.pth')):  
        model = APNPDecoder(args, point_net_encoding_size=64, point_cloud=True, d_latents=0)
        model.load_state_dict(torch.load(os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'best_model.pth'))) 
    else:
        model = get_model(args)
        model.eval()
    
    # Create a PFTDPW instance
    true_com = [0.1, 0.1]  # Example center of mass
    pftdpw = PFTDPW(model, dataset, true_com)
    
    # Define a test state (center of mass coordinates)
    state = [0.1, 0.1]  # COM x, y
    initial_angle = 0.0  # Initial object angle
    
    # Create arrays to store results
    angles = np.arange(0, 1, 0.05)
    results_x = []
    results_y = []
    results_z = []
    results_theta = []
    
    # Run simulations for each angle
    print("\n=== Testing do_singular_action ===")
    print("Angle (deg) | Final X | Final Y | Final Z | Final Theta")
    print("-" * 60)
    
    for angle in angles:
        # Call do_singular_action
        result, distribution = pftdpw.do_singular_action(state, angle, initial_angle)
        
        # Store results
        results_x.append(result[0])
        results_y.append(result[1])
        results_z.append(result[2])
        results_theta.append(result[3])
        
        # Print results
        print(f"{angle:10.1f} | {result[0]:7.3f} | {result[1]:7.3f} | {result[2]:7.3f} | {result[3]:11.3f}")
        
        # Print distribution parameters
        if angle % 90 == 0:  # Only print for some angles to avoid too much output
            mean = distribution.mean.cpu().numpy()
            std = distribution.stddev.cpu().numpy()
            print(f"  Distribution mean: {mean}")
            print(f"  Distribution std: {std}")
    
    # Create output directory
    output_dir = os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'debug')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the results
    plt.figure(figsize=(15, 12))
    
    # Plot X position vs angle
    plt.subplot(4, 1, 1)
    plt.plot(angles, results_x, 'b-o')
    plt.title('X Position vs Push Angle (do_singular_action)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final X Position')
    plt.grid(True)
    
    # Plot Y position vs angle
    plt.subplot(4, 1, 2)
    plt.plot(angles, results_y, 'r-o')
    plt.title('Y Position vs Push Angle (do_singular_action)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final Y Position')
    plt.grid(True)
    
    # Plot Z position vs angle
    plt.subplot(4, 1, 3)
    plt.plot(angles, results_z, 'm-o')
    plt.title('Z Position vs Push Angle (do_singular_action)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final Z Position')
    plt.grid(True)
    
    # Plot theta vs angle
    plt.subplot(4, 1, 4)
    plt.plot(angles, results_theta, 'g-o')
    plt.title('Final Orientation vs Push Angle (do_singular_action)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final Orientation (degrees)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'do_singular_action_results.png'))
    
    # Also create a 2D trajectory plot
    plt.figure(figsize=(8, 8))
    plt.scatter(results_x, results_y, c=angles, cmap='hsv')
    plt.colorbar(label='Push Angle (degrees)')
    plt.title('Object Final Positions for Different Push Angles (do_singular_action)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'do_singular_action_trajectory.png'))

def debug_real_simulate(args, dataset):
    
    # Define a test object state
    # Format: [com_x, com_y, obj_x, obj_y, obj_z, obj_theta]
    # Ensure state has 6 elements as expected by real_simulate function
    com_x, com_y = 0.1, 0.1  # Center of mass
    obj_x, obj_y, obj_z = 0.1, 0.1, 0.0  # Object position
    obj_theta = 0.0  # Initial object orientation
    state = [com_x, com_y, obj_x, obj_y, obj_z, obj_theta]
    
    # Create arrays to store results
    angles = np.arange(0, 1, 0.05)
    results_x = []
    results_y = []
    results_z = []
    results_theta = []
    
    # Run simulations for each angle
    print("\n=== Testing real_simulate ===")
    print("Angle (deg) | Final X | Final Y | Final Z | Final Theta")
    print("-" * 60)
    
    for angle in angles:
        # Call real_simulate with the correct number of arguments
        result = real_simulate(state, angle, dataset)
        
        # Store results
        results_x.append(result[0])
        results_y.append(result[1])
        results_z.append(result[2])
        results_theta.append(result[3])
        
        # Print results
        print(f"{angle:10.1f} | {result[0]:7.3f} | {result[1]:7.3f} | {result[2]:7.3f} | {result[3]:11.3f}")
    
    # Create output directory
    output_dir = os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'debug')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the results
    plt.figure(figsize=(15, 12))
    
    # Plot X position vs angle
    plt.subplot(4, 1, 1)
    plt.plot(angles, results_x, 'b-o')
    plt.title('X Position vs Push Angle (real_simulate)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final X Position')
    plt.grid(True)
    
    # Plot Y position vs angle
    plt.subplot(4, 1, 2)
    plt.plot(angles, results_y, 'r-o')
    plt.title('Y Position vs Push Angle (real_simulate)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final Y Position')
    plt.grid(True)
    
    # Plot Z position vs angle
    plt.subplot(4, 1, 3)
    plt.plot(angles, results_z, 'm-o')
    plt.title('Z Position vs Push Angle (real_simulate)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final Z Position')
    plt.grid(True)
    
    # Plot theta vs angle
    plt.subplot(4, 1, 4)
    plt.plot(angles, results_theta, 'g-o')
    plt.title('Final Orientation vs Push Angle (real_simulate)')
    plt.xlabel('Push Angle (degrees)')
    plt.ylabel('Final Orientation (degrees)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_simulate_results.png'))
    
    # Also create a 2D trajectory plot
    plt.figure(figsize=(8, 8))
    plt.scatter(results_x, results_y, c=angles, cmap='hsv')
    plt.colorbar(label='Push Angle (degrees)')
    plt.title('Object Final Positions for Different Push Angles (real_simulate)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'real_simulate_trajectory.png'))

def compare_methods(args):
    # Run both methods
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset, args.instance)
    with open(os.path.join(dataset_path, 'train_dataset.pkl'), 'rb') as f: 
        train_dataset = pickle.load(f) 

    # Create output directory
    output_dir = os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'debug')
    os.makedirs(output_dir, exist_ok=True)
    
    debug_do_singular_action(args, train_dataset)
    debug_real_simulate(args, train_dataset)
    
    print(f"\nDebugging complete. Check the generated PNG files in {output_dir} for visualizations.")

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

    compare_methods(args)
