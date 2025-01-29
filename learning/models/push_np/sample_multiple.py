from matplotlib.patches import Ellipse
import seaborn as sns
from matplotlib import pyplot as plt, transforms
import pandas as pd
from tqdm import tqdm
import numpy as np 
import pickle
import argparse
import os
import torch 
from torch.utils.data import DataLoader 
from learning.models.push_np.dataset import PushNPDataset, collate_fn 
from queue import PriorityQueue
import random

from learning.models.push_np.attention_push_np import AttentionPushNP

def plot_push_predictions(all_positions, all_ground_truths, instance_path, 
                           filename='prediction_visualization.png'):
    """
    Randomly select and plot a datapoint with its ground truth and predicted positions
    
    Parameters:
    - all_positions: numpy array of predicted positions (shape: n_samples, l_samples, features)
    - all_ground_truths: numpy array of ground truth positions (shape: n_samples, features)
    - instance_path: directory path to save the plot
    - filename: name of the output file
    """
    # Randomly select a datapoint
    # random_index = random.randint(0, len(all_ground_truths) - 1)
    random_index = 0
    
    # Extract positions and ground truth for the selected datapoint
    positions = all_positions[random_index]  # Shape: (l_samples, features)
    ground_truth = all_ground_truths[random_index]  # Shape: (features,)
    
    # Create a figure with two subplots side by side
    plt.figure(figsize=(12, 5))
    
    # Plot 1: X-Y Position
    plt.subplot(1, 2, 1)
    
    # Plot ground truth
    plt.scatter(ground_truth[0], ground_truth[1], color='red', s=200, label='Ground Truth', marker='x')
    
    # Plot predicted positions
    plt.scatter(positions[:, 0], positions[:, 1], color='blue', alpha=0.7, label='Predicted Positions')
    
    plt.title('X-Y Position Prediction')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Set x and y axis limits centered on ground truth with 0.2 expansion
    x_center, y_center = ground_truth[0], ground_truth[1]
    x_width = max(np.ptp(positions[:, 0]), np.ptp(positions[:, 0][np.abs(positions[:, 0] - x_center) < 1])) + 0.4
    y_width = max(np.ptp(positions[:, 1]), np.ptp(positions[:, 1][np.abs(positions[:, 1] - y_center) < 1])) + 0.4
    
    plt.xlim(x_center - x_width/2, x_center + x_width/2)
    plt.ylim(y_center - y_width/2, y_center + y_width/2)
    
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Z Rotation
    plt.subplot(1, 2, 2)
    
    # Plot ground truth z rotation
    plt.scatter(ground_truth[2], ground_truth[3], color='red', s=200, label='Ground Truth', marker='x')
    
    # Plot predicted z rotations
    plt.scatter(positions[:, 2], positions[:, 3], color='blue', alpha=0.7, label='Predicted Rotations')
    
    plt.title('Z Position, Rotation Prediction')
    plt.xlabel('Z')
    plt.ylabel('Rotation')
    
    # Set z and rotation axis limits centered on ground truth with 0.2 expansion
    z_center, rot_center = ground_truth[2], ground_truth[3]
    z_width = max(np.ptp(positions[:, 2]), np.ptp(positions[:, 2][np.abs(positions[:, 2] - z_center) < 1])) + 0.4
    rot_width = max(np.ptp(positions[:, 3]), np.ptp(positions[:, 3][np.abs(positions[:, 3] - rot_center) < 1])) + 0.4
    
    plt.xlim(z_center - z_width/2, z_center + z_width/2)
    plt.ylim(rot_center - rot_width/2, rot_center + rot_width/2)
    
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save plot
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(instance_path, exist_ok=True)
    
    # Full path for saving the plot
    full_path = os.path.join(instance_path, filename)
    plt.savefig(full_path)
    plt.close()  # Close the figure to free up memory
    
    print(f"Plot saved to {full_path}")

def main(args): 
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset)
    instance_path = os.path.join(dataset_path, args.instance)
    training_args_path = os.path.join(instance_path, 'args.pkl') 

    with open(training_args_path, 'rb') as f: 
        training_args = pickle.load(f) 

    print(training_args) 
    model = AttentionPushNP(training_args)
    model.load_state_dict(torch.load(os.path.join(instance_path, 'best_model.pth')))

    if args.use_test_dataset:
        validation_dataset = PushNPDataset(os.path.join(dataset_path, "test_dataset.pkl"), training_args.num_points)
    else:
        with open(os.path.join(instance_path, "validation_dataset.pkl"), "rb") as handle:
            validation_dataset = pickle.load(handle)

    # with open(os.path.join(instance_path, "train_dataset.pkl"), "rb") as handle: 
    #     validation_dataset = pickle.load(handle) 

    data_loader = DataLoader(
        validation_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


    model = model.cuda() 
    model.eval() 
    all_ground_truths = []
    all_positions = []
    with torch.no_grad(): 
        for i, data in enumerate(tqdm(data_loader)): 
            # if args.num_points != -1 and args.num_points < 100 and i > 10:
            #     break
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            if training_args.use_obj_prop: 
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)
            else:
                obj_data = None
            # print(i, data["mass"][0], data["friction"][0], data["com"][0])  

            max_context_pushes = data["angle"].shape[1]
            n_context_pushes = torch.randint(1, max_context_pushes, (1,)).item()            
            if args.n_context != -1: 
                n_context_pushes = args.n_context 
            perm = torch.randperm(max_context_pushes)[:n_context_pushes]

            target_xs = torch.stack((data["angle"], data["push_velocities"], data["initials"]), dim=2)

            target_xs = torch.cat((
                    target_xs,
                    data["contact_points"],
                    data["normal_vector"],
                ), dim=2,)

            target_ys = torch.cat([data["final_position"], data["final_z_rotation"].unsqueeze(2)], dim=2) 
            # print(target_xs[0])
            # print(target_ys[0])

            context_xs = target_xs[:, perm] 
            context_ys = target_ys[:, perm] 
            # all_pushes_x.append(target_xs.reshape(-1, target_xs.shape[-1]).cpu().numpy())   
            # all_pushes_y.append(target_ys.reshape(-1, target_ys.shape[-1]).cpu().numpy())

            # target_xs = target_xs[:, :1]
            # target_ys = target_ys[:, :1]

            # print(target_xs)
            if torch.cuda.is_available():
                context_xs = context_xs.cuda().float()
                context_ys = context_ys.cuda().float()
                target_xs = target_xs.cuda().float()
                target_ys = target_ys.cuda().float()
                mesh_data = mesh_data.cuda().float()
                if training_args.use_obj_prop:
                    obj_data = obj_data.cuda().float()

            m = [] 
            for i in range(args.l_samples):
                total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy = model(context_xs, context_ys, target_xs, target_ys, mesh_data, obj_data, "validate") 
                ground_truths = target_ys 
                ground_truths = ground_truths.cpu().numpy()
                ground_truths = np.reshape(ground_truths, (-1, ground_truths.shape[-1]))
                if i == 0:
                    all_ground_truths.append(ground_truths)

                # print(entropy)

                mu_ = mu.cpu().numpy()
                mu_ = np.reshape(mu_, (-1, mu_.shape[-1]))

                m.append(mu_)

            
            all_positions.append(np.concatenate(m, axis=1)) 

            break 
    all_positions = np.concatenate(all_positions, axis=0) 
    all_ground_truths = np.concatenate(all_ground_truths, axis=0) 
    # test_array = np.ones(shape=(1)) 
    # pos_max = np.concatenate([validation_dataset.data["final_position_max"][: all_positions.shape[-1] - 1], test_array], axis=0)
    # test_array = np.zeros(shape=(1)) 
    # pos_min = np.concatenate([validation_dataset.data["final_position_min"][: all_positions.shape[-1] - 1], test_array], axis=0)

    # all_positions = (pos_max - pos_min) * all_positions + pos_min 
    # all_ground_truths = (pos_max - pos_min) * all_ground_truths + pos_min 
    # all_mu = (pos_max - pos_min) * all_mu + pos_min  
    # all_sigma = all_sigma * (pos_max - pos_min) ** 2
    # all_final_positions = all_final_positions * (pos_max - pos_min) + pos_min

    all_positions = np.reshape(all_positions, newshape=(all_positions.shape[0], args.l_samples, -1))
    plot_push_predictions(all_positions, all_ground_truths, instance_path)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', type=str, default='pushing')
    parser.add_argument('--instance', type=str, default='default')
    # parser.add_argument("--num-points", type=int, default=-1)
    parser.add_argument("--n-context", type=int, default=-1)
    parser.add_argument('--use-test-dataset', action='store_true')
    parser.add_argument('--l-samples', type=int, default=5)

    args = parser.parse_args()
    main(args)