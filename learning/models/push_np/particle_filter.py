import os
import pickle
import numpy as np 
import argparse
import torch
import scipy as sp
import wandb
import copy
import torch.nn.functional as F

from matplotlib.patches import Ellipse
from tqdm import tqdm
from learning.models.push_np.attention_push_np import APNPDecoder
from learning.models.push_np.dataset import PushNPDataset, collate_fn
from torch.utils.data import DataLoader
from learning.models.pointnet import PointNetRegressor
import matplotlib.pyplot as plt

class PushParticleFilter: 
    def __init__(self, model, dataset, num_particles): 
        """
        Initializes the PushParticleFilter with random particles and uniform log probabilities.

        Args:
            model (APNPDecoder): The trained APNPDecoder model.
            dataset (PushNPDataset): The dataset to perform particle filtering on.
            num_particles (int): Number of particles per data point.
        """
        self.particles = np.random.rand(len(dataset), num_particles, 2)  # Assuming 2D particles
        self.model = model 
        self.dataset = dataset 
        self.num_particles = num_particles 
        self.log_probs = np.log(np.ones(shape=(len(dataset), num_particles)) / num_particles)
        
    def do_step(self, idx, push_idx): 
        """
        Performs a single particle filter step for a specific data point and push index.

        Args:
            idx (int): Index of the data point in the dataset.
            push_idx (int): Index of the push action within the data point.
        """
        with torch.no_grad(): 
            log_weights = torch.zeros(self.num_particles).cuda()
            particles = self.particles[idx]
            particles = torch.from_numpy(particles).float().cuda()

            data = self.dataset[idx] 
            # mesh_data = None

            for i in range(self.num_particles): 
                obj_data = torch.cat((
                    torch.from_numpy(np.array([data["mass"]])),
                    torch.from_numpy(np.array([data["friction"]]))
                ), dim=0).cuda().float()
                
                obj_data = torch.cat((
                    obj_data.unsqueeze(0), 
                    particles[i].unsqueeze(0)
                ), dim=1)
                
                # print(obj_data.shape, torch.from_numpy(np.array([data["com"][2]])).cuda().float().unsqueeze(0).shape)
                obj_data = torch.cat((
                    obj_data, 
                    torch.from_numpy(np.array([data["com"][2]])).cuda().float().unsqueeze(0)
                ), dim=1) 
                # print(obj_data.shape)

                push_data = torch.stack((
                    torch.from_numpy(data["angle"]).cuda().float(),
                    torch.from_numpy(data["push_velocities"]).cuda().float(),
                    torch.from_numpy(data["initials"]).cuda().float(),
                ), dim=1) 
                
                if not self.args.no_contact:
                    push_data = torch.cat((
                        push_data,
                        torch.from_numpy(data["contact_points"]).squeeze().cuda().float(),
                        torch.from_numpy(data["normal_vector"]).squeeze().cuda().float()
                    ), dim=1)

                distributions = self.model(push_data.unsqueeze(0), None, None, None, obj_data)[0]
                ys = torch.cat((
                    torch.from_numpy(np.array([data["final_position"][push_idx]])), 
                    torch.from_numpy(np.array([data["final_z_rotation"][push_idx]])).unsqueeze(1)
                ), dim=1).cuda().float()
                
                log_weights[i] = distributions[0][push_idx].log_prob(ys).item()

        # Update log probabilities
        log_weights = log_weights.cpu().numpy() + self.log_probs[idx]
        total_probs = sp.special.logsumexp(log_weights) 
        log_probs = log_weights - total_probs 

        self.log_probs[idx] = log_probs 

    def do_number_steps(self, num_steps, amt=-1):
        """
        Performs multiple particle filter steps.

        Args:
            num_steps (int): Number of steps to perform.
        
        Returns:
            np.ndarray: Final normalized probabilities for all particles across the dataset.
        """
        for i in tqdm(range(num_steps), desc="Particle Filter Steps"): 
            for j in tqdm(range(len(self.dataset) if amt == -1 else amt), desc="Data Points", leave=False): 
                self.do_step(j, i)
        
        return np.exp(self.log_probs) 

    def do_number_steps_vectorized(self, num_steps, amt=-1, batch_size=64):
        """
        Vectorized approach to achieve the same effect as do_number_steps,
        but in fewer/larger batches to leverage the parallelism of PyTorch.
        
        Args:
            num_steps (int): How many pushes (steps) to perform.
            amt (int): Max number of data points to process (useful for debugging).
            batch_size (int): How many data points to process at once in each vectorized call.
                              Adjust to avoid GPU OOM; larger is often faster if memory allows.

        Returns:
            np.ndarray: Final normalized probabilities for all particles.
        """
        data_len = len(self.dataset) if amt == -1 else amt

        # We'll loop over each push_idx, but for that push_idx we process
        # the entire (or chunked) dataset in a *single model call* for *all particles*.
        for push_idx in tqdm(range(num_steps), desc="Particle Filter Steps"):

            # Process the dataset in batches
            for start_idx in tqdm(range(0, data_len, batch_size),
                                  desc="Data Points (Vectorized)",
                                  leave=False):
                end_idx = min(start_idx + batch_size, data_len)
                
                # ============= 1) Gather batched data across [start_idx, end_idx)  =============
                # We'll build:
                #   - a big obj_data_batch of shape (B * num_particles, D+3)
                #   - a big push_data_batch of shape (B * num_particles, #pushes, <push-dim>)
                #   - a big ys_batch of shape (B * num_particles, <final-dim>)
                
                # But note that "push_data" has shape (#pushes, <something>) for each item.
                # We'll replicate each data's push_data for each of its num_particles in the batch dimension.
                # Similarly for obj_data.  This yields a batch dimension of size (B * num_particles).

                batch_size_local = end_idx - start_idx

                # -- We'll accumulate these in Python lists first, then cat/stack them:
                obj_data_list = []
                push_data_list = []
                ys_list = []

                for data_index in range(start_idx, end_idx):
                    data = self.dataset[data_index]

                    # Construct the "static" portion of obj_data for this data item:
                    # shape (2,) => [mass, friction]
                    mass_friction = np.array([data["mass"], data["friction"]], dtype=np.float32)

                    # shape (1,) => [com_z]
                    com_z = np.array([data["com"][2]], dtype=np.float32)

                    # We'll also get the full set of (num_particles, D) from self.particles:
                    particles_np = self.particles[data_index]  # shape (num_particles, D)

                    # Turn them into torch tensors
                    particles_torch = torch.from_numpy(particles_np).float().cuda()  # (num_particles, D)

                    # Replicate mass_friction & com_z for each particle in this data item
                    mass_friction_torch = torch.from_numpy(mass_friction).unsqueeze(0).cuda()  # (1, 2)
                    com_z_torch         = torch.from_numpy(com_z).unsqueeze(0).cuda()          # (1, 1)

                    # Now cat mass_friction + each particle + com_z along dim=1
                    # shape => (num_particles, 2 + D + 1) = (num_particles, D+3)
                    # We do this by:
                    #   [ (1,2).expand(num_particles, -1)  cat  particles_torch  cat  (1,1).expand(num_particles,-1) ]
                    # but simpler approach is building them individually:
                    #   mass_friction -> shape (num_particles, 2)
                    #   particles_torch -> shape (num_particles, D)
                    #   com_z_torch -> shape (num_particles, 1)
                    # then cat them
                    repeated_mass_friction = mass_friction_torch.expand(particles_torch.size(0), -1)
                    repeated_com_z         = com_z_torch.expand(particles_torch.size(0), -1)

                    full_obj_data = torch.cat(
                        (repeated_mass_friction, particles_torch, repeated_com_z),
                        dim=1
                    )  # shape (num_particles, D+3)

                    obj_data_list.append(full_obj_data)

                    # 2) Construct push_data for this data item
                    #    shape (#pushes, <push_features>)
                    angle_torch = torch.from_numpy(data["angle"]).float().cuda()
                    push_vel_torch = torch.from_numpy(data["push_velocities"]).float().cuda()
                    initials_torch = torch.from_numpy(data["initials"]).float().cuda()
                    contact_pts_torch = torch.from_numpy(data["contact_points"]).squeeze().float().cuda()
                    normal_vec_torch = torch.from_numpy(data["normal_vector"]).squeeze().float().cuda()

                    # shape (#pushes, 3)
                    push_data_item = torch.stack(
                        (angle_torch, push_vel_torch, initials_torch), dim=1
                    )
                    # cat contact points + normal vector
                    if not self.args.no_contact: 
                        push_data_item = torch.cat(
                            (push_data_item, contact_pts_torch, normal_vec_torch),
                            dim=1
                        )  # shape (#pushes, 3 + cpoints + normals)

                    # Repeat push_data_item for each particle => (num_particles, #pushes, <push_features>)
                    # We do this by unsqueeze(0) to get (1, #pushes, <push_features>)
                    # then expand
                    push_data_item = push_data_item.unsqueeze(0).expand(
                        self.num_particles, -1, -1
                    )  # shape: (num_particles, #pushes, push_dim)

                    push_data_list.append(push_data_item)

                    # 3) Construct ys (the final ground truth) for push_idx
                    #    shape (1, final_dim) => e.g. (1, 3) or something
                    y_position = torch.from_numpy(np.array([data["final_position"][push_idx]]))
                    y_rotation = torch.from_numpy(np.array([data["final_z_rotation"][push_idx]])).unsqueeze(1)
                    ys_item = torch.cat((y_position, y_rotation), dim=1).float().cuda()
                    # shape (1, final_dim)

                    # That same target is repeated for each particle
                    ys_item = ys_item.expand(self.num_particles, -1)
                    # shape (num_particles, final_dim)

                    ys_list.append(ys_item)

                # Now concatenate along the batch dimension => B * num_particles
                # obj_data_batch => shape (B*num_particles, D+3)
                obj_data_batch  = torch.cat(obj_data_list, dim=0)  
                # push_data_batch => shape (B*num_particles, #pushes, push_dim)
                push_data_batch = torch.cat(push_data_list, dim=0)
                # ys_batch => shape (B*num_particles, final_dim)
                ys_batch        = torch.cat(ys_list, dim=0)
                # print(obj_data_batch.shape, push_data_batch.shape, ys_batch.shape)

                # ============= 2) Forward pass through the model =============
                with torch.no_grad():
                    # Suppose your model returns something shaped like
                    # (B*num_particles, #pushes, DistributionType) or a list-of-distributions
                    # Check how your model actually returns distributions.  
                    distributions = self.model(
                        push_data_batch,  # shape (B*num_particles, #pushes, push_dim)
                        None,
                        None,
                        None,
                        obj_data_batch    # shape (B*num_particles, D+3)
                    )[0]
                    # In your original code, you do `[0]` indexing, so adapt as needed:
                    # e.g., if your model returns a tuple, you might do: distributions = self.model(...)[0]
                    # We'll assume here it returns the distribution array itself.

                    # distributions => shape (B*num_particles, #pushes, ?)
                    # We want the distribution at index "push_idx" for each of the B*num_particles items
                    # so something like: distributions[:, push_idx].log_prob(...)
                    # Then log_prob(ys_batch) => shape (B*num_particles,)

                    # Evaluate the log_prob
                    log_weights_batch = torch.zeros(batch_size_local * self.num_particles).cuda()
                    # print(ys_batch.shape)
                    for i in range(len(distributions)):
                        log_weights_batch[i] = distributions[i][0].log_prob(ys_batch[i]) 
                    # log_weights_batch = distributions[:, push_idx].log_prob(ys_batch)
                    # shape (B*num_particles,)

                # ============= 3) Update self.log_probs exactly like do_step does =============
                # We must add these log-weights to the *correct* part of self.log_probs.
                # The tricky bit: we have them in a flattened (B*num_particles) order.
                # We can reshape them to (B, num_particles), then add to self.log_probs for each data item.

                log_weights_batch = log_weights_batch.view(batch_size_local, self.num_particles).cpu().numpy()

                # Update for each data item
                idx_counter = 0
                for data_index in range(start_idx, end_idx):
                    # old log_probs for this data item: shape (num_particles,)
                    old_lp = self.log_probs[data_index]
                    new_lp = old_lp + log_weights_batch[idx_counter]
                    idx_counter += 1

                    # normalize via logsumexp
                    total_lp  = sp.special.logsumexp(new_lp)
                    new_lp    = new_lp - total_lp
                    self.log_probs[data_index] = new_lp

        # Finally, exponentiate to get normalized probabilities
        return np.exp(self.log_probs)

    def plot_particles(self, index, file_name="particles.png"):
        """
        Plots the particles for a given data point index with their probabilities and the true COM.

        Args:
            index (int): The index of the data point to plot.
        """
        # Validate the index
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} is out of bounds for the dataset of size {len(self.dataset)}.")
        
        # Retrieve particles and log probabilities
        particles = self.particles[index]  # Shape: [num_particles, 2]
        log_probs = self.log_probs[index]  # Shape: [num_particles]
        
        # Convert log probabilities to probabilities
        probs = np.exp(log_probs)
        
        # Normalize probabilities to ensure they sum to 1
        probs /= probs.sum()
        
        # Retrieve the true COM for the data point
        data = self.dataset[index]
        if "com" not in data:
            raise KeyError(f"The data point at index {index} does not contain 'com' information.")
        
        true_com = data["com"][:2]  # Assuming 'com' has at least two dimensions (x, y)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Scatter plot of particles colored by probability
        scatter = plt.scatter(
            particles[:, 0], 
            particles[:, 1], 
            c=probs, 
            cmap='viridis', 
            alpha=0.6, 
            edgecolor='k',
            s=100  # Size of particles
        )
        
        # Add a color bar to indicate probability scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Probability', fontsize=12)
        
        # Plot the true COM
        plt.scatter(
            true_com[0], 
            true_com[1], 
            c='red', 
            marker='X', 
            s=200, 
            label='True COM'
        )
        
        # Enhance plot aesthetics
        plt.xlabel('X Position', fontsize=14)
        plt.ylabel('Y Position', fontsize=14)
        plt.title(f'Particle Filter Visualization for Data Index {index}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Show the plot
        # print(file_name)
        plt.savefig(file_name)
        plt.close()

    def save_particles(self, directory_name):
        with open(os.path.join(directory_name, "particles.pkl"), "wb") as f:
            pickle.dump(self.particles, f) 
        with open(os.path.join(directory_name, "log_probs.pkl"), "wb") as f:
            pickle.dump(self.log_probs, f)

    def load_particles(self, directory_name):
        with open(os.path.join(directory_name, "particles.pkl"), "rb") as f:
            self.particles = pickle.load(f)
        with open(os.path.join(directory_name, "log_probs.pkl"), "rb") as f:
            self.log_probs = pickle.load(f)

    def plot_results(self, save_path="push_prediction_results.png"):
        """
        Plots the distances between predicted final positions and ground truth final positions
        for each push, using the most likely particle's COM for predictions.
        
        Args:
            save_path (str): Path where to save the resulting plot.
        """
        distances = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(self.dataset)), desc="Calculating prediction distances"):
                # Get the most likely particle's position
                most_likely_idx = np.argmax(self.log_probs[idx])
                best_particle = self.particles[idx][most_likely_idx]
                
                data = self.dataset[idx]
                
                # Prepare input data for the model using the best particle
                obj_data = torch.cat((
                    torch.from_numpy(np.array([data["mass"]])),
                    torch.from_numpy(np.array([data["friction"]]))
                ), dim=0).cuda().float()
                
                obj_data = torch.cat((
                    obj_data.unsqueeze(0), 
                    torch.from_numpy(best_particle).cuda().float().unsqueeze(0)
                ), dim=1)
                
                obj_data = torch.cat((
                    obj_data, 
                    torch.from_numpy(np.array([data["com"][2]])).cuda().float().unsqueeze(0)
                ), dim=1)
                
                # Process each push in the sequence
                push_data = torch.stack((
                    torch.from_numpy(data["angle"]).cuda().float(),
                    torch.from_numpy(data["push_velocities"]).cuda().float(),
                    torch.from_numpy(data["initials"]).cuda().float(),
                ), dim=1)
                
                if not self.args.no_contact:
                    push_data = torch.cat((
                        push_data,
                        torch.from_numpy(data["contact_points"]).squeeze().cuda().float(),
                        torch.from_numpy(data["normal_vector"]).squeeze().cuda().float()
                    ), dim=1)
                    
                # Get model predictions
                distributions = self.model(push_data.unsqueeze(0), None, None, None, obj_data)[0]
                
                # Get ground truth final positions
                true_finals = torch.cat((
                    torch.from_numpy(data["final_position"]),
                    torch.from_numpy(data["final_z_rotation"]).unsqueeze(1)
                ), dim=1).cuda().float()
                
                # Calculate distances for each push
                for push_idx in range(len(distributions[0])):
                    pred_final = distributions[0][push_idx].mean
                    true_final = true_finals[push_idx]
                    
                    # Calculate Euclidean distance for position (x,y) only
                    distance = torch.norm(pred_final[:2] - true_final[:2], p=2).item()
                    distances.append(distance)
        
        # Convert to numpy array for easier manipulation
        distances = np.array(distances)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot distances over all pushes
        plt.plot(range(len(distances)), distances, 'b.', alpha=0.3, label='Individual Predictions')
        
        # Add a running average line
        window_size = min(50, len(distances))
        running_avg = np.convolve(distances, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(distances)), running_avg, 'r-', 
                linewidth=2, label=f'Running Average (window={window_size})')
        
        # Add horizontal line for mean
        mean_distance = np.mean(distances)
        plt.axhline(y=mean_distance, color='g', linestyle='--', 
                   label=f'Mean Distance: {mean_distance:.3f}m')
        
        # Enhance plot aesthetics
        plt.xlabel('Push Index', fontsize=12)
        plt.ylabel('Prediction Error (meters)', fontsize=12)
        plt.title('Push Prediction Errors Using Most Likely Particle COM', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add summary statistics as text
        stats_text = f'Summary Statistics:\n'
        stats_text += f'Mean Error: {mean_distance:.3f}m\n'
        stats_text += f'Median Error: {np.median(distances):.3f}m\n'
        stats_text += f'Std Dev: {np.std(distances):.3f}m\n'
        stats_text += f'Max Error: {np.max(distances):.3f}m\n'
        stats_text += f'Min Error: {np.min(distances):.3f}m'
        
        plt.text(0.98, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return summary statistics
        return {
            'mean_error': mean_distance,
            'median_error': np.median(distances),
            'std_error': np.std(distances),
            'max_error': np.max(distances),
            'min_error': np.min(distances)
        }
 
    def plot_prediction_distributions(self, num_samples=20, save_path="push_distributions.png"):
        """
        Plots prediction distributions and ground truth for randomly sampled pushes.
        Shows mean predictions and 2-sigma uncertainty ellipses.
        
        Args:
            num_samples (int): Number of random pushes to visualize
            save_path (str): Path where to save the resulting plot
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Randomly sample data points and push indices
        data_indices = np.random.randint(0, len(self.dataset), size=num_samples)
        
        # Create a colormap for different pushes
        colors = plt.cm.rainbow(np.linspace(0, 1, num_samples))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 15))
        
        with torch.no_grad():
            for i, (data_idx, color) in enumerate(zip(data_indices, colors)):
                # Get the most likely particle's position
                most_likely_idx = np.argmax(self.log_probs[data_idx])
                best_particle = self.particles[data_idx][most_likely_idx]
                
                data = self.dataset[data_idx]
                
                # Randomly select one push for this data point
                push_idx = np.random.randint(0, len(data["angle"]))
                
                # Prepare input data for the model
                obj_data = torch.cat((
                    torch.from_numpy(np.array([data["mass"]])),
                    torch.from_numpy(np.array([data["friction"]]))
                ), dim=0).cuda().float()
                
                obj_data = torch.cat((
                    obj_data.unsqueeze(0), 
                    torch.from_numpy(best_particle).cuda().float().unsqueeze(0)
                ), dim=1)
                
                obj_data = torch.cat((
                    obj_data, 
                    torch.from_numpy(np.array([data["com"][2]])).cuda().float().unsqueeze(0)
                ), dim=1)
                
                # Prepare push data for the selected push
                push_data = torch.stack((
                    torch.from_numpy(data["angle"][push_idx:push_idx+1]).cuda().float(),
                    torch.from_numpy(data["push_velocities"][push_idx:push_idx+1]).cuda().float(),
                    torch.from_numpy(data["initials"][push_idx:push_idx+1]).cuda().float(),
                ), dim=1)
                
                # print(push_data.shape, data["contact_points"][push_idx:push_idx+1].shape, data["normal_vector"][push_idx:push_idx+1].shape)
                push_data = torch.cat((
                    push_data,
                    torch.from_numpy(data["contact_points"][push_idx:push_idx+1]).cuda().float(),
                    torch.from_numpy(data["normal_vector"][push_idx:push_idx+1]).cuda().float()
                ), dim=1)
                
                # Get model prediction
                distributions = self.model(push_data.unsqueeze(0), None, None, None, obj_data)[0]
                pred_distribution = distributions[0][0]  # Get the first (and only) prediction
                
                # Get ground truth
                true_final = torch.cat((
                    torch.from_numpy(data["final_position"][push_idx]),
                    torch.from_numpy(data["final_z_rotation"][push_idx:push_idx+1])
                ), dim=0).cuda().float()
                
                # Extract mean and covariance for position (x,y)
                mean = pred_distribution.mean[:2].cpu().numpy()
                covar = pred_distribution.covariance_matrix[:2, :2].cpu().numpy()
                
                # Calculate eigenvectors and eigenvalues of covariance matrix
                eigenvals, eigenvecs = np.linalg.eigh(covar)
                
                # Calculate angle and standard deviations for the ellipse
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                std_devs = 2 * np.sqrt(eigenvals)  # 2 sigma ellipse
                
                # Plot prediction mean
                ax.scatter(mean[0], mean[1], color=color, marker='x', s=100, 
                         label=f'Push {i+1} Prediction')
                
                # Plot ground truth
                true_pos = true_final[:2].cpu().numpy()
                ax.scatter(true_pos[0], true_pos[1], color=color, marker='o', s=100,
                         facecolors='none')
                
                # Plot uncertainty ellipse
                ellipse = Ellipse(mean, width=std_devs[0], height=std_devs[1], 
                                angle=angle, facecolor='none', edgecolor=color, 
                                alpha=0.3)
                ax.add_patch(ellipse)
                
                # Draw line connecting prediction to ground truth
                ax.plot([mean[0], true_pos[0]], [mean[1], true_pos[1]], 
                       color=color, linestyle='--', alpha=0.3)
        
        # Enhance plot aesthetics
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Push Predictions with Uncertainty\n(x: predicted mean, o: ground truth)', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend for the first few pushes to avoid cluttering
        num_legend = min(5, num_samples)
        legend_elements = [
            plt.Line2D([0], [0], marker='x', color=colors[i], linestyle='',
                      label=f'Push {i+1} Prediction', markersize=10) 
            for i in range(num_legend)
        ]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='black',
                                        linestyle='', label='Ground Truth',
                                        fillstyle='none', markersize=10))
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Make axes equal to preserve ellipse shapes
        ax.set_aspect('equal')
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def calc_loss(distribs, target_ys):
    """
    Calculates the negative log-likelihood loss between the predicted distributions and target values.

    Args:
        distribs (list or torch.distributions.Distribution): Predicted distributions.
        target_ys (torch.Tensor): Ground truth target values.

    Returns:
        torch.Tensor: Calculated loss.
    """
    loss = 0
    for i in range(target_ys.shape[0]):
        for j in range(target_ys.shape[1]):
            loss -= distribs[i][j].log_prob(target_ys[i][j]) 
    return loss 

def compute_distance(distribs, target_ys):
    """
    Computes the average Euclidean distance between the predicted means and the ground truth final positions.

    Args:
        distribs (list or torch.distributions.Distribution): Predicted distributions.
        target_ys (torch.Tensor): Ground truth target values.

    Returns:
        float: Average distance.
    """
    dist = 0 
    for i in range(target_ys.shape[0]):
        for j in range(target_ys.shape[1]):
            # Compute the L2 distance between predicted mean and target
            dist += torch.norm(distribs[i][j].mean[:2] - target_ys[i][j][:2], p=2).item()
    return dist

def train_decoder(args, train_loader, test_loader): 
    """
    Trains the APNPDecoder model with PointNetRegressor on the provided training data,
    evaluates on the test data, and logs metrics to Weights and Biases (wandb).

    Args:
        args: Namespace containing training configurations.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the validation/test dataset.
    """
    
    # Initialize Weights and Biases
    wandb.init(
        project="pushing-neural-process", 
        config={
            "learning_rate": 1e-3,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,  # Assuming args has batch_size
            "dataset": args.dataset,
            "instance": args.instance,
            # Add other hyperparameters as needed
        },
        name=args.dataset + " " + args.instance,
        reinit=True  # Allow multiple runs in the same script
    )
    
    config = wandb.config

    # Initialize models
    point_cloud = PointNetRegressor(6, 64, 2, False, False).cuda().float()
    model = APNPDecoder(args, point_net_encoding_size=64, point_cloud=True, d_latents=0).cuda().float()
    
    # Log model architectures (optional)
    wandb.watch(model, log="all")
    wandb.watch(point_cloud, log="all")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Initialize best validation loss for checkpointing
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        # Training Phase
        model.train()
        point_cloud.train()
        
        running_train_loss = 0.0
        running_train_distance = 0.0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Training", leave=False):
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
            obj_data = torch.cat((obj_data, data["com"]), dim=1)
        
            max_context_pushes = data["angle"].shape[1]
            n_context_pushes = torch.randint(1, max_context_pushes, (1,)).item()            
            perm = torch.randperm(max_context_pushes)[:n_context_pushes]
        
            target_xs = torch.stack((data["angle"], data["push_velocities"], data["initials"]), dim=2)
        
            if not args.no_contact:
                target_xs = torch.cat((
                    target_xs,
                    data["contact_points"],
                    data["normal_vector"],
                ), dim=2,)
        
            target_ys = torch.cat([data["final_position"], data["final_z_rotation"].unsqueeze(2)], dim=2) 
        
            if torch.cuda.is_available():
                target_xs = target_xs.cuda().float()
                target_ys = target_ys.cuda().float()
                mesh_data = mesh_data.cuda().float()
                obj_data = obj_data.cuda().float()
            
            # Forward pass through PointNetRegressor
            mesh_vector, _ = point_cloud(mesh_data.transpose(1, 2)) 
        
            # Forward pass through APNPDecoder
            results = model(target_xs, None, None, mesh_vector, obj_data)
            distribs = results[0]
        
            # Compute loss
            loss = calc_loss(distribs, target_ys)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
            
            # Compute distance
            distance = compute_distance(distribs, target_ys)
            running_train_distance += distance
        
        avg_train_loss = running_train_loss / train_batches
        avg_train_distance = running_train_distance / train_batches
        
        # Validation Phase
        model.eval()
        point_cloud.eval()
        
        running_val_loss = 0.0
        running_val_distance = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Validation", leave=False):
                mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)
            
                max_context_pushes = data["angle"].shape[1]
                n_context_pushes = torch.randint(1, max_context_pushes, (1,)).item()            
                perm = torch.randperm(max_context_pushes)[:n_context_pushes]
            
                target_xs = torch.stack((data["angle"], data["push_velocities"], data["initials"]), dim=2)
            
                if not args.no_contact:
                    target_xs = torch.cat((
                            target_xs,
                            data["contact_points"],
                            data["normal_vector"],
                        ), dim=2,)
            
                target_ys = torch.cat([data["final_position"], data["final_z_rotation"].unsqueeze(2)], dim=2) 
            
                if torch.cuda.is_available():
                    target_xs = target_xs.cuda().float()
                    target_ys = target_ys.cuda().float()
                    mesh_data = mesh_data.cuda().float()
                    obj_data = obj_data.cuda().float()
                
                # Forward pass through PointNetRegressor
                mesh_vector, _ = point_cloud(mesh_data.transpose(1, 2)) 
            
                # Forward pass through APNPDecoder
                results = model(target_xs, None, None, mesh_vector, obj_data)
                distribs = results[0]
            
                # Compute loss
                loss = calc_loss(distribs, target_ys)
                
                running_val_loss += loss.item()
                val_batches += 1
                
                # Compute distance
                distance = compute_distance(distribs, target_ys)
                running_val_distance += distance
        
        avg_val_loss = running_val_loss / val_batches
        avg_val_distance = running_val_distance / val_batches
        
        # Log metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Train Distance": avg_train_distance,
            "Validation Distance": avg_val_distance
        })
        
        # Checkpointing: Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            save_path = os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'best_model.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_model_state, save_path)
            
            # Optionally, save the model as a wandb artifact
            wandb.save(save_path)
    return model 
    
def get_model(args): 
    print(args)
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset) 
    instance_path = os.path.join(dataset_path, args.instance) 
    train_data = os.path.join(dataset_path, "train_dataset.pkl")
    validation_data = os.path.join(dataset_path, "samegeo_test_dataset.pkl")
    
    if os.path.exists(instance_path) and os.path.exists(os.path.join(instance_path, 'best_model.pth')): 
        print(f'Instance {args.instance} found in dataset {args.dataset}.') 
        with open(os.path.join(instance_path, 'train_dataset.pkl'), 'rb') as f: 
            train_dataset = pickle.load(f) 
        with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'rb') as f: 
            validation_dataset = pickle.load(f)   
    else: 
        os.makedirs(instance_path, exist_ok=True)
        train_dataset = PushNPDataset(train_data, args.num_points)
        validation_dataset = PushNPDataset(validation_data, args.num_points) 

        with open(os.path.join(instance_path, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f) 
        with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'wb') as f:
            pickle.dump(validation_dataset, f)

    with open(os.path.join(instance_path, 'args.pkl'), 'wb') as f: 
        pickle.dump(args, f) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )   
    
    test_loader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    ) 

    return train_decoder(args, train_loader, test_loader) 
    
def main(args):
    """
    Main function to handle data loading, model training, particle filtering, and visualization.

    Args:
        args: Parsed command-line arguments.
    """
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset) 
    instance_path = os.path.join(dataset_path, args.instance) 
    train_data = os.path.join(dataset_path, "train_dataset.pkl")
    validation_data = os.path.join(dataset_path, "samegeo_test_dataset.pkl")
    
    if os.path.exists(instance_path) and os.path.exists(os.path.join(instance_path, 'best_model.pth')): 
        print(f'Instance {args.instance} found in dataset {args.dataset}.') 
        with open(os.path.join(instance_path, 'train_dataset.pkl'), 'rb') as f: 
            train_dataset = pickle.load(f) 
        with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'rb') as f: 
            validation_dataset = pickle.load(f)   
    else: 
        os.makedirs(instance_path, exist_ok=True)
        train_dataset = PushNPDataset(train_data, args.num_points)
        validation_dataset = PushNPDataset(validation_data, args.num_points) 

        with open(os.path.join(instance_path, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f) 
        with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'wb') as f:
            pickle.dump(validation_dataset, f)

    with open(os.path.join(instance_path, 'args.pkl'), 'wb') as f: 
        pickle.dump(args, f) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )   
    
    test_loader = DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )   

    # Train the model if the best_model.pth does not exist
    best_model_path = os.path.join(instance_path, 'best_model.pth')
    if not os.path.exists(best_model_path):
        train_decoder(args, train_loader, test_loader)

    # Load the best model
    model = APNPDecoder(args, point_net_encoding_size=64, point_cloud=True, d_latents=0).cuda().float() 
    model.load_state_dict(torch.load(best_model_path))

    # Initialize and run the particle filter
    particle_filter = PushParticleFilter(model, validation_dataset, num_particles=1000)
    particle_filter.plot_results(os.path.join(instance_path, 'push_prediction_results.png'))

    if os.path.exists(os.path.join(instance_path, 'particles.pkl')):
        particle_filter.load_particles(instance_path) 
    else: 
        particle_filter.do_number_steps(num_steps=5) 
        particle_filter.plot_results(os.path.join(instance_path, 'push_prediction_results_after_steps.png'))
        
        # Example: Plot particles for a specific data point index
        for index_to_plot in range(50):
            particle_filter.plot_particles(index_to_plot, os.path.join(instance_path, f"particles{index_to_plot}.png"))
        particle_filter.save_particles(instance_path) 
    particle_filter.plot_prediction_distributions(num_samples=20, save_path=os.path.join(instance_path, 'push_distributions.png'))
            
if __name__ == '__main__':
    # print("HELLO")
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
    main(args) 
