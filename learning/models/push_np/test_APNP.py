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

from learning.models.push_np.attention_push_np import AttentionPushNP

def find_similar(x, y, mu, instance_path, all_sigma): 
    pq = PriorityQueue() 
    for j in range(7, len(x)):
        norm = np.linalg.norm(x[j] - x[6])
        pq.put((-norm, j))
        if pq.qsize() > 10: 
            pq.get()
    # print(x[0], y[0], mu[0])
    # for best, i in list(pq.queue):
    #     print(x[i], y[i], mu[i])
    
    pq.put((6, 0))

    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(list(pq.queue))))
    for idx, (best, i) in enumerate(list(pq.queue)):
        plt.scatter(y[i][0], y[i][1], color=colors[idx], label=f'Predicted {i}' if i == 0 else "")
        plt.scatter(mu[i][0], mu[i][1], color=colors[idx], marker='x', s=100, label=f'True {i}' if i == 0 else "")
        plt.plot([y[i][0], mu[i][0]], 
                [y[i][1], mu[i][1]], 
                color=colors[idx], linestyle='--', alpha=0.5)
        cov = np.array([[all_sigma[i, 0, 0], all_sigma[i, 0, 1]],
                        [all_sigma[i, 1, 0], all_sigma[i, 1, 1]]])
        confidence_ellipse(y[i][0], y[i][1], cov, plt.gca(), n_std=2.0, edgecolor=colors[idx], alpha=0.3)

    plt.savefig(os.path.join(instance_path, "similar_points.png"))
    exit()


def confidence_ellipse(x, y, cov, ax, n_std=2.0, facecolor="none", **kwargs):
    # cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(np.mean(x), np.mean(y))
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def expectation_distance(mean, lower_tril, target, n_samples=1000): 
    """
    Compute a monte-carlo sample of the expectation of the distance between the distribution and the target.
    """
    distance_total = 0
    distribution = np.random.multivariate_normal(mean, lower_tril @ lower_tril.T, size=n_samples)
    for i in range(n_samples): 
        sample = distribution[i]
        distance_total += np.linalg.norm(sample - target) / n_samples 
    return distance_total 

def main(args):
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset)
    instance_path = os.path.join(dataset_path, args.instance)
    training_args_path = os.path.join(instance_path, 'args.pkl') 

    with open(training_args_path, 'rb') as f: 
        training_args = pickle.load(f) 

    print(training_args) 
    model = AttentionPushNP(training_args)
    model.load_state_dict(torch.load(os.path.join(instance_path, 'best_model.pth')))


    # if args.use_test_dataset:
    #     validation_dataset = PushNPDataset(os.path.join(dataset_path, "test_dataset.pkl"), training_args.num_points)
    # else:
    #     with open(os.path.join(instance_path, "validation_dataset.pkl"), "rb") as handle:
    #         validation_dataset = pickle.load(handle)

    with open(os.path.join(instance_path, "train_dataset.pkl"), "rb") as handle: 
        validation_dataset = pickle.load(handle) 

    data_loader = DataLoader(
        validation_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    all_final_positions = []
    all_sigma = []
    all_mu = []
    all_entropy = [] 

    all_pushes_x = [] 
    all_pushes_y = []   

    model = model.cuda() 
    model.eval() 
    with torch.no_grad(): 
        for i, data in tqdm(enumerate(data_loader)): 
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
            all_pushes_x.append(target_xs.reshape(-1, target_xs.shape[-1]).cpu().numpy())   
            all_pushes_y.append(target_ys.reshape(-1, target_ys.shape[-1]).cpu().numpy())

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

            total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy = model(context_xs, context_ys, target_xs, target_ys, mesh_data, obj_data, "validate") 
            ground_truths = target_ys 
            ground_truths = ground_truths.cpu().numpy()
            ground_truths = np.reshape(ground_truths, (-1, ground_truths.shape[-1]))
            all_final_positions.append(ground_truths)

            # print(entropy)
            all_entropy.append(entropy.cpu().numpy()) 

            sigma = np.reshape(sigma.cpu().numpy(), (-1, mu.shape[-1], mu.shape[-1]))
            all_sigma.append(sigma)

            mu_ = mu.cpu().numpy()
            mu_ = np.reshape(mu_, (-1, mu_.shape[-1]))

            all_mu.append(mu_)
            # print(mu_.shape, sigma.shape, ground_truths.shape)

    all_pushes_x = np.concatenate(all_pushes_x, axis=0)
    all_pushes_y = np.concatenate(all_pushes_y, axis=0)
    all_mu = np.concatenate(all_mu, axis=0)
    all_sigma = np.concatenate(all_sigma, axis=0)
    all_final_positions = np.concatenate(all_final_positions, axis=0)
    # print(all_pushes_x[0:10])

    # find_similar(all_pushes_x, all_pushes_y, all_mu, instance_path, all_sigma)    

    average_entropy = np.mean(all_entropy) 
    print(f"Average entropy: {average_entropy}") 
    
    num_points = args.num_points if args.num_points != -1 else all_mu.shape[0] 

    # print(all_final_positions.shape, all_mu.shape, all_sigma.shape)

    # Rescale back the data to original scale
    test_array = np.ones(shape=(1)) 
    pos_max = np.concatenate([validation_dataset.data["final_position_max"][: all_mu.shape[-1] - 1], test_array], axis=0)
    test_array = np.zeros(shape=(1)) 
    pos_min = np.concatenate([validation_dataset.data["final_position_min"][: all_mu.shape[-1] - 1], test_array], axis=0)

    all_mu = (pos_max - pos_min) * all_mu + pos_min  
    all_sigma = all_sigma * (pos_max - pos_min) ** 2
    all_final_positions = all_final_positions * (pos_max - pos_min) + pos_min
# 
    perm = np.random.permutation(len(all_mu))[:num_points]
    all_mu = all_mu[perm]
    all_sigma = all_sigma[perm]
    all_final_positions = all_final_positions[perm]

    # all_mu = all_mu[num_points:2*num_points]
    # all_sigma = all_sigma[num_points:2*num_points]
    # all_final_positions = all_final_positions[num_points:2*num_points] 


    norms = np.zeros(all_sigma.shape[0])
    eigen_vectors = np.zeros(all_sigma.shape)
    eigen_values = np.zeros(all_sigma.shape[:-1])
    projected_variances = np.zeros(all_sigma.shape[:-1])
    mean_distance = np.zeros(all_sigma.shape[0])

    # print(all_sigma.shape)
    # print(mean_distance.shape) 

    for i in tqdm(range(all_sigma.shape[0])):
        norms[i] = np.linalg.norm(all_sigma[i], "fro")
        mean_distance[i] = expectation_distance(all_mu[i], all_sigma[i], all_final_positions[i]) 
        # print(np.linalg.eig(all_sigma[i]))
        eigen_values[i], eigen_vectors[i] = np.linalg.eig(all_sigma[i])

        for j in range(all_sigma.shape[-1]):
            projected_variances[i, j] = (
                eigen_values[i][j] * (eigen_vectors[i][j, 0] ** 2)
                + eigen_values[i][1] * (eigen_vectors[i][j, 1] ** 2)
                + eigen_values[i][2] * (eigen_vectors[i][j, 2] ** 2)
            )

    average_distance = np.mean(mean_distance) 
    print(f"Average distance: {average_distance}") 
    predicted_orientations = all_mu[:, 3]  # Assuming the orientation is the 4th column
    ground_truth_orientations = all_final_positions[:, 3]
    df = pd.DataFrame(
        {
            "mean_distnace": mean_distance,
            "pred_x": all_mu[:, 0],
            "pred_y": all_mu[:, 1],
            "pred_z": all_mu[:, 2],
            "true_x": all_final_positions[:, 0],
            "true_y": all_final_positions[:, 1],
            "true_z": all_final_positions[:, 2],
            "norm": norms,
            "var_x": projected_variances[:, 0],
            "var_y": projected_variances[:, 1],
            "var_z": projected_variances[:, 2],
            "pred_orientation": predicted_orientations,
            "true_orientation": ground_truth_orientations,
        }
    )

    # print(predicted_orientations)
    # print(ground_truth_orientations) 

    if num_points > 100: 
        sns.set_theme()
        sns.scatterplot(data=df, x="pred_x", y="pred_y", hue="norm")
        plt.savefig(os.path.join(instance_path, "xy_norm.png"))
        plt.close()
        sns.scatterplot(data=df, x="pred_x", y="pred_y", hue="pred_z")
        plt.savefig(os.path.join(instance_path, "xyz.png"))
        plt.close()
        sns.scatterplot(data=df, x="var_x", y="var_y", hue="var_z")
        plt.savefig(os.path.join(instance_path, "variances.png"))
        plt.close()

    if num_points <= 100: 
        # Create a colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, num_points))

        # Plot predicted vs true positions with lines and covariance ellipses
        plt.figure(figsize=(12, 8))
        for i in range(num_points):
            plt.scatter(df['pred_x'].iloc[i], df['pred_y'].iloc[i], color=colors[i], label=f'Predicted {i}' if i == 0 else "")
            plt.scatter(df['true_x'].iloc[i], df['true_y'].iloc[i], color=colors[i], marker='x', s=100, label=f'True {i}' if i == 0 else "")
            plt.plot([df['pred_x'].iloc[i], df['true_x'].iloc[i]], 
                    [df['pred_y'].iloc[i], df['true_y'].iloc[i]], 
                    color=colors[i], linestyle='--', alpha=0.5)
            
            # Add covariance ellipse
            if not training_args.regression:
                cov = np.array([[all_sigma[i, 0, 0], all_sigma[i, 0, 1]],
                                [all_sigma[i, 1, 0], all_sigma[i, 1, 1]]])
                confidence_ellipse(df['pred_x'].iloc[i], df['pred_y'].iloc[i], cov, plt.gca(), n_std=2.0, edgecolor=colors[i], alpha=0.3)

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f"Predicted vs True Positions with Covariance Ellipses (n={num_points}, context={args.n_context})")
        plt.legend(['Predicted', 'True'])
        plt.savefig(os.path.join(instance_path, "predicted_vs_true_positions_with_ellipses.png"))
        plt.close()


        sns.set_theme()
        # xy_norm plot with individual covariance ellipses
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x="var_x", y="var_y", hue="norm", ax=ax)
        for i in range(len(df)):
            cov = np.array([[all_sigma[i, 0, 0], all_sigma[i, 0, 1]],
                            [all_sigma[i, 1, 0], all_sigma[i, 1, 1]]])
            confidence_ellipse(df['var_x'].iloc[i], df['var_y'].iloc[i], cov, ax, edgecolor='red', alpha=0.1)
        plt.savefig(os.path.join(instance_path, "xy_norm_with_individual_ellipses.png"))
        plt.close()

        # xyz plot with individual covariance ellipses
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = sns.scatterplot(data=df, x="var_x", y="var_y", hue="var_z", ax=ax)
        for i in range(len(df)):
            cov = np.array([[all_sigma[i, 0, 0], all_sigma[i, 0, 1]],
                            [all_sigma[i, 1, 0], all_sigma[i, 1, 1]]])
            confidence_ellipse(df['var_x'].iloc[i], df['var_y'].iloc[i], cov, ax, edgecolor='red', alpha=0.1)
        plt.savefig(os.path.join(instance_path, "xyz_with_individual_ellipses.png"))
        plt.close()

        plt.figure(figsize=(12, 8))
        orientation_variance = all_sigma[:, 3, 3]
        orientation_std = np.sqrt(orientation_variance)

        plt.scatter(range(num_points), df['pred_orientation'], label='Predicted Orientation', alpha=0.7)
        plt.scatter(range(num_points), df['true_orientation'], label='True Orientation', alpha=0.7)

        # Plot prediction intervals
        for i in range(num_points):
            plt.plot([i, i], [df['pred_orientation'].iloc[i], df['true_orientation'].iloc[i]], 
                    color='gray', linestyle='--', alpha=0.5)
            
            # Add 2 standard deviation bars for each prediction
            if not training_args.regression:
                plt.plot([i, i], 
                        [df['pred_orientation'].iloc[i] - 2*orientation_std[i], 
                        df['pred_orientation'].iloc[i] + 2*orientation_std[i]],
                        color='red', alpha=0.5)

        plt.xlabel('Sample Index')
        plt.ylabel('Orientation (radians)')
        plt.title(f"Predicted vs True Orientations with 2σ intervals (n={num_points}, context={args.n_context})")
        plt.legend()

        # Adjust y-axis limits to show all error bars
        y_min = min(df['pred_orientation'].min(), df['true_orientation'].min()) - 4*orientation_std.max()
        y_max = max(df['pred_orientation'].max(), df['true_orientation'].max()) + 4*orientation_std.max()
        plt.ylim(y_min, y_max)

        plt.savefig(os.path.join(instance_path, "predicted_vs_true_orientations_with_intervals.png"))
        plt.close()
        print("Model loaded successfully!")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', type=str, default='pushing')
    parser.add_argument('--instance', type=str, default='default')
    parser.add_argument("--num-points", type=int, default=-1)
    parser.add_argument("--n-context", type=int, default=-1)
    parser.add_argument('--use-test-dataset', action='store_true')

    args = parser.parse_args()
    main(args)
    