from matplotlib import transforms
from matplotlib.patches import Ellipse
from tqdm import tqdm
import argparse
import os
import torch
from learning.models.push_np.dataset import collate_fn
from learning.models.push_np.push_neural_process import PushNP
import pickle
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    print(args)

    instance_path = os.path.join(
        "learning", "data", "pushing", args.dataset_name, args.instance
    )

    state_dict = torch.load(os.path.join(instance_path, "best_model.pth"))

    with open(os.path.join(instance_path, "args.pkl"), "rb") as handle:
        training_args = pickle.load(handle)

    features = {
        "angle",
        "mesh",
        "friction",
        "com",
        "mass",
        "final_position",
        "trajectory_data",
        "push_velocities",
        "normals",
        "contact_points",
        "normal_vector",
    }

    if not training_args.use_obj_properties:
        features.remove("mass")
        features.remove("com")
        features.remove("friction")

    print(training_args)

    model = PushNP(features, 3, d_latents=5)
    model = model.cuda()

    model.load_state_dict(state_dict)
    model.eval()

    with open(os.path.join(instance_path, "validation_dataset.pkl"), "rb") as handle:
        validation_dataset = pickle.load(handle)

    data_loader = DataLoader(
        validation_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_mu = []
    all_sigma = []
    all_final_positions = []
    with torch.no_grad():
        for val_data in tqdm(data_loader):

            mesh_data = torch.cat((val_data["mesh"], val_data["normals"]), dim=2)

            if training_args.use_obj_properties:
                obj_data = torch.stack((val_data["mass"], val_data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, val_data["com"]), dim=1)
            else:
                obj_data = (None, None, None)

            push_data = torch.stack(
                (val_data["angle"], val_data["push_velocities"]), dim=2
            )
            # dimensions = val_data["angle"].shape
            push_data = torch.cat(
                (push_data, val_data["contact_points"], val_data["normal_vector"]),
                dim=2,
            )

            if torch.cuda.is_available():
                mesh_data = mesh_data.cuda().float()
                push_data = push_data.cuda().float()
                if training_args.use_obj_properties:
                    obj_data = obj_data.cuda().float()
                val_data["final_position"] = val_data["final_position"].cuda().float()
                val_data["trajectory_data"] = val_data["trajectory_data"].cuda().float()

            max_pushes = push_data.shape[1]
            n_pushes = torch.randint(
                low=1, high=max_pushes, size=(1,)
            )  # Should ensure that both sides are at least 1

            if args.n_context != -1:
                n_pushes = args.n_context 
            perm = torch.randperm(max_pushes)
            context_pushes = perm[:n_pushes]
            target_pushes = perm[n_pushes:]
            context_push_data = push_data[:, context_pushes, :]
            context_push_data = torch.cat(
                [
                    context_push_data,
                    val_data["final_position"][:, context_pushes, :],
                ],
                dim=2,
            )
            target_push_data = push_data[:, target_pushes, :]

            model_data = (mesh_data, obj_data, target_push_data, context_push_data)

            predictions, mu, lower_tril, q_z_partial = model.forward(model_data)

            ground_truths = val_data["final_position"][:, target_pushes, :]
            ground_truths = ground_truths[:, :, :3]
            ground_truths = ground_truths.cpu().numpy()
            ground_truths = np.reshape(ground_truths, (-1, ground_truths.shape[-1]))
            all_final_positions.append(ground_truths)

            lower_tril_ = lower_tril.cpu().numpy()
            variance = lower_tril_ @ lower_tril_.transpose((0, 1, 3, 2))
            variance = np.reshape(variance, (-1, mu.shape[-1], mu.shape[-1]))
            all_sigma.append(variance)

            mu_ = mu.cpu().numpy()
            mu_ = np.reshape(mu_, (-1, mu_.shape[-1]))

            all_mu.append(mu_)

    all_mu = np.concatenate(all_mu, axis=0)
    all_sigma = np.concatenate(all_sigma, axis=0)
    all_final_positions = np.concatenate(all_final_positions, axis=0)
    num_points = args.num_points if args.num_points != -1 else all_mu.shape[0] 

    print(all_final_positions.shape, all_mu.shape, all_sigma.shape)

    # Rescale back the data to original scale
    pos_max = validation_dataset.data["final_position_max"][: all_mu.shape[-1]]
    pos_min = validation_dataset.data["final_position_min"][: all_mu.shape[-1]]
    all_mu = (pos_max - pos_min) * all_mu + pos_min  
    all_sigma = all_sigma * (pos_max - pos_min) ** 2
    all_final_positions = all_final_positions * (pos_max - pos_min) + pos_min

    perm = np.random.permutation(len(all_mu))[:num_points]
    all_mu = all_mu[perm]
    all_sigma = all_sigma[perm]
    all_final_positions = all_final_positions[perm] 


    norms = np.zeros(all_sigma.shape[0])
    eigen_vectors = np.zeros(all_sigma.shape)
    eigen_values = np.zeros(all_sigma.shape[:-1])
    projected_variances = np.zeros(all_sigma.shape[:-1])
    mean_distance = np.zeros(all_sigma.shape[0])

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
        }
    )



    if num_points > 100: 
        sns.set_theme()
        sns.scatterplot(data=df, x="var_x", y="var_y", hue="norm")
        plt.savefig(os.path.join(instance_path, "xy_norm.png"))
        plt.close()
        sns.scatterplot(data=df, x="var_x", y="var_y", hue="var_z")
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
            cov = np.array([[all_sigma[i, 0, 0], all_sigma[i, 0, 1]],
                            [all_sigma[i, 1, 0], all_sigma[i, 1, 1]]])
            confidence_ellipse(df['pred_x'].iloc[i], df['pred_y'].iloc[i], cov, plt.gca(), n_std=2.0, edgecolor=colors[i], alpha=0.3)

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f"Predicted vs True Positions with Covariance Ellipses (n={num_points})")
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

        # variances plot with individual covariance ellipses
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x="var_x", y="var_y", hue="var_z", ax=ax)
        for i in range(len(df)):
            cov = np.array([[projected_variances[i, 0], 0],
                            [0, projected_variances[i, 1]]])
            confidence_ellipse(df['var_x'].iloc[i], df['var_y'].iloc[i], cov, ax, edgecolor='red', alpha=0.1)
        plt.savefig(os.path.join(instance_path, "variances_with_individual_ellipses.png"))
        plt.close()

    print("Model loaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--num-points", type=int, default=-1)
    parser.add_argument("--n-context", type=int, default=-1)

    args = parser.parse_args()

    main(args)
