from tqdm import tqdm
import argparse
import pickle
import torch
import os
from torch.utils.data import DataLoader
import wandb
import torch.nn.functional as F
import numpy as np
import tracemalloc
import gc

from learning.models.push_np.create_bar_plot import plot_probabilities
from learning.models.push_np.dataset import PushNPDataset, collate_fn
from learning.models.push_np.push_neural_process import PushNP


def loss_fn(predictions, targets, q_z, q_z_partial):
    batch_loss = 0
    kld_loss = 0
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            # batch_loss += F.mse_loss(predictions[i][j].mean, targets[i][j][:3])

            batch_loss += predictions[i][j].log_prob(targets[i][j])

            # if np.random.rand() < 0.01:
            #     print(f"Prediction: {predictions[i][j].mean} Target: {targets[i][j][:3]}")
        kld_loss += torch.distributions.kl_divergence(q_z[i], q_z_partial[i])

    # print(f"Loss2: {-loss} Loss: {-loss / (targets.shape[0] * targets.shape[1])}, Shape: {targets.shape[0] * targets.shape[1]}")
    # print(q_z, q_z_partial)
    # kld_loss = torch.distributions.kl_divergence(q_z, q_z_partial).sum()
    batch_loss /= targets.shape[0] * targets.shape[1] 
    kld_loss /= targets.shape[0] 
    return -batch_loss + kld_loss, -batch_loss, kld_loss


def accuracy_fn(predictions, mu, targets):
    total_distance = 0
    max_distance = 0

    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            distance = torch.sqrt(
                F.mse_loss(mu[i][j][:3], targets[i][j][:3], reduction="sum")
            )
            total_distance += distance
            max_distance += torch.sqrt(
                torch.tensor(3.0)
            )  # Maximum possible distance in 3D space

    # Calculate accuracy as a percentage
    total_distance /= targets.shape[0] * targets.shape[1]
    return total_distance 


def train(model, train_dataloader, test_dataloader, args, n_epochs=10):
    if "INSTANCE_NUMBER" not in os.environ:
        os.environ["INSTANCE_NUMBER"] = "0"
    os.environ["INSTANCE_NUMBER"] = str(int(os.environ["INSTANCE_NUMBER"]) + 1)
    wandb.init(
        project="pushing-neural-process",
        config=args,
        name=f"{args.file_name}_{args.instance_name}{os.environ['INSTANCE_NUMBER']}",
    )  # This is to visualize the training process in an easy way.
    model.train()
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    best_loss = float("inf")

    for epoch in range(n_epochs):
        print(f"----Epoch {epoch}----")

        model.train()

        train_epoch_kld_loss = 0
        train_epoch_batch_loss = 0
        train_epoch_total_loss = 0

        # print("LEN: ", len(train_dataloader))
        for data in tqdm(train_dataloader):
            # break
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            if args.use_obj_properties:
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)
            else:
                obj_data = (None, None, None)

            # print(data["angle"].shape, data["push_velocities"].shape)
            push_data = torch.stack((data["angle"], data["push_velocities"], data["initials"]), dim=2)
            # print(data["contact_points"].shape, data["normal_vector"].shape, data["orientation"].shape)
            push_data = torch.cat(
                (
                    push_data,
                    data["contact_points"],
                    data["normal_vector"],
                    # data["orientation"],
                ),
                dim=2,
            )
            # print(push_data[0, 0, :])

            if torch.cuda.is_available():
                mesh_data = mesh_data.cuda().float()
                push_data = push_data.cuda().float()
                if args.use_obj_properties:
                    obj_data = obj_data.cuda().float()
                data["final_position"] = data["final_position"].cuda().float()
                data["final_z_rotation"] = data["final_z_rotation"].cuda().float()
                # data["trajectory_data"] = data["trajectory_data"].cuda().float()

            max_pushes = push_data.shape[1]
            n_pushes = torch.randint(low=1, high=max_pushes + 1, size=(1,))
            n_indices = torch.randperm(max_pushes)[:n_pushes]
            n_push_data = push_data[:, n_indices, :]
            # if args.use_full_trajectory_feat:
            #     n_push_data = torch.cat(
            #         [
            #             n_push_data,
            #             torch.flatten(
            #                 data["trajectory_data"][:, n_indices, :], start_dim=2
            #             ),
            #         ],
            #         dim=2,
            #     )
            #     total_model_data = (
            #         mesh_data,
            #         obj_data,
            #         None,
            #         torch.cat(
            #             [
            #                 push_data,
            #                 torch.flatten(data["trajectory_data"], start_dim=2),
            #             ],
            #             dim=2,
            #         ),
            #     )
            # else:

            ground_truth = torch.cat([data["final_position"], data["final_z_rotation"][:, :, None]], dim=2)
            n_push_data = torch.cat(
                [n_push_data, ground_truth[:, n_indices, :]], dim=2
            )
            total_model_data = (
                mesh_data,
                obj_data,
                None,
                torch.cat([push_data, ground_truth], dim=2),
            )

            # print(n_push_data.shape)

            model_data = (mesh_data, obj_data, push_data, n_push_data)

            optimizer.zero_grad()
            predictions, _, _, q_z_partial = model.forward(model_data)
            q_z, _ = model.get_latent_space(total_model_data)

            loss, batch_loss, kld_loss = loss_fn(
                predictions, ground_truth, q_z, q_z_partial
            )
            loss.backward()

            train_epoch_total_loss += loss.item()
            train_epoch_batch_loss += batch_loss.item()
            train_epoch_kld_loss += kld_loss.item()

            optimizer.step()

            # print(loss.shape)
            # print(f"Loss: {loss}")
            # wandb.log({"train/loss": loss, "train/batch_loss": batch_loss, "train/kld_loss": kld_loss})
            # del loss

        print("Total Training Loss: ", train_epoch_total_loss)

        model.eval()
        val_epoch_total_loss = 0
        val_epoch_average_accuracy = 0


        with torch.no_grad(): 
            model.eval()
            # print("LEN: ", len(train_dataloader))
            for data in tqdm(test_dataloader):
                # break
                mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
                if args.use_obj_properties:
                    obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                    obj_data = torch.cat((obj_data, data["com"]), dim=1)
                else:
                    obj_data = (None, None, None)

                # print(data["angle"].shape, data["push_velocities"].shape)
                push_data = torch.stack((data["angle"], data["push_velocities"], data["initials"]), dim=2)
                # print(data["contact_points"].shape, data["normal_vector"].shape, data["orientation"].shape)
                push_data = torch.cat(
                    (
                        push_data,
                        data["contact_points"],
                        data["normal_vector"],
                        # data["orientation"],
                    ),
                    dim=2,
                )

                if torch.cuda.is_available():
                    mesh_data = mesh_data.cuda().float()
                    push_data = push_data.cuda().float()
                    if args.use_obj_properties:
                        obj_data = obj_data.cuda().float()
                    data["final_position"] = data["final_position"].cuda().float()
                    data["final_z_rotation"] = data["final_z_rotation"].cuda().float()
                    # data["trajectory_data"] = data["trajectory_data"].cuda().float()

                max_pushes = push_data.shape[1]
                n_pushes = torch.randint(low=1, high=max_pushes + 1, size=(1,))
                n_indices = torch.randperm(max_pushes)[:n_pushes]
                n_push_data = push_data[:, n_indices, :]
                # if args.use_full_trajectory_feat:
                #     n_push_data = torch.cat(
                #         [
                #             n_push_data,
                #             torch.flatten(
                #                 data["trajectory_data"][:, n_indices, :], start_dim=2
                #             ),
                #         ],
                #         dim=2,
                #     )
                #     total_model_data = (
                #         mesh_data,
                #         obj_data,
                #         None,
                #         torch.cat(
                #             [
                #                 push_data,
                #                 torch.flatten(data["trajectory_data"], start_dim=2),
                #             ],
                #             dim=2,
                #         ),
                #     )
                # else:

                ground_truth = torch.cat([data["final_position"], data["final_z_rotation"][:, :, None]], dim=2)
                n_push_data = torch.cat(
                    [n_push_data, ground_truth[:, n_indices, :]], dim=2
                )
                total_model_data = (
                    mesh_data,
                    obj_data,
                    None,
                    torch.cat([push_data, ground_truth], dim=2),
                )

                # print(n_push_data.shape)

                model_data = (mesh_data, obj_data, push_data, n_push_data)

                predictions, mu, _, q_z_partial = model.forward(model_data)
                q_z, _ = model.get_latent_space(total_model_data)

                loss, batch_loss, kld_loss = loss_fn(
                    predictions, ground_truth, q_z, q_z_partial
                )

                val_epoch_average_accuracy += accuracy_fn(predictions, mu, ground_truth)
                val_epoch_total_loss += loss.item()



        # with torch.no_grad():
        #     for val_data in tqdm(test_dataloader):
        #         # break
        #         mesh_data = torch.cat((val_data["mesh"], val_data["normals"]), dim=2)

        #         if args.use_obj_properties:
        #             obj_data = torch.stack(
        #                 (val_data["mass"], val_data["friction"]), dim=1
        #             )
        #             obj_data = torch.cat((obj_data, val_data["com"]), dim=1)
        #         else:
        #             obj_data = (None, None, None)

        #         push_data = torch.stack(
        #             (val_data["angle"], val_data["push_velocities"]), dim=2
        #         )
        #         # dimensions = val_data["angle"].shape
        #         push_data = torch.cat(
        #             (
        #                 push_data,
        #                 val_data["contact_points"],
        #                 val_data["normal_vector"],
        #                 val_data["orientation"],
        #             ),
        #             dim=2,
        #         )

        #         if torch.cuda.is_available():
        #             mesh_data = mesh_data.cuda().float()
        #             push_data = push_data.cuda().float()
        #             if args.use_obj_properties:
        #                 obj_data = obj_data.cuda().float()
        #             val_data["final_position"] = (
        #                 val_data["final_position"].cuda().float()
        #             )
        #             val_data["trajectory_data"] = (
        #                 val_data["trajectory_data"].cuda().float()
        #             )

        #         max_pushes = push_data.shape[1]
        #         n_pushes = torch.randint(
        #             low=1, high=max_pushes, size=(1,)
        #         )  # Should ensure that both sides are at least 1
        #         perm = torch.randperm(max_pushes)
        #         context_pushes = perm[:n_pushes]
        #         target_pushes = perm[n_pushes:]
        #         context_push_data = push_data[:, context_pushes, :]
        #         if args.use_full_trajectory_feat:
        #             context_push_data = torch.cat(
        #                 [
        #                     context_push_data,
        #                     torch.flatten(
        #                         val_data["trajectory_data"][:, context_pushes, :],
        #                         start_dim=2,
        #                     ),
        #                 ],
        #                 dim=2,
        #             )
        #             total_model_data = (
        #                 mesh_data,
        #                 obj_data,
        #                 None,
        #                 torch.cat(
        #                     [
        #                         push_data,
        #                         torch.flatten(val_data["trajectory_data"], start_dim=2),
        #                     ],
        #                     dim=2,
        #                 ),
        #             )
        #         else:
        #             context_push_data = torch.cat(
        #                 [
        #                     context_push_data,
        #                     val_data["final_position"][:, context_pushes, :],
        #                 ],
        #                 dim=2,
        #             )
        #             total_model_data = (
        #                 mesh_data,
        #                 obj_data,
        #                 None,
        #                 torch.cat(push_data, val_data["final_position"], dim=2),
        #             )
        #         # print(context_push_data.shape)
        #         target_push_data = push_data[:, target_pushes, :]

        #         model_data = (mesh_data, obj_data, target_push_data, context_push_data)
        #         predictions, mu, _, q_z_partial = model.forward(model_data)

        #         q_z, _ = model.get_latent_space(total_model_data)
        #         loss, _, _ = loss_fn(
        #             predictions,
        #             val_data["final_position"][:, target_pushes, :],
        #             q_z,
        #             q_z_partial,
        #         )

        #         val_epoch_total_loss += loss.item()

        #         acc = accuracy_fn(
        #             predictions,
        #             mu,
        #             val_data["final_position"][:, target_pushes, :],
        #             probabilities,
        #         )

        #         val_epoch_average_accuracy += acc

        val_epoch_average_accuracy /= len(test_dataloader)

        wandb.log(
            {
                "val/total_loss": val_epoch_total_loss,
                "train/kld_loss": train_epoch_kld_loss,
                "train/batch_loss": train_epoch_batch_loss,
                "train/total_loss": train_epoch_total_loss,
                "val/average_accuracy": val_epoch_average_accuracy,
            }
        )

        print("Total Validation Loss: ", val_epoch_total_loss)
        print("Average Validation Accuracy: ", val_epoch_average_accuracy)
        if val_epoch_total_loss < best_loss:
            best_loss = val_epoch_total_loss
            torch.save(
                model.state_dict(),
                os.path.join(
                    "learning",
                    "data",
                    "pushing",
                    args.file_name,
                    args.instance_name,
                    "best_model.pth",
                ),
            )

            # plot_probabilities(probabilities, args)

            # histogram = np.histogram(probabilities)
            # wandb.log({"Probabilities": wandb.Histogram(np_histogram=histogram)})

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        gc.collect()

        # for stat in top_stats[:10]:
        #     print(stat)
        #     print("---------------------------------")

        # for i in range(3):
        #     stat = top_stats[i]
        #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        #     for line in stat.traceback.format():
        #         print(line)


def main(args):
    # tracemalloc.start()
    data_path = os.path.join("learning", "data", "pushing", args.file_name)
    train_data = os.path.join(data_path, "train_dataset.pkl")
    validation_data = os.path.join(data_path, "samegeo_test_dataset.pkl")
    instance_path = os.path.join(data_path, args.instance_name)

    # This is so that we don't have to create a dataset everytime.
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
        args_path = os.path.join(data_path, args.instance_name, "args.pkl")
        with open(args_path, "wb") as handle:
            pickle.dump(args, handle)

        train_dataset = PushNPDataset(train_data, args.n_samples, args.balance_dataset)
        validation_dataset = PushNPDataset(
            validation_data, args.n_samples, args.balance_dataset
        )
        with open(os.path.join(instance_path, "train_dataset.pkl"), "wb") as handle:
            pickle.dump(train_dataset, handle)
        with open(
            os.path.join(instance_path, "validation_dataset.pkl"), "wb"
        ) as handle:
            pickle.dump(validation_dataset, handle)
    else:
        print(f"Instance {args.instance_name} already exists")
        with open(os.path.join(instance_path, "args.pkl"), "rb") as handle:
            # args = pickle.load(handle)
            with open(os.path.join(instance_path, "train_dataset.pkl"), "rb") as handle:
                train_dataset = pickle.load(handle)
            with open(
                os.path.join(instance_path, "validation_dataset.pkl"), "rb"
            ) as handle:
                validation_dataset = pickle.load(handle)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=validation_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
    )
    # For now we will keep all of the features!!
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
        "orientation",
    }
    if not args.use_obj_properties:
        features.remove("mass")
        features.remove("com")
        features.remove("friction")
    if not args.use_full_trajectory_feat:
        features.remove("trajectory_data")
    model = PushNP(features, 4, d_latents=args.d_latents)
    train(model, train_dataloader, test_dataloader, args, args.n_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize final positions and orientations of sliders from pushing data."
    )
    parser.add_argument(
        "--file-name",
        type=str,
        help="Name of the file that you want to test PushNP Dataset on.",
        required=True,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="Number of samples generated for each of the point nets",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for dataloader", required=True
    )
    parser.add_argument(
        "--use-obj-properties",
        action="store_true",
        help="Use only if you want the actual mass,com,friction to be used in model.",
    )
    parser.add_argument(
        "--n-epochs", type=int, help="Number of epochs for training", required=True
    )

    parser.add_argument(
        "--instance-name", type=str, help="Name of the instance", required=True
    )
    parser.add_argument(
        "--balance-dataset", action="store_true", help="Balance the dataset"
    )

    parser.add_argument(
        "--d-latents", type=int, help="Dimensionality of the latent space", default=5
    )

    parser.add_argument(
        "--use-full-trajectory-feat",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
