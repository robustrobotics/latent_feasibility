from tqdm import tqdm
import argparse
import pickle
import torch
import os
from torch.utils.data import DataLoader
import wandb
import torch.nn.functional as F
import numpy as np

from learning.models.push_np.create_bar_plot import plot_probabilities
from learning.models.push_np.dataset import PushNPDataset, collate_fn
from learning.models.push_np.push_neural_process import PushNP


def loss_fn(predictions, targets):
    loss = 0
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            loss += predictions[i][j].log_prob(targets[i][j][:3])

            if np.random.rand() < 0.01:
                print(f"Prediction: {predictions[i][j].mean} Target: {targets[i][j][:3]}")


    # print(f"Loss2: {-loss} Loss: {-loss / (targets.shape[0] * targets.shape[1])}, Shape: {targets.shape[0] * targets.shape[1]}")
    return -loss 


def accuracy_fn(predictions, mu, targets, probabilities):
    total_distance = 0
    max_distance = 0

    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            distance = torch.sqrt(
                F.mse_loss(mu[i][j], targets[i][j][:3], reduction="sum")
            )
            probabilities.append(torch.exp(predictions[i][j].log_prob(targets[i][j][:3])).item())  
            total_distance += distance
            max_distance += torch.sqrt(
                torch.tensor(3.0)
            )  # Maximum possible distance in 3D space

    # Calculate accuracy as a percentage
    accuracy = 100 * (1 - total_distance / max_distance)
    return accuracy.item()


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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    train_epoch_total_loss = 0

    for epoch in range(n_epochs):
        print(f"----Epoch {epoch}----")

        model.train()
        for data in tqdm(train_dataloader):
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            if args.use_obj_properties:
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)
            else:
                obj_data = (None, None, None)

            push_data = torch.stack((data["angle"], data["push_velocities"]), dim=2)
            push_data = torch.cat((push_data, data["contact_points"]), dim=2)

            if torch.cuda.is_available():
                mesh_data = mesh_data.cuda().float()
                push_data = push_data.cuda().float()
                if args.use_obj_properties:
                    obj_data = obj_data.cuda().float()
                data["final_position"] = data["final_position"].cuda().float()
                data["trajectory_data"] = data["trajectory_data"].cuda().float()

            max_pushes = push_data.shape[1]
            n_pushes = torch.randint(low=1, high=max_pushes + 1, size=(1,))
            n_indices = torch.randperm(max_pushes)[:n_pushes]
            n_push_data = push_data[:, n_indices, :]

            model_data = (mesh_data, obj_data, push_data, n_push_data)

            optimizer.zero_grad()
            predictions, _ = model.forward(model_data)
            loss = loss_fn(predictions, data["final_position"])
            train_epoch_total_loss += loss
            # print(f"Loss: {loss}")
            wandb.log({"train/loss": loss})

            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        total_acc = []
        probabilities = []
        with torch.no_grad():
            for val_data in tqdm(test_dataloader):
                mesh_data = torch.cat((val_data["mesh"], val_data["normals"]), dim=2)

                if args.use_obj_properties:
                    obj_data = torch.stack(
                        (val_data["mass"], val_data["friction"]), dim=1
                    )
                    obj_data = torch.cat((obj_data, val_data["com"]), dim=1)
                else:
                    obj_data = (None, None, None)

                push_data = torch.stack(
                    (val_data["angle"], val_data["push_velocities"]), dim=2
                )
                # dimensions = val_data["angle"].shape
                push_data = torch.cat((push_data, val_data["contact_points"]), dim=2)

                if torch.cuda.is_available():
                    mesh_data = mesh_data.cuda().float()
                    push_data = push_data.cuda().float()
                    if args.use_obj_properties:
                        obj_data = obj_data.cuda().float()
                    val_data["final_position"] = (
                        val_data["final_position"].cuda().float()
                    )
                    val_data["trajectory_data"] = (
                        val_data["trajectory_data"].cuda().float()
                    )

                max_pushes = push_data.shape[1]
                n_pushes = torch.randint(
                    low=1, high=max_pushes, size=(1,)
                )  # Should ensure that both sides are at least 1
                perm = torch.randperm(max_pushes)
                context_pushes = perm[:n_pushes]
                target_pushes = perm[n_pushes:]
                context_push_data = push_data[:, context_pushes, :]
                target_push_data = push_data[:, target_pushes, :]

                model_data = (mesh_data, obj_data, target_push_data, context_push_data)
                predictions, mu = model.forward(model_data)
                loss = loss_fn(
                    predictions, val_data["final_position"][:, target_pushes, :]
                )
                acc = accuracy_fn(predictions, mu, val_data["final_position"][:, target_pushes, :], probabilities)

                # wandb.log({"val/loss": loss})
                total_val_loss += loss
                total_acc.append(acc)

        wandb.log(
            {
                "val/total_loss": total_val_loss,
                "val/total_acc": np.array(total_acc).mean(),
                "train/total_loss": train_epoch_total_loss,
            }
        )
        # wandb.log({"val/total_acc": np.array(total_acc).mean()})
        if total_val_loss < best_loss:
            best_loss = total_val_loss
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

            plot_probabilities(probabilities, args)


def main(args):
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

        train_dataset = PushNPDataset(train_data, args.n_samples)
        validation_dataset = PushNPDataset(validation_data, args.n_samples)
        with open(os.path.join(instance_path, "train_dataset.pkl"), "wb") as handle:
            pickle.dump(train_dataset, handle)
        with open(
            os.path.join(instance_path, "validation_dataset.pkl"), "wb"
        ) as handle:
            pickle.dump(validation_dataset, handle)
    else:
        print(f"Instance {args.instance_name} already exists")
        with open(os.path.join(instance_path, "args.pkl"), "rb") as handle:
            args = pickle.load(handle)
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
    }
    if not args.use_obj_properties:
        features.remove("mass")
        features.remove("com")
        features.remove("friction")
    model = PushNP(features, 3, d_latents=5)
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

    args = parser.parse_args()

    main(args)
