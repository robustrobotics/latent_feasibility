import argparse
import pickle
import torch
import os
from torch.utils.data import DataLoader

from learning.models.push_np.dataset import PushNPDataset, collate_fn
from learning.models.push_np.push_neural_process import PushNP

def loss_fn(predictions, targets):
    loss = 0 
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            loss += predictions[i][j].log_prob(targets[i][j][:3]).sum()
    return -loss

def train(model, train_dataloader, args, n_epochs=10):
    model.train() 
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        print(f"----Epoch {epoch}----")

        model.train()
        for data in train_dataloader:
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            if args.use_obj_properties:
                # obj_data = torch.Tensor(data["com"], data["friction"], data["mass"])
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)
                # print(obj_data.shape)
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

            predictions = model.forward(model_data)
            loss = loss_fn(predictions, data["final_position"])
            # print(f"Loss: {loss}")
            optimizer.zero_grad()

            loss.backward()  
            optimizer.step()


def main(args):
    data_path = os.path.join("learning", "data", "pushing", args.file_name)
    train_data = os.path.join(data_path, "train_dataset.pkl")
    validation_data = os.path.join(data_path, "test_dataset.pkl")
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
        with open(os.path.join(instance_path, "validation_dataset.pkl"), "wb") as handle:
            pickle.dump(validation_dataset, handle)
    else:
        print(f"Instance {args.instance_name} already exists") 
        with open(os.path.join(instance_path, "args.pkl"), "rb") as handle:
            args = pickle.load(handle) 
            with open(os.path.join(instance_path, "train_dataset.pkl"), "rb") as handle: 
                train_dataset = pickle.load(handle)
            with open(os.path.join(instance_path, "validation_dataset.pkl"), "rb") as handle:
                validation_dataset = pickle.load(handle)
            
    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
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
    train(model, train_dataloader, args, args.n_epochs)


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
        "--n-epochs", type=int, help="Number of epochs for training", required=True)

    parser.add_argument("--instance-name", type=str, help="Name of the instance", required=True)

    args = parser.parse_args()


    main(args)
