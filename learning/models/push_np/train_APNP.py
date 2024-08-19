from tqdm import tqdm
import argparse
import os
import pickle
from torch.utils.data import DataLoader
import torch

from learning.models.push_np.attention_push_np import AttentionPushNP
from learning.models.push_np.dataset import PushNPDataset, collate_fn 
from torchinfo import summary

import wandb

def train(model, args, train_dataloader, val_dataloader): 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 
    summary(model) 
    wandb.init(
        project="pushing-neural-process",
        config=args,
        name=args.instance,
    )  # This is to visualize the training process in an easy way.
    if torch.cuda.is_available():
        model = model.cuda() 


    best_val_loss = float('inf') 
    for epoch in range(args.num_epochs): 
        total_total_loss = 0
        total_kld_loss = 0
        total_bce_loss = 0
        val_loss = 0
        val_avg_dist = 0
        total_train_dist = 0
        print(f"Epoch {epoch}:")

        
        model.train() 
        for data in tqdm(train_dataloader): 

            # break
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            if args.use_obj_prop: 
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)
            else:
                obj_data = None

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

            context_xs = target_xs[:, perm] 
            context_ys = target_ys[:, perm] 

            if torch.cuda.is_available():
                context_xs = context_xs.cuda().float()
                context_ys = context_ys.cuda().float()
                target_xs = target_xs.cuda().float()
                target_ys = target_ys.cuda().float()
                mesh_data = mesh_data.cuda().float()
                if args.use_obj_prop:
                    obj_data = obj_data.cuda().float()

            optimizer.zero_grad()
            total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy = model(context_xs, context_ys, target_xs, target_ys, mesh_data, obj_data, "train") 
            total_loss.backward() 
            optimizer.step() 
            # total_total_loss += total_loss.item()
            total_kld_loss += kl_loss.item()
            total_bce_loss += bce_loss.item()   
            total_train_dist += distance.item()
        



        with torch.no_grad(): 
            for data in tqdm(val_dataloader): 
                mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
                if args.use_obj_prop: 
                    obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                    obj_data = torch.cat((obj_data, data["com"]), dim=1)
                else:
                    obj_data = None

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

                context_xs = target_xs[:, perm] 
                context_ys = target_ys[:, perm] 

                if torch.cuda.is_available():
                    context_xs = context_xs.cuda().float()
                    context_ys = context_ys.cuda().float()
                    target_xs = target_xs.cuda().float()
                    target_ys = target_ys.cuda().float()
                    mesh_data = mesh_data.cuda().float()
                    if args.use_obj_prop:
                        obj_data = obj_data.cuda().float()

                total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy = model(context_xs, context_ys, target_xs, target_ys, mesh_data, obj_data, "validate") 
                val_loss += bce_loss.item() 
                val_avg_dist += distance.item()
        total_total_loss = total_kld_loss + total_bce_loss 

        wandb.log({
            "train/total_loss": total_total_loss / len(train_dataloader),
            "train/kl_loss": total_kld_loss / len(train_dataloader),
            "train/bce_loss": total_bce_loss / len(train_dataloader),
            "val/loss": val_loss / len(val_dataloader),
            "val/dist": val_avg_dist / len(val_dataloader), 
            "train/dist": total_train_dist / len(train_dataloader) 
        })
        print(f"Total Loss: {total_total_loss / len(train_dataloader)}")
        print(f"KL Loss: {total_kld_loss / len(train_dataloader)}")
        print(f"BCE Loss: {total_bce_loss / len(train_dataloader)}")
        print(f"Train Distance: {total_train_dist / len(train_dataloader)}") 
        print(f"Validation Loss: {val_loss / len(val_dataloader)}")
        print(f"Validation Distance: {val_avg_dist / len(val_dataloader)}") 

        if val_loss < best_val_loss: 
            best_val_loss = val_loss 
            torch.save(model.state_dict(), os.path.join('learning', 'data', 'pushing', args.dataset, args.instance, 'best_model.pth')) 











def main(args): 

    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset) 
    instance_path = os.path.join(dataset_path, args.instance) 
    train_data = os.path.join(dataset_path, "train_dataset.pkl")
    validation_data = os.path.join(dataset_path, "samegeo_test_dataset.pkl")
    
    if os.path.exists(instance_path): 
        print(f'Instance {args.instance} found in dataset {args.dataset}.') 
        with open(os.path.join(instance_path, 'train_dataset.pkl'), 'rb') as f: 
            train_dataset = pickle.load(f) 
        with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'rb') as f: 
            validation_dataset = pickle.load(f)   
    else: 
        os.makedirs(instance_path)
        train_dataset = PushNPDataset(train_data, args.num_points)
        validation_dataset = PushNPDataset(validation_data, args.num_points) 

        with open(os.path.join(instance_path, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f) 
        with open(os.path.join(instance_path, 'validation_dataset.pkl'), 'wb') as f:
            pickle.dump(validation_dataset, f)

    with open(os.path.join(instance_path, 'args.pkl'), 'wb') as f: 
        pickle.dump(args, f) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)   
    test_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)   

    model = AttentionPushNP(args) 
    train(model, args, train_loader, test_loader) 




if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--instance', type=str, required=True)
    parser.add_argument('--use-obj-prop', action='store_true') 
    parser.add_argument('--use-full-trajectory', action='store_true') 
    parser.add_argument('--num-points', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=32) 
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--d_latents', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--attention-encoding', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-3)

    args = parser.parse_args() 
    main(args) 