import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm
import argparse

# Import necessary functions and classes from your existing code
from learning.models.push_np.dataset import collate_fn
from learning.models.push_np.attention_push_np import AttentionPushNP

def evaluate_model(model, data_loader, context_num, training_args):
    all_distances = []
    all_entropies = []
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc=f"Evaluating with {context_num} contexts"):
            mesh_data = torch.cat((data["mesh"], data["normals"]), dim=2)
            obj_data = None
            if training_args.use_obj_prop:
                obj_data = torch.stack((data["mass"], data["friction"]), dim=1)
                obj_data = torch.cat((obj_data, data["com"]), dim=1)

            max_context_pushes = data["angle"].shape[1]
            perm = torch.randperm(max_context_pushes)[:context_num]

            target_xs = torch.stack((data["angle"], data["push_velocities"], data["initials"]), dim=2)
            target_xs = torch.cat((
                target_xs,
                data["contact_points"],
                data["normal_vector"],
            ), dim=2)

            target_ys = torch.cat([data["final_position"], data["final_z_rotation"].unsqueeze(2)], dim=2)

            context_xs = target_xs[:, perm]
            context_ys = target_ys[:, perm]

            if torch.cuda.is_available():
                context_xs = context_xs.cuda().float()
                context_ys = context_ys.cuda().float()
                target_xs = target_xs.cuda().float()
                target_ys = target_ys.cuda().float()
                mesh_data = mesh_data.cuda().float()
                if training_args.use_obj_prop:
                    obj_data = obj_data.cuda().float()

            total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy = model(context_xs, context_ys, target_xs, target_ys, mesh_data, obj_data, "validate")
            
            # print(entropy)
            all_distances.append(distance.cpu().numpy())
            all_entropies.append(entropy.cpu().numpy())

    # print(np.mean(all_entropies))
    return np.array(all_distances), np.array(all_entropies)

def main(args):
    print(args)
    dataset_path = os.path.join('learning', 'data', 'pushing', args.dataset)
    instance_path = os.path.join(dataset_path, args.instance)
    training_args_path = os.path.join(instance_path, 'args.pkl')
    # print(training_args_path)

    with open(training_args_path, 'rb') as f:
        training_args = pickle.load(f)

    # training_args.no_deterministic = False
    print(training_args)
    model = AttentionPushNP(training_args)
    model.load_state_dict(torch.load(os.path.join(instance_path, 'best_model.pth')))
    model = model.cuda()

    with open(os.path.join(instance_path, "validation_dataset.pkl"), "rb") as handle:
        validation_dataset = pickle.load(handle)

    data_loader = DataLoader(
        validation_dataset,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    context_numbers = range(1, args.max_context + 1)
    avg_distances = []
    avg_entropies = []
    sem_distances = []
    sem_entropies = []

    for context_num in context_numbers:
        distances, entropies = evaluate_model(model, data_loader, context_num, training_args)
        
        avg_distances.append(np.mean(distances))
        avg_entropies.append(np.mean(entropies))
        
        # Calculate standard error of the mean (SEM)
        sem_distances.append(np.std(distances) / np.sqrt(len(distances)))
        sem_entropies.append(np.std(entropies) / np.sqrt(len(entropies)))

    print("Average Distances:", avg_distances)
    print("Average Entropies:", avg_entropies)

    # Plot results with error bars
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(context_numbers, avg_distances, yerr=sem_distances, fmt='o-', capsize=5)
    plt.title('Average Distance vs. Number of Contexts')
    plt.xlabel('Number of Contexts')
    plt.ylabel('Average Distance')

    plt.subplot(1, 2, 2)
    plt.errorbar(context_numbers, avg_entropies, yerr=sem_entropies, fmt='o-', capsize=5)
    plt.title('Average Entropy vs. Number of Contexts')
    plt.xlabel('Number of Contexts')
    plt.ylabel('Average Entropy')

    plt.tight_layout()
    plt.savefig(os.path.join(instance_path, "performance_vs_contexts.png"))
    plt.close()

    print("Evaluation complete. Results saved as 'performance_vs_contexts.png'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pushing')
    parser.add_argument('--instance', type=str, default='default')
    parser.add_argument('--max-context', type=int, default=10, help='Maximum number of contexts to evaluate')

    args = parser.parse_args()
    main(args)