import argparse
import copy

import numpy as np
import pickle
import torch

from torch.nn import functional as F
from torch.utils.data import DataLoader

from learning.active.utils import ActiveExperimentLogger
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from learning.models.grasp_np.grasp_neural_process import CustomGraspNeuralProcess


def check_to_cuda(tensor_list):
    if torch.cuda.is_available():
        return [tensor.cuda() for tensor in tensor_list]
    else:
        return tensor_list


def get_accuracy(y_probs, target_ys, test=False, save=False, object_properties=None):
    assert (y_probs.shape == target_ys.shape)
    if test == True:
        if y_probs.shape[0] > 100000:
            n_grasps = 50
        else:
            n_grasps = 10

        per_obj_probs = y_probs.view(-1, n_grasps)
        per_obj_target = target_ys.view(-1, n_grasps)
        per_obj_acc = ((per_obj_probs > 0.5) == per_obj_target).float().mean(dim=1)
        print(per_obj_probs.shape, per_obj_acc.shape)
        if save:
            with open('learning/experiments/metadata/grasp_np/accs.pkl', 'wb') as handle:
                pickle.dump((per_obj_acc, per_obj_target), handle)
                print(per_obj_probs.shape)
        print('HIST:', np.histogram(per_obj_acc.cpu(), bins=10))
        if save:
            with open('learning/experiments/metadata/grasp_np/results_val.pkl', 'wb') as handle:
                pickle.dump((y_probs.cpu().numpy(), target_ys.cpu().numpy()), handle)

    acc = ((y_probs > 0.5) == target_ys).float().mean()
    # if not object_properties is None and acc > 0.9 and test:
    #     bad_indices = np.arange(500)[per_obj_acc.cpu().numpy() < 0.6]
    #     for ix in bad_indices:
    #         print(object_properties[ix])
    #     import IPython; IPython.embed()
    return acc


def get_loss(y_probs, target_ys):
    bce_loss = F.binary_cross_entropy(y_probs.squeeze(), target_ys.squeeze(), reduction='sum')
    return bce_loss

def train(train_dataloader, val_dataloader, model, n_epochs=10):
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 10000
    best_weights = None

    alpha = 0.
    for ep in range(n_epochs):
        print(f'----- Epoch {ep} -----')
        alpha *= 0.75
        epoch_loss, train_probs, train_targets = 0, [], []
        model.train()
        for bx, (context_data, target_data, meshes) in enumerate(train_dataloader):

            # sample a t \in \{0, ..., max_grasps_per_object}
            # sampling t data points (grasps + geoms + labels) from batched objects (make sure the
            # sampling we choose per object have the same indices
            # REMEMBER THEM!!!

            # TODO: simplify-nn -- add in the curvature and the grasp position data
            c_grasp_geoms, c_grasp_points, c_curvatures, c_midpoints, c_forces, c_labels, object_props = check_to_cuda(context_data)
            # t_grasp_geoms, t_grasp_points, t_curvatures, t_midpoints, t_forces, t_labels, object_props = check_to_cuda(target_data)
            if torch.cuda.is_available():
                meshes = meshes.cuda()

            optimizer.zero_grad()

            # TODO: simplify-nn -- add in the curvature and the grasp position data
            # pass forward for max_n_grasps
            y_probs = model.conditional_forward(
                (c_grasp_geoms, c_grasp_points, c_curvatures, c_midpoints, c_forces),
                meshes,
                object_props
            )
            y_probs = y_probs.squeeze()

            loss = get_loss(y_probs, c_labels)
            if bx == 0:
                print(f'Loss: {loss.item()}')

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_probs.append(y_probs.flatten())
            train_targets.append(c_labels.flatten())

        epoch_loss /= len(train_dataloader.dataset)
        train_acc = get_accuracy(
            torch.cat(train_probs).flatten(),
            torch.cat(train_targets).flatten()
        )
        print(f'Train Loss: {epoch_loss}\tTrain Acc: {train_acc}')

        model.eval()
        means = []
        val_loss, val_probs, val_targets = 0, [], []
        with torch.no_grad():
            for bx, (context_data, target_data, meshes) in enumerate(val_dataloader):
                t_grasp_geoms, t_grasp_points, t_curvatures, t_midpoints, t_forces, t_labels, object_props = check_to_cuda(target_data)
                if torch.cuda.is_available():
                    meshes = meshes.cuda()

                # TODO: simplify-nn -- add in the curvature and the grasp position data
                y_probs = model.conditional_forward(
                    (t_grasp_geoms, t_grasp_points, t_curvatures, t_midpoints, t_forces),
                    meshes,
                    object_props
                )
                y_probs = y_probs.squeeze()


                val_loss += get_loss(y_probs, t_labels).item()

                val_probs.append(y_probs.flatten())
                val_targets.append(t_labels.flatten())

            val_loss /= len(val_dataloader.dataset)
            val_acc = get_accuracy(
                torch.cat(val_probs),
                torch.cat(val_targets),
                test=True, save=True,
                object_properties=val_dataloader.dataset.object_properties
            )
            print(f'Val Loss: {val_loss}\tVal Acc: {val_acc}\t% Stable: {np.mean(c_labels.cpu().numpy())}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                print('New best loss: ', val_loss)

    model.load_state_dict(best_weights)
    return model


def print_dataset_stats(dataset, name):
    print(f'----- {name} Dataset Statistics -----')
    print(f'N: {len(dataset)}')
    print(f'Context Shape: {dataset.contexts[0].shape}')
    print(f'Target xs Shape: {dataset.target_xs[0].shape}')
    print(f'Target ys Shape: {dataset.target_xs[0].shape}')


def run(args):
    # set up logger  # args.exp_name
    logger = ActiveExperimentLogger.setup_experiment_directory(args)

    # build the model # args.5
    model = CustomGraspNeuralProcess(d_latents=5, use_geoms=True)

    # load datasets
    with open(args.train_dataset_fname, 'rb') as handle:
        train_data = pickle.load(handle)
    with open(args.val_dataset_fname, 'rb') as handle:
        val_data = pickle.load(handle)

    train_dataset = CustomGNPGraspDataset(data=train_data)
    val_dataset_eval = CustomGNPGraspDataset(data=val_data, context_data=train_data)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda items: custom_collate_fn(items, True),
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset_eval,
        collate_fn=lambda items: custom_collate_fn(items, True),
        batch_size=args.batch_size,
        shuffle=False
    )

    # train model
    model = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        n_epochs=args.n_epochs
    )

    # save model
    logger.save_dataset(dataset=train_dataset, tx=0)
    logger.save_val_dataset(val_dataset=val_dataset_eval, tx=0)
    logger.save_neural_process(gnp=model, tx=0, symlink_tx0=False)

    return logger.exp_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset-fname', type=str, required=True)
    parser.add_argument('--val-dataset-fname', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--n-epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    args = parser.parse_args()
    args.use_latents = False  # NOTE: this is for the specific workaround for block stacking that assumes
    # a different NN architecture
    run(args)
