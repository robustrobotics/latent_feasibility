import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import trimesh
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody
from scipy.spatial.transform import Rotation as R


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

def quat_to_euler(logs): 
    ret = []
    for log in logs: 
        q = log[3:] 
        ret.append(R.from_quat(q).as_euler('xyz', degrees=False)) 
    return ret 


class OldPushNPDataset(Dataset):
    """
    Specific dataset for the PushNP. You can choose to give trajectories as predictions or just the final point.
    """

    def __init__(self, data_path, n_samples, balance_dataset=False):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        angle_data = []
        mesh_data = []
        com_data = []
        mass_data = []
        friction_data = []
        name_data = []
        final_position_data = []
        trajectory_data = []
        push_velocity_data = []
        normal_data = []
        contact_point_data = []
        normal_vector_data = []
        quaternion_data = [] 
        euler_angle_data = []

        for object_data in tqdm(data):
            # print(object_data)
            body = object_data[0][0][3]  # Get the body
            # print(body) 
            name, com, mass, friction = body
            # print(com, mass, friction) 
            sim_client = GraspSimulationClient(body, False)
            # print(sim_client._get_object_urdf(body))
            mesh = sim_client.mesh
            points, indices = mesh.sample(n_samples, return_index=True)
            points = np.array(points).reshape(-1, 3)
            normals = np.array(sim_client.mesh.face_normals[indices, :]).reshape(-1, 3)

            sim_client.disconnect()
            bad = False
            angles = []
            final_positions = []
            trajectories = []
            push_velocities = []
            contact_points = []
            normal_vectors = []
            quaternions = []
            euler_angles = []
            for push_data in object_data:
                input_params, (success, logs) = push_data
                if len(input_params) == 6:
                    angle, contact_point, normal_vector, body, push_velocity, quaternion = input_params
                else: 
                    angle, contact_point, normal_vector, body, push_velocity, quaternion, _ = input_params

                if contact_point is None:
                    bad = True
                    break
                # print(push_data) 

                quaternion = np.array(quaternion) 
                norm = np.linalg.norm(quaternion)
                quaternion = quaternion / norm 
                if quaternion[0] < 0: 
                    quaternion = -quaternion 
                # print(push_data)

                quaternions.append(quaternion) 
                push_velocities.append(push_velocity)
                final_positions.append(logs[-1])
                trajectories.append(logs)
                normal_vectors.append(normal_vector)
                contact_points.append(contact_point)
                euler_angles.append(quat_to_euler(logs))
                angles.append(angle)
            if bad:
                continue
            if balance_dataset: 
                norms = np.linalg.norm(np.array(final_positions), axis=1)
                # print(norms.shape)
                tot = np.sum(norms)
                n = len(object_data) 
                probs = norms / tot   
        

                # print(n) 
                # print(probs.shape)
                
                kept_indices = np.random.choice(np.arange(n), n // 4 * 3, p=probs, replace=False) 

                angles = [angles[i] for i in kept_indices]
                final_positions = [final_positions[i] for i in kept_indices]
                trajectories = [trajectories[i] for i in kept_indices]
                push_velocities = [push_velocities[i] for i in kept_indices]
                contact_points = [contact_points[i] for i in kept_indices]
                normal_vectors = [normal_vectors[i] for i in kept_indices]
                quaternion_data = [quaternion_data[i] for i in kept_indices] 
    
                
                     
            euler_angle_data.append(euler_angles) 
            quaternion_data.append(quaternions) 
            normal_vector_data.append(normal_vectors)
            contact_point_data.append(contact_points)
            push_velocity_data.append(push_velocities)
            angle_data.append(angles)
            mesh_data.append(points)
            com_data.append(com)
            mass_data.append(mass)
            friction_data.append(friction)
            name_data.append(name)
            trajectory_data.append(trajectories)
            final_position_data.append(final_positions)
            normal_data.append(normals)

        new_data = {}
        new_data["angle"] = np.array(angle_data)
        new_data["mesh"] = np.array(mesh_data)
        new_data["friction"] = np.array(friction_data)
        new_data["com"] = np.array(com_data)
        new_data["mass"] = np.array(mass_data)
        new_data["final_position"] = np.array(final_position_data)
        new_data["trajectory_data"] = np.array(trajectory_data)
        new_data["push_velocities"] = np.array(push_velocity_data)
        new_data["normals"] = np.array(normal_data)
        new_data["contact_points"] = np.array(contact_point_data)
        new_data["normal_vector"] = np.array(normal_vector_data) 
        new_data["orientation"] = np.array(quaternion_data) 
        new_data["euler_angles"] = np.array(euler_angle_data) 
        print("BRUH: ", new_data["contact_points"].shape)


        # Standardizing the data
        self.data = {}
        scaler = StandardScaler()

        self.data["orientation"] = new_data["orientation"]

        contact_point_shape = new_data["contact_points"].shape
        contact_point_flat = new_data["contact_points"].reshape(-1, 3)
        self.data["contact_points_min"] = np.zeros(3)
        self.data["contact_points_max"] = np.zeros(3)
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            contact_point_flat[:, i] = min_max_scaler.fit_transform(
                contact_point_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["contact_points_min"][i] = min_max_scaler.data_min_ 
            self.data["contact_points_max"][i] = min_max_scaler.data_max_ 

        self.data["contact_points"] = contact_point_flat.reshape(contact_point_shape)
        print(f"shape of contact_points: {self.data['contact_points'].shape}")

        # Normalize angles, friction, mass, final positions, trajectory data, and push velocities
        for key in ["angle", "friction", "mass", "push_velocities"]:
            flat_value = new_data[key].reshape(-1, 1)
            standardized_value = scaler.fit_transform(flat_value).reshape(
                new_data[key].shape
            )
            self.data[key] = standardized_value
            print(f"shape of {key}: {self.data[key].shape}")

        # Normalize mesh data by scaling each dimension (x, y, z) separately
        mesh_shape = new_data["mesh"].shape
        mesh_flat = new_data["mesh"].reshape(-1, 3)
        self.data["mesh_min"] = np.zeros(3)
        self.data["mesh_max"] = np.zeros(3)
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            mesh_flat[:, i] = min_max_scaler.fit_transform(
                mesh_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["mesh_min"][i] = min_max_scaler.data_min_
            self.data["mesh_max"][i] = min_max_scaler.data_max_ 
        self.data["mesh"] = mesh_flat.reshape(mesh_shape)
        print(f"shape of mesh: {self.data['mesh'].shape}")

        # Normalize center of mass (com) by scaling each dimension (x, y, z) separately
        com_shape = new_data["com"].shape
        com_flat = new_data["com"].reshape(-1, 3)
        self.data["com_min"] = np.zeros(3)
        self.data["com_max"] = np.zeros(3)   
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            com_flat[:, i] = min_max_scaler.fit_transform(
                com_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["com_min"][i] = min_max_scaler.data_min_
            self.data["com_max"][i] = min_max_scaler.data_max_
        self.data["com"] = com_flat.reshape(com_shape)
        print(f"shape of com: {self.data['com'].shape}")

        # Normalize normals to unit vectors
        norms = np.linalg.norm(new_data["normals"], axis=2, keepdims=True)
        normalized_normals = new_data["normals"] / norms
        self.data["normals"] = normalized_normals
        print(f"shape of normals: {self.data['normals'].shape}")

        final_positions_shape = new_data["final_position"].shape
        final_positions_flat = new_data["final_position"].reshape(-1, 7)
        self.data["final_position_min"] = np.zeros(7)
        self.data["final_position_max"] = np.zeros(7)
        for i in range(7):
            min_max_scaler = MinMaxScaler()
            final_positions_flat[:, i] = min_max_scaler.fit_transform(
                final_positions_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["final_position_min"][i] = min_max_scaler.data_min_
            self.data["final_position_max"][i] = min_max_scaler.data_max_ 
        self.data["final_position"] = final_positions_flat.reshape(
            final_positions_shape
        )
        print(f"shape of final_positions: {self.data['final_position'].shape}")

        trajectories_shape = new_data["trajectory_data"].shape
        trajectories_flat = new_data["trajectory_data"].reshape(-1, 7)
        self.data["trajectory_data_min"] = np.zeros(7)
        self.data["trajectory_data_max"] = np.zeros(7) 
        for i in range(7):
            min_max_scaler = MinMaxScaler()
            trajectories_flat[:, i] = min_max_scaler.fit_transform(
                trajectories_flat[:, i].reshape(-1, 1)
            ).flatten()

            self.data["trajectory_min"] = min_max_scaler.data_min_
            self.data["trajectory_max"] = min_max_scaler.data_max_ 
        self.data["trajectory_data"] = trajectories_flat.reshape(trajectories_shape)
        print(f"shape of trajectories: {self.data['trajectory_data'].shape}")
        self.data["normal_vector"] = new_data["normal_vector"] 
        print(f"shape of normal_vector: {self.data['normal_vector'].shape}")

        self.data["euler_angles"] = new_data["euler_angles"]

    def __getitem__(self, idx):
        item = {}
        for key, value in self.data.items():
            if key[-3:] != "min" and key[-3:] != "max":
                item[key] = value[idx]
            else: 
                item[key] = value 
        return item

    def __len__(self):
        return self.data["angle"].shape[0]


def collate_fn(items):
    # print(len(np.array(items)))
    total_dict = {}
    for item in items:
        for key in item.keys():
            total_dict[key] = []
            if key[-3:] == "min" or key[-3:] == "max":
                total_dict[key] = item[key]
        break
    for item in items:
        for key, value in item.items():
            if key[-3:] != "min" and key[-3:] != "max":
                total_dict[key].append(value)
    for key, value in total_dict.items():
        total_dict[key] = torch.tensor(value)
    # print(total_dict)
    return total_dict





class PushNPDataset(Dataset):
    """
    Specific dataset for the PushNP. You can choose to give trajectories as predictions or just the final point.
    """

    def __init__(self, data_path, n_samples, balance_dataset=False):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        angle_data = []
        mesh_data = []
        com_data = []
        mass_data = []
        friction_data = []
        name_data = []
        final_position_data = []
        push_velocity_data = []
        normal_data = []
        contact_point_data = []
        normal_vector_data = []
        initial_data = []
        final_z_rotation_data = [] 


        for object_data in tqdm(data):
            # print(object_data)
            body = object_data[0][0][3]  # Get the body
            if body is None: 
                continue
            # print(body) 
            name, com, mass, friction = body 
            # print(com, mass, friction) body
            sim_client = GraspSimulationClient(body, False)
            # print(sim_client._get_object_urdf(body))
            mesh = sim_client.mesh
            points, indices = mesh.sample(n_samples, return_index=True)
            points = np.array(points).reshape(-1, 3)
            normals = np.array(sim_client.mesh.face_normals[indices, :]).reshape(-1, 3)

            sim_client.disconnect()
            bad = False
            angles = []
            final_positions = []
            final_z_rotations = [] 
            push_velocities = []
            contact_points = []
            normal_vectors = []
            initials = []
            for push_data in object_data:
                # input_params, (success, logs) = push_data
                if len(push_data[0]) == 6: 
                    ((angle, contact_point, normal_vector, body, push_velocity, initial), (transformation, fallen)) = push_data
                else: 
                    ((angle, contact_point, normal_vector, body, push_velocity, initial, rotation_angle), (transformation, fallen)) = push_data

                if contact_point is None or fallen:
                    bad = True
                    break
                push_velocities.append(push_velocity)
                final_positions.append(transformation[:3, 3])
                final_z_rotations.append(R.from_matrix(transformation[:3, :3]).as_euler('xyz', degrees=False)[2]) 
                initials.append(R.from_matrix(initial[:3, :3]).as_euler('xyz', degrees=False)[2])

                if len(np.array(contact_point).shape) > 1: 
                    contact_point = np.mean(np.array(contact_point), axis=0) 
                    normal_vector = np.mean(np.array(normal_vector), axis=0)
                normal_vectors.append(normal_vector)
                contact_points.append(contact_point)
                angles.append(angle)
            if bad:
                continue

            normal_vector_data.append(normal_vectors)
            contact_point_data.append(contact_points)
            push_velocity_data.append(push_velocities)
            angle_data.append(angles)
            final_position_data.append(final_positions)
            initial_data.append(initials)
            final_z_rotation_data.append(final_z_rotations)

            mesh_data.append(points)
            normal_data.append(normals)
            com_data.append(com)
            mass_data.append(mass)
            friction_data.append(friction)
            name_data.append(name)
            # trajectory_data.append(trajectories)

        new_data = {}


        new_data["normals"] = np.array(normal_data) # DONE
        new_data["mesh"] = np.array(mesh_data) # DONE
        new_data["friction"] = np.array(friction_data) #DONE
        new_data["com"] = np.array(com_data) # DONE
        new_data["mass"] = np.array(mass_data) # DONE
        
        new_data["final_z_rotation"] = np.array(final_z_rotation_data)
        new_data["angle"] = np.array(angle_data) # DONE
        new_data["final_position"] = np.array(final_position_data) 
        new_data["push_velocities"] = np.array(push_velocity_data) # DONE
        new_data["contact_points"] = np.array(contact_point_data) # DONE
        new_data["normal_vector"] = np.array(normal_vector_data) # DONE
        new_data["initials"] = np.array(initial_data) 

        # new_data["orientation"] = np.array(quaternion_data) 
        # new_data["euler_angles"] = np.array(euler_angle_data) 

        # new_data["trajectory_data"] = np.array(trajectory_data)

        # Standardizing the data
        self.data = {}
        self.data["initials"] = new_data["initials"] 
        print(f"shape of initials: {self.data['initials'].shape}")
        self.data["final_z_rotation"] = new_data["final_z_rotation"] 
        print(f"shape of final_z_rotation: {self.data['final_z_rotation'].shape}")

        scaler = StandardScaler()

        # self.data["orientation"] = new_data["orientation"]

        contact_point_shape = new_data["contact_points"].shape
        contact_point_flat = new_data["contact_points"].reshape(-1, 3)
        self.data["contact_points_min"] = np.zeros(3)
        self.data["contact_points_max"] = np.zeros(3)
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            contact_point_flat[:, i] = min_max_scaler.fit_transform(
                contact_point_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["contact_points_min"][i] = min_max_scaler.data_min_ 
            self.data["contact_points_max"][i] = min_max_scaler.data_max_ 

        self.data["contact_points"] = contact_point_flat.reshape(contact_point_shape)
        print(f"shape of contact_points: {self.data['contact_points'].shape}")

        # Normalize angles, friction, mass, final positions, trajectory data, and push velocities
        for key in ["angle", "friction", "mass", "push_velocities"]:
            flat_value = new_data[key].reshape(-1, 1)
            standardized_value = scaler.fit_transform(flat_value).reshape(
                new_data[key].shape
            )
            self.data[key] = standardized_value
            print(f"shape of {key}: {self.data[key].shape}")

        # Normalize mesh data by scaling each dimension (x, y, z) separately
        mesh_shape = new_data["mesh"].shape
        mesh_flat = new_data["mesh"].reshape(-1, 3)
        self.data["mesh_min"] = np.zeros(3)
        self.data["mesh_max"] = np.zeros(3)
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            mesh_flat[:, i] = min_max_scaler.fit_transform(
                mesh_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["mesh_min"][i] = min_max_scaler.data_min_
            self.data["mesh_max"][i] = min_max_scaler.data_max_ 
        self.data["mesh"] = mesh_flat.reshape(mesh_shape)
        print(f"shape of mesh: {self.data['mesh'].shape}")

        # Normalize center of mass (com) by scaling each dimension (x, y, z) separately
        com_shape = new_data["com"].shape
        com_flat = new_data["com"].reshape(-1, 3)
        self.data["com_min"] = np.zeros(3)
        self.data["com_max"] = np.zeros(3)   
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            com_flat[:, i] = min_max_scaler.fit_transform(
                com_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["com_min"][i] = min_max_scaler.data_min_
            self.data["com_max"][i] = min_max_scaler.data_max_
        self.data["com"] = com_flat.reshape(com_shape)
        print(f"shape of com: {self.data['com'].shape}")

        # Normalize normals to unit vectors
        norms = np.linalg.norm(new_data["normals"], axis=2, keepdims=True)
        normalized_normals = new_data["normals"] / norms
        self.data["normals"] = normalized_normals
        print(f"shape of normals: {self.data['normals'].shape}")

        final_positions_shape = new_data["final_position"].shape
        final_positions_flat = new_data["final_position"].reshape(-1, 3)
        self.data["final_position_min"] = np.zeros(7)
        self.data["final_position_max"] = np.zeros(7)
        for i in range(3):
            min_max_scaler = MinMaxScaler()
            final_positions_flat[:, i] = min_max_scaler.fit_transform(
                final_positions_flat[:, i].reshape(-1, 1)
            ).flatten()
            self.data["final_position_min"][i] = min_max_scaler.data_min_
            self.data["final_position_max"][i] = min_max_scaler.data_max_ 
        self.data["final_position"] = final_positions_flat.reshape(
            final_positions_shape
        )
        print(f"shape of final_positions: {self.data['final_position'].shape}")

        # trajectories_shape = new_data["trajectory_data"].shape
        # trajectories_flat = new_data["trajectory_data"].reshape(-1, 7)
        # self.data["trajectory_data_min"] = np.zeros(7)
        # self.data["trajectory_data_max"] = np.zeros(7) 
        # for i in range(7):
        #     min_max_scaler = MinMaxScaler()
        #     trajectories_flat[:, i] = min_max_scaler.fit_transform(
        #         trajectories_flat[:, i].reshape(-1, 1)
        #     ).flatten()

        #     self.data["trajectory_min"] = min_max_scaler.data_min_
        #     self.data["trajectory_max"] = min_max_scaler.data_max_ 
        # self.data["trajectory_data"] = trajectories_flat.reshape(trajectories_shape)
        # print(f"shape of trajectories: {self.data['trajectory_data'].shape}")
        self.data["normal_vector"] = new_data["normal_vector"] 
        print(f"shape of normal_vector: {self.data['normal_vector'].shape}")

        # self.data["euler_angles"] = new_data["euler_angles"]

    def __getitem__(self, idx):
        item = {}
        for key, value in self.data.items():
            if key[-3:] != "min" and key[-3:] != "max":
                item[key] = value[idx]
            else: 
                item[key] = value 
        return item

    def __len__(self):
        return self.data["angle"].shape[0]


def main(args):
    file_name = os.path.join(
        "learning", "data", "pushing", args.file_name, "train_dataset.pkl"
    )

    # file_name = os.path.join("learning", "data", "pushing", "shapenet_only", "train_dataset.pkl")
    # print(file_name)
    train_dataset = PushNPDataset(file_name, args.n_samples)


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
    args = parser.parse_args()
    main(args)
