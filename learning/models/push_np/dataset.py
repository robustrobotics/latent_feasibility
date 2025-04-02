import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import trimesh
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pb_robot.planners.antipodalGraspPlanner import GraspSimulationClient, GraspableBody
from scipy.spatial.transform import Rotation as R


def quat_to_euler(logs): 
    ret = []
    for log in logs: 
        q = log[3:] 
        ret.append(R.from_quat(q).as_euler('xyz', degrees=False)) 
    return ret 


def collate_fn(items):
    total_dict = {}
    for item in items:
        for key in item.keys():
            if key not in total_dict:
                total_dict[key] = []
            if key.endswith("min") or key.endswith("max"):
                total_dict[key] = item[key]
        break
    for item in items:
        for key, value in item.items():
            if not (key.endswith("min") or key.endswith("max")):
                total_dict[key].append(value)
    for key, value in total_dict.items():
        total_dict[key] = torch.tensor(value)
    return total_dict


class PushNPDataset(Dataset):
    """
    Specific dataset for the PushNP. This class allows for easy inversion of transformed data back to its original scale.
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

        for idx, object_data in enumerate(tqdm(data)):
            # if idx > 10: 
            #     break
            body = object_data[0][0][3]  # Get the body
            if body is None: 
                continue
            name, com, mass, friction = body 

            sim_client = GraspSimulationClient(body, False)
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
                if len(push_data[0]) == 6: 
                    ((angle, contact_point, normal_vector, body, push_velocity, initial), (transformation, fallen)) = push_data
                else: 
                    ((angle, contact_point, normal_vector, body, push_velocity, initial, rotation_angle), (transformation, fallen)) = push_data

                if contact_point is None or fallen: 
                    bad = True 
                    break 
                final_x = transformation[0, 3] 
                final_y = transformation[1, 3] 
                if (final_x ** 2 + final_y ** 2) > 1.21:
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

        new_data = {}

        # Assign raw data
        new_data["normals"] = np.array(normal_data)  # DONE
        new_data["mesh"] = np.array(mesh_data)        # DONE
        new_data["friction"] = np.array(friction_data)  # DONE
        new_data["com"] = np.array(com_data)            # DONE
        new_data["mass"] = np.array(mass_data)          # DONE
        new_data["final_z_rotation"] = np.array(final_z_rotation_data)
        new_data["angle"] = np.array(angle_data)        # DONE
        new_data["final_position"] = np.array(final_position_data) 
        new_data["push_velocities"] = np.array(push_velocity_data)  # DONE
        new_data["contact_points"] = np.array(contact_point_data)    # DONE
        new_data["normal_vector"] = np.array(normal_vector_data)      # DONE
        new_data["initials"] = np.array(initial_data)                # DONE

        # Initialize dictionaries to store scaling parameters
        self.data = {}
        self.scalers = {}  # To store scalers for inversion

        # Define which fields are scaled with which scaler
        self.minmax_scaled_fields = ["contact_points", "mesh", "com", "final_position", "normal_vector", "angle"]  
        self.standard_scaled_fields = ["friction", "mass", "push_velocities"]

        # Process fields scaled with MinMaxScaler
        for field in self.minmax_scaled_fields:
            field_min_key = f"{field}_min"
            field_max_key = f"{field}_max"
            print(f"Processing {field} with shape: {new_data[field].shape}")
            
            # Special handling for angle field
            if field == "angle":
                # For angle, we only need size 1 arrays
                self.data[field_min_key] = np.zeros(1)
                self.data[field_max_key] = np.zeros(1)
                
                # Reshape angle data for scaling
                field_shape = new_data[field].shape
                field_flat = new_data[field].reshape(-1, 1)
                scaled_flat = np.zeros_like(field_flat)
                
                # Scale the angle data
                scaler = MinMaxScaler()
                scaled_flat = scaler.fit_transform(field_flat)
                self.data[field_min_key][0] = scaler.data_min_
                self.data[field_max_key][0] = scaler.data_max_
                self.scalers[field_min_key] = self.data[field_min_key]
                self.scalers[field_max_key] = self.data[field_max_key]
                
                # Reshape back to original
                self.data[field] = scaled_flat.reshape(field_shape)
            else:
                # For other fields, use the last dimension size
                n_features = new_data[field].shape[-1]
                self.data[field_min_key] = np.zeros(n_features)
                self.data[field_max_key] = np.zeros(n_features)
                
                # Flatten the data for scaling
                field_shape = new_data[field].shape
                field_flat = new_data[field].reshape(-1, field_shape[-1])
                scaled_flat = np.zeros_like(field_flat)
                
                # Scale each feature independently
                for i in range(field_flat.shape[1]):
                    scaler = MinMaxScaler()
                    scaled_flat[:, i] = scaler.fit_transform(field_flat[:, i].reshape(-1, 1)).flatten()
                    self.data[field_min_key][i] = scaler.data_min_
                    self.data[field_max_key][i] = scaler.data_max_
                    self.scalers[field_min_key] = self.data[field_min_key]
                    self.scalers[field_max_key] = self.data[field_max_key]
                
                # Reshape back to original
                self.data[field] = scaled_flat.reshape(field_shape)
            
            print(f"Processed {field} -> Output shape: {self.data[field].shape}")

        # Process fields scaled with StandardScaler
        for field in self.standard_scaled_fields:
            scaler = StandardScaler()
            field_shape = new_data[field].shape
            field_flat = new_data[field].reshape(-1, 1)
            scaled_flat = scaler.fit_transform(field_flat).reshape(field_shape)
            self.data[field] = scaled_flat
            self.scalers[f"{field}_mean"] = scaler.mean_
            self.scalers[f"{field}_scale"] = scaler.scale_
            print(f"Shape of {field}: {self.data[field].shape}")

        # Process fields that were not scaled but transformed (e.g., normalization)
        self.data["initials"] = new_data["initials"] 
        print(f"Shape of initials: {self.data['initials'].shape}")
        self.data["final_z_rotation"] = new_data["final_z_rotation"] 
        print(f"Shape of final_z_rotation: {self.data['final_z_rotation'].shape}")

        # Normalize normals to unit vectors (already normalized, so no inversion needed)
        norms = np.linalg.norm(new_data["normals"], axis=2, keepdims=True)
        normalized_normals = new_data["normals"] / norms
        self.data["normals"] = normalized_normals
        print(f"Shape of normals: {self.data['normals'].shape}")

        # Normalize normal vectors to unit vectors
        norms = np.linalg.norm(new_data["normal_vector"], axis=2, keepdims=True)
        normalized_normal_vectors = new_data["normal_vector"] / norms
        self.data["normal_vector"] = normalized_normal_vectors
        print(f"Shape of normal_vector: {self.data['normal_vector'].shape}")

        # Assign other fields without scaling
        # If needed, you can add more fields here

    def __getitem__(self, idx):
        item = {}
        for key, value in self.data.items():
            if key.endswith("min") or key.endswith("max"):
                item[key] = value
            else:
                item[key] = value[idx]
        return item

    def __len__(self):
        return self.data["angle"].shape[0]

    def inverse_transform(self, transformed_item):
        """
        Invert the transformations applied to the data to retrieve the original scale.

        Args:
            transformed_item (dict): A dictionary containing transformed data and scaling parameters.

        Returns:
            dict: A dictionary containing the original untransformed data.
        """
        original_item = {}

        for key, value in transformed_item.items():
            if key.endswith("_min") or key.endswith("_max"):
                # These are scaling parameters; skip them
                continue
            elif key in self.standard_scaled_fields:
                # Invert StandardScaler
                mean = self.scalers.get(f"{key}_mean")
                scale = self.scalers.get(f"{key}_scale")
                if mean is None or scale is None:
                    raise ValueError(f"Scaler parameters for {key} not found.")
                # Ensure the value is a numpy array
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                original = value * scale + mean
                original_item[key] = original
            elif key in self.minmax_scaled_fields:
                # Invert MinMaxScaler
                min_vals = self.scalers.get(f"{key}_min")
                max_vals = self.scalers.get(f"{key}_max")
                if min_vals is None or max_vals is None:
                    raise ValueError(f"Scaler parameters for {key} not found.")
                # Ensure the value is a numpy array
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                # print(key, value, min_vals, max_vals)
                original = value * (max_vals - min_vals) + min_vals
                original_item[key] = original
            else:
                # Fields that were not scaled; return as-is
                if isinstance(value, torch.Tensor):
                    original_item[key] = value.cpu().numpy()
                else:
                    original_item[key] = value

        return original_item
