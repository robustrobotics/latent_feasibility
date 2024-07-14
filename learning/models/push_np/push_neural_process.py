import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.models.pointnet import PointNetRegressor


class PushNP(nn.Module):
    """
    Input Features tells the neural network which features to use!
    """

    def __init__(self, input_features, d_mesh_enc, d_latents):
        super().__init__()

        self.input_features = input_features

        n_mesh_features = 6  # The points, as well as the normals
        n_geom_features = 2  # Same as ^
        n_push_features = 5

        conv_1_sz = 64
        conv_2_sz = 256
        conv_3_sz = 1024
        self.linear_layer1 = nn.Linear(conv_3_sz, 256)
        self.linear_layer2 = nn.Linear(256, 128)
        self.linear_layer3 = nn.Linear(128, d_latents * (d_latents + 1) / 2 + d_latents)
        self.point_net_encoder = PointNetRegressor(
            n_mesh_features, d_mesh_enc, n_geometric_features=n_geom_features
        )  # We will only use this
        linear_input_sz = conv_3_sz + 1024
        if "com" in input_features:
            linear_input_sz += 5

        self.conv_layer_1 = nn.Conv1d(n_push_features, conv_1_sz, 1)
        self.conv_layer_2 = nn.Conv1d(conv_1_sz, conv_2_sz, 1)
        self.conv_layer_3 = nn.Conv1d(conv_2_sz, conv_3_sz, 1)
        self.d_latents = d_latents

    def forward(self, x):
        mu, sigma = self.get_latent_space(x)
        return None

    def get_latent_space(self, x):
        (mesh_data, obj_data, push_data) = x
        print(f"object data sz: {obj_data.shape}")
        print(f"mesh data sz: {mesh_data.shape}")
        mesh_vector, _ = self.point_net_encoder(mesh_data.transpose(1, 2).float())
        print(f"push_data sz: {push_data.shape}")
        push_vector = F.relu(self.conv_layer_1(push_data.swapaxes(1, 2).float()))
        push_vector = F.relu(self.conv_layer_2(push_vector))
        push_vector = self.conv_layer_3(push_vector)
        push_vector = torch.max(push_vector, dim=2, keepdim=True)[0]
        print(push_vector.shape)

        print(f"mesh vector sz: {mesh_vector.shape}")
        print(f"push vector sz: {push_vector.shape}")

        if "com" in self.input_features:
            input_vector = torch.cat([mesh_vector, push_vector, obj_data])
        else:
            input_vector = torch.cat([mesh_vector, push_vector])
        results = F.relu(self.linear_layer1(input_vector))
        results = F.relu(self.linear_layer2(results))
        results = self.linear_layer3(results)

        mu = results[: self.d_latents]
        variance_matrix = results[self.d_latents :]
        tril_indices = torch.tril_indices(self.d_latents, self.d_latents)

        output = torch.zeros((self.d_latents, self.d_latents), dtype=torch.float32)
        output[tril_indices[0], tril_indices[1]] = variance_matrix
        for i in range(self.d_latents):
            output[i][i] = F.softplus(output[i][i])
        sigma = torch.matmul(output, output.T)

        return mu, sigma
