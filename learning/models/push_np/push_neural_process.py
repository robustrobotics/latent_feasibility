import torch
import torch.nn as nn

from learning.models.pointnet import PointNetRegressor


class PushNP(nn.Module):
    """
    Neural network model for predicting the final position of an object after a series of pushes.

    Args:
        input_features (list): List of input features to use for prediction.
        d_output (int): Dimensionality of the output.
        d_latents (int): Dimensionality of the latent space.
    """

    def __init__(self, input_features, d_output, d_latents):
        super().__init__()

        self.input_features = input_features

        n_mesh_features = 6  # The points, as well as the normals
        n_geom_features = 2  # Same as ^
        n_push_features = 5
        self.n_obj_features = 5

        conv_1_sz = 64
        conv_2_sz = 256
        conv_3_sz = 1024
        point_net_output_sz = 512
        linear_input_sz = conv_3_sz + point_net_output_sz
        linear_layer_1_sz = 256
        linear_layer_2_sz = 128
        linear_layer_3_sz = d_latents * (d_latents + 1) // 2 + d_latents

        decoder_layer_1_sz = 256
        decoder_layer_2_sz = 1024
        decoder_layer_3_sz = d_output * (d_output + 1) // 2 + d_output

        self.linear_layer1 = nn.Linear(linear_input_sz, linear_layer_1_sz)
        self.linear_layer2 = nn.Linear(linear_layer_1_sz, linear_layer_2_sz)
        self.linear_layer3 = nn.Linear(linear_layer_2_sz, linear_layer_3_sz)
        self.point_net_encoder = PointNetRegressor(
            n_mesh_features, point_net_output_sz, n_geometric_features=n_geom_features
        )

        self.conv_layer_1 = nn.Conv1d(n_push_features, conv_1_sz, 1)
        self.conv_layer_2 = nn.Conv1d(conv_1_sz, conv_2_sz, 1)
        self.conv_layer_3 = nn.Conv1d(conv_2_sz, conv_3_sz, 1)

        self.decoder_layer_1 = nn.Linear(
            d_latents
            + n_push_features
            + (self.n_obj_features if "com" in input_features else 0),
            decoder_layer_1_sz,
        )
        self.decoder_layer_2 = nn.Linear(decoder_layer_1_sz, decoder_layer_2_sz).float()
        self.decoder_layer_3 = nn.Linear(decoder_layer_2_sz, decoder_layer_3_sz).float()
        print(f"Decoder layer 3 sz: {self.decoder_layer_3.weight.shape}")
        print(f"Convolution Layer 3 sz: {self.conv_layer_3.weight.shape}")

        self.d_latents = d_latents
        self.input_features = input_features
        self.d_output = d_output

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        q_zs = self.get_latent_space(x)
        (mesh_data, obj_data, push_data, n_push_data) = x

        zs = torch.stack([q_z.rsample() for q_z in q_zs]).to(mesh_data.device)
        zs = zs.to(mesh_data.device)

        latent_size = self.d_latents
        if "com" in self.input_features:
            zs = torch.cat([zs, obj_data], dim=1)
            latent_size += self.n_obj_features

        zs = zs.unsqueeze(2)  # Now zs has shape [5, 10, 1]
        zs = zs.expand(
            zs.shape[0], latent_size, push_data.shape[1]
        )  # Now zs has shape [5, 10, 1]
        zs = zs.swapaxes(1, 2)
        # print(f"zs sz: {zs.shape}")

        input_vector = torch.cat([push_data, zs], dim=2)
        input_vector = self.relu(self.decoder_layer_1(input_vector))
        input_vector = self.relu(self.decoder_layer_2(input_vector))
        input_vector = self.decoder_layer_3(input_vector)

        # print(f"input_vector sz: {input_vector.shape}")
        mu = input_vector[:, :, : self.d_output]
        sigma = input_vector[:, :, self.d_output :]

        tril_indices = torch.tril_indices(
            self.d_output, self.d_output, device=sigma.device
        )

        output = torch.zeros(
            (
                input_vector.shape[0],
                input_vector.shape[1],
                self.d_output,
                self.d_output,
            ),
            device=sigma.device,
        )

        for i in range(input_vector.shape[0]):
            for k in range(input_vector.shape[1]):
                output[i, k, tril_indices[0], tril_indices[1]] = sigma[i, k]
                diag_indices = range(self.d_output)
                output[i, k, diag_indices, diag_indices] = self.softplus(
                    output[i, k, diag_indices, diag_indices]
                )

        result = [
            [
                torch.distributions.MultivariateNormal(mu, scale_tril=output[i][j])
                for j in range(mu.shape[1])
            ]
            for i in range(mu.shape[0])
        ]
        return result

    def get_latent_space(self, x):
        (mesh_data, _, _, n_push_data) = x
        mesh_vector, _ = self.point_net_encoder(mesh_data.transpose(1, 2).float())
        push_vector = self.relu(self.conv_layer_1(n_push_data.swapaxes(1, 2).float()))
        push_vector = self.relu(self.conv_layer_2(push_vector))
        push_vector = self.conv_layer_3(push_vector)
        push_vector = torch.max(push_vector, dim=2)[0]

        input_vector = torch.cat([mesh_vector, push_vector], dim=1)
        results = self.relu(self.linear_layer1(input_vector))
        results = self.relu(self.linear_layer2(results))
        results = self.linear_layer3(results)

        mu = results[:, : self.d_latents]
        variance_matrix = results[:, self.d_latents :]
        tril_indices = torch.tril_indices(
            self.d_latents, self.d_latents, device=variance_matrix.device
        )

        output = torch.zeros(
            (results.shape[0], self.d_latents, self.d_latents),
            device=variance_matrix.device,
        )
        for i in range(results.shape[0]):
            output[i, tril_indices[0], tril_indices[1]] = variance_matrix[i]
            diag_indices = range(self.d_latents)
            output[i, diag_indices, diag_indices] = self.softplus(
                output[i, diag_indices, diag_indices]
            )

        return [
            torch.distributions.MultivariateNormal(mu[i], scale_tril=output[i])
            for i in range(mu.shape[0])
        ]
