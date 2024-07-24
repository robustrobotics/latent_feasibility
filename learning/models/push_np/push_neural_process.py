import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from learning.models.pointnet import PointNetRegressor

class PushNP(nn.Module):
    """
    Neural network model for predicting the final position of an object after a series of pushes.

    Args:
        input_features (List[str]): List of input features to use for prediction.
        d_output (int): Dimensionality of the output.
        d_latents (int): Dimensionality of the latent space.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
    """

    def __init__(
        self,
        input_features: List[str],
        d_output: int,
        d_latents: int,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.input_features = input_features
        self.d_output = d_output
        self.d_latents = d_latents

        # Constants
        self.N_MESH_FEATURES = 6
        self.N_GEOM_FEATURES = 2
        self.N_PUSH_FEATURES = 5 + 3
        self.N_PUSH_FEATURES_LATENT = (
            self.N_PUSH_FEATURES + 7
        )  # 5 for the push data, 7 for the final_position
        self.N_OBJ_FEATURES = 5

        # Layer sizes
        conv_sizes = [64, 256, 512]
        point_net_output_sz = 512
        linear_sizes = [
            conv_sizes[-1] + point_net_output_sz,
            256,
            128,
            d_latents * (d_latents + 1) // 2 + d_latents,
        ]
        decoder_sizes = [256, 512, d_output * (d_output + 1) // 2 + d_output]

        # Encoder layers
        self.point_net_encoder = PointNetRegressor(
            self.N_MESH_FEATURES,
            point_net_output_sz,
            n_geometric_features=self.N_GEOM_FEATURES,
        )
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(self.N_PUSH_FEATURES_LATENT, conv_sizes[0], 1),
                nn.Conv1d(conv_sizes[0], conv_sizes[1], 1),
                nn.Conv1d(conv_sizes[1], conv_sizes[2], 1),
            ]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(linear_sizes[i], linear_sizes[i + 1])
                for i in range(len(linear_sizes) - 1)
            ]
        )

        # Decoder layers
        decoder_input_size = (
            d_latents
            + self.N_PUSH_FEATURES
            + (self.N_OBJ_FEATURES if "com" in input_features else 0)
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.Linear(decoder_input_size, decoder_sizes[0]),
                nn.Linear(decoder_sizes[0], decoder_sizes[1]),
                nn.Linear(decoder_sizes[1], decoder_sizes[2]),
            ]
        )

        self.activation = activation or nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[List[List[torch.distributions.MultivariateNormal]], torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input data.

        Returns:
            Tuple[List[List[torch.distributions.MultivariateNormal]], torch.Tensor]: Predicted distributions and means.
        """
        q_zs = self.get_latent_space(x)
        mesh_data, obj_data, push_data, _ = x

        # Simplified versions of the model for debugging:
        # result = [[torch.distributions.MultivariateNormal(torch.ones(self.d_output, device=mesh_data.device, requires_grad=True), torch.eye(self.d_output, device=mesh_data.device, requires_grad=True)) for j in range(push_data.shape[1])] for i in range(push_data.shape[0])], None, q_zs
        # return result

        zs = torch.stack([q_z.rsample() for q_z in q_zs]).to(mesh_data.device)

        latent_size = self.d_latents
        if "com" in self.input_features:
            zs = torch.cat([zs, obj_data], dim=1)
            latent_size += self.N_OBJ_FEATURES

        zs = zs.unsqueeze(2).expand(-1, latent_size, push_data.shape[1])
        zs = zs.transpose(1, 2)

        input_vector = torch.cat([push_data, zs], dim=2)
        for i, layer in enumerate(self.decoder_layers):
            input_vector = layer(input_vector)
            if i < len(self.decoder_layers) - 1:
                input_vector = self.activation(input_vector)

        mu = input_vector[:, :, : self.d_output]
        sigma = input_vector[:, :, self.d_output :]

        tril_indices = torch.tril_indices(
            self.d_output, self.d_output, device=sigma.device
        )
        output = torch.zeros(
            input_vector.shape[0],
            input_vector.shape[1],
            self.d_output,
            self.d_output,
            device=sigma.device,
        )
        output[:, :, tril_indices[0], tril_indices[1]] = sigma
        output[:, :, range(self.d_output), range(self.d_output)] = self.softplus(
            output[:, :, range(self.d_output), range(self.d_output)]
        )

        result = [
            [
                torch.distributions.MultivariateNormal(
                    mu[i][j], scale_tril=output[i][j]
                )
                for j in range(mu.shape[1])
            ]
            for i in range(mu.shape[0])
        ]
        return result, mu, q_zs

    def get_latent_space(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[torch.distributions.MultivariateNormal]:
        """
        Compute the latent space representation.

        Args:
            x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input data.

        Returns:
            List[torch.distributions.MultivariateNormal]: Latent space distributions.
        """

        mesh_data, _, _, n_push_data = x
        # return [torch.distributions.MultivariateNormal(torch.ones(self.d_latents), covariance_matrix=torch.eye(self.d_latents)) for i in range(mesh_data.shape[0])]
        mesh_vector, _ = self.point_net_encoder(mesh_data.transpose(1, 2).float())

        push_vector = n_push_data.transpose(1, 2).float()
        for conv_layer in self.conv_layers:
            push_vector = self.activation(conv_layer(push_vector))
        # push_vector = torch.max(push_vector, dim=2)[0]
        push_vector = torch.sum(push_vector, dim=2)  # Try to aggregate with sum instead of max 

        input_vector = torch.cat([mesh_vector, push_vector], dim=1)
        for i, layer in enumerate(self.linear_layers):
            input_vector = layer(input_vector)
            if i < len(self.linear_layers) - 1:
                input_vector = self.activation(input_vector)

        mu = input_vector[:, : self.d_latents]
        variance_matrix = input_vector[:, self.d_latents :]

        tril_indices = torch.tril_indices(
            self.d_latents, self.d_latents, device=variance_matrix.device
        )
        output = torch.zeros(
            input_vector.shape[0],
            self.d_latents,
            self.d_latents,
            device=variance_matrix.device,
        )
        output[:, tril_indices[0], tril_indices[1]] = variance_matrix
        output[:, range(self.d_latents), range(self.d_latents)] = self.softplus(
            output[:, range(self.d_latents), range(self.d_latents)]
        )

        return [
            torch.distributions.MultivariateNormal(mu[i], scale_tril=output[i])
            for i in range(mu.shape[0])
        ]
