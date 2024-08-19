import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


from learning.models.pointnet import PointNetRegressor


class AttentionPushNP(nn.Module):
    def __init__(self, args):
        """
        Initializes the AttentionPushNP model.

        This model is used for predicting the feasibility of pushing actions in a simulated environment.
        It consists of a PointNet regressor for encoding mesh features, an APNPEncoderLatent for encoding latent variables,
        an APNPEncoderDeterministic for encoding deterministic variables, and an APNPDecoder for decoding the output.

        Args:
            args: An object containing the arguments for the model.

        """
        super().__init__()
        self.N_MESH_FEATURES = 6
        self.POINT_NET_ENCODING_SIZE = 64 
        self.N_GEOM_FEATURES = 2
        self.point_net_encoder = PointNetRegressor(
            self.N_MESH_FEATURES,
            self.POINT_NET_ENCODING_SIZE,
            n_geometric_features=self.N_GEOM_FEATURES,
            use_stn=False,
            use_batch_norm=False,
        )
        self.latent_encoder = APNPEncoderLatent(
            args,
            self.POINT_NET_ENCODING_SIZE,
            dropout=args.dropout,
            d_latents=args.d_latents,
            embed_dimension=args.attention_encoding,
        )
        self.deterministic_encoder = APNPEncoderDeterministic(
            args, dropout=args.dropout, embed_dimension=args.attention_encoding
        )
        self.decoder = APNPDecoder(args, d_latents=args.d_latents, point_net_encoding_size=self.POINT_NET_ENCODING_SIZE)

    def forward(
        self, context_x, context_y, target_x, target_y, mesh, obj, mode="train"
    ):
        mesh_vector, _ = self.point_net_encoder(mesh.transpose(1, 2))
        if mode == "train":  # Training
            qzs = self.latent_encoder(target_x, target_y, mesh_vector)
            qz_contexts = self.latent_encoder(context_x, context_y, mesh_vector)
        else:
            qzs = self.latent_encoder(context_x, context_y, mesh_vector)

        # print(qzs)

        deterministic = self.deterministic_encoder(context_x, context_y, target_x)
        # print(torch.max(deterministic[0][0] - deterministic[0][1]))

        distributions, mu, sigma = self.decoder(
            target_x, deterministic, qzs, mesh_vector, obj
        )

        bce_loss = self.bce_loss_fn(distributions, target_y)
        kl_loss = None
        entropy = None 
        total_loss = bce_loss

        if mode == "train":
            kl_loss = self.kl_loss_fn(qzs, qz_contexts)
            total_loss = kl_loss + total_loss 
        else: # We assume this is validation.
            entropy = torch.stack([d.entropy() for d in qzs]).mean()
        distance = self.average_distance(distributions, target_y)

        # print(total_loss, bce_loss)

        return total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy 

    def bce_loss_fn(self, distributions, target_y):
        loss = 0
        for i in range(target_y.shape[0]):
            for j in range(target_y.shape[1]):
                if torch.rand(1) < 0.0001: 
                    print(torch.exp(distributions[i][j].log_prob(target_y[i][j])), distributions[i][j].mean, target_y[i][j])
                loss -= distributions[i][j].log_prob(target_y[i][j])
        return loss / target_y.shape[0] / target_y.shape[1]

    def kl_loss_fn(self, qzs, qz_contexts):
        loss = 0
        for i in range(len(qzs)):
            loss += torch.distributions.kl_divergence(qzs[i], qz_contexts[i])
        return loss / len(qzs)

    def average_distance(self, distributions, target_y):
        distance = 0
        for i in range(target_y.shape[0]):
            for j in range(target_y.shape[1]):
                distance += torch.sqrt(
                    F.mse_loss(
                        distributions[i][j].mean[:3],
                        target_y[i][j][:3],
                        reduction="sum",
                    )
                )
        return distance / target_y.shape[0] / target_y.shape[1]


class APNPEncoderLatent(nn.Module):
    def __init__(
        self,
        args,
        point_net_encoding_size,
        embed_dimension=512,
        h=8,
        dropout=0.05,
        d_latents=8,
    ):
        """
        The latent variable side of the encoder. Takes in a amount of pushes 
        and turns them into a latent distribution to represent the uncertainty in the 
        pushes. In practice during training, this has not shown to be super reliable.

        Looks like the model relies more on the determinimistic side of the encoder. 
        Args: 
            args: The arguments for the model (passed down from APNP)
            point_net_encoding_size: The size of the point net encoding
            embed_dimension: The size of the embedding
            h: The number of heads in the multihead attention
            dropout: The dropout rate
            d_latents: The size of the latent distribution

        """
        super().__init__()

        self.POINT_NET_ENCODING_SIZE = point_net_encoding_size
        if args.use_full_trajectory:
            self.PUSH_DIMENSION = (
                5 + 3 + 4 + 7 * 50 + self.POINT_NET_ENCODING_SIZE
            )  # defunct case
        else:
            self.PUSH_DIMENSION = 9 + 4 + self.POINT_NET_ENCODING_SIZE
        self.EMBED_DIMENSION = embed_dimension
        self.HEADS = h

        self.first = nn.Linear(self.PUSH_DIMENSION, self.EMBED_DIMENSION)
        self.multihead = nn.ModuleList(
            nn.MultiheadAttention(
                embed_dim=self.EMBED_DIMENSION,
                num_heads=self.HEADS,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(2)
        )
        self.penultimate = nn.Sequential(
            nn.Linear(self.EMBED_DIMENSION, self.EMBED_DIMENSION), nn.ReLU()
        )
        self.mean = nn.Linear(self.EMBED_DIMENSION, d_latents)
        self.log_var = nn.Linear(self.EMBED_DIMENSION, d_latents)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, context_x, context_y, mesh_encoding):
        (B, N, D) = context_x.shape
        mesh_encoding = mesh_encoding.unsqueeze(1)
        mesh_encoding = mesh_encoding.expand(B, N, self.POINT_NET_ENCODING_SIZE)
        pushes = torch.cat([context_x, context_y, mesh_encoding], dim=2)

        pushes = self.first(pushes)
        # [B, N, D] -> [N, B, D]
        # print(self.multihead[0].device)
        # print(pushes.device)
        # print(pushes)
        for i in range(2):
            pushes = self.multihead[i](pushes, pushes, pushes)[0]

        pushes = pushes.mean(dim=1)
        # print(pushes)
        pushes = self.penultimate(pushes)
        mean = self.mean(pushes)
        log_var = self.log_var(pushes)
        # var = torch.exp(log_var)
        var = 0.1 + 0.9 * self.sigmoid(log_var) 

        # print(mean)

        return [
            torch.distributions.MultivariateNormal(
                mean[i], covariance_matrix=torch.diag(var[i])
            )
            for i in range(B)
        ]


class APNPEncoderDeterministic(nn.Module):

    def __init__(self, args, embed_dimension=512, h=8, dropout=0.05):
        """
        Here we basically just want to use cross-attention on the target queries and compare them to context. 
        This seems like it would be a pretty effective idea since it would allow the model to learn the connection 
        between the contexts and the targets.

        Args:
            args: The arguments for the model
            embed_dimension: The size of the embedding
            h: The number of heads in the multihead attention
            dropout: The dropout rate
        """
        super().__init__()
        self.EMBED_DIMENSION = embed_dimension
        self.FINAL = 8
        self.HEADS = h
        self.X_DIMENSION = 9
        if args.use_full_trajectory:
            self.CONTEXT_DIMENSION = self.X_DIMENSION + 7 * 50
        else:
            self.CONTEXT_DIMENSION = self.X_DIMENSION + 4
        self.to_embed = nn.Linear(self.CONTEXT_DIMENSION, self.EMBED_DIMENSION)
        self.to_query = nn.Linear(self.X_DIMENSION, self.EMBED_DIMENSION)
        self.to_key = nn.Linear(self.X_DIMENSION, self.EMBED_DIMENSION)
        self.self_attention_heads = nn.ModuleList(
            nn.MultiheadAttention(
                self.EMBED_DIMENSION, self.HEADS, batch_first=True, dropout=dropout
            )
            for _ in range(2)
        )
        self.cross_attention_heads = nn.ModuleList(
            nn.MultiheadAttention(
                self.EMBED_DIMENSION, self.HEADS, batch_first=True, dropout=dropout
            )
            for _ in range(2)
        )
        self.shorten = nn.Linear(self.EMBED_DIMENSION, self.FINAL)

    def forward(self, context_x, context_y, target_x):
        # print("HERE")
        context = torch.cat([context_x, context_y], dim=2)
        context = self.to_embed(context)

        for i in range(2):
            context = self.self_attention_heads[i](context, context, context)[0]

        keys = self.to_key(context_x)
        queries = self.to_key(target_x)

        # print(queries.shape, keys.shape, context.shape)
        for i in range(2):
            queries = self.cross_attention_heads[i](
                query=queries, key=keys, value=context
            )[0]
            # print(queries.shape)

        queries = F.relu(queries)
        queries = self.shorten(queries) 
        return queries


class APNPDecoder(nn.Module):
    def __init__(
        self,
        args,
        embed_dimension=8,
        d_latents=5,
        point_net_encoding_size=512,
        output_size=4,
    ):
        """
        The decoder for the model which takes the latent variable and 
        the deterministic encoding for the target_x and predictions target_y. 
        Args: 
            args: The arguments for the model
            embed_dimension: The size of the embedding
            d_latents: The size of the latent distribution
            point_net_encoding_size: The size of the point net encoding
            output_size: The size of the output
        """
        super().__init__()

        self.args = args
        self.X_DIMENSION = 9
        self.OBJ_PROPERTIES = 5 if args.use_obj_prop else 0  # mass, friction, com
        self.INPUT_SIZE = (
            embed_dimension
            + d_latents
            + point_net_encoding_size
            + self.X_DIMENSION
            + self.OBJ_PROPERTIES
        )
        self.OUTPUT_SIZE = output_size
        self.POINT_NET_ENCODING_SIZE = point_net_encoding_size

        self.layer_sizes = [512, 128]
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, self.layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.layer_sizes[0], self.layer_sizes[1]),
            nn.ReLU(),
        )

        self.mean = nn.Linear(self.layer_sizes[1], output_size)
        self.var = nn.Linear(self.layer_sizes[1], output_size * (output_size + 1) // 2)

    def forward(
        self, target_x, deterministic, latent_distribution, mesh_vector, obj_data
    ):
        (B, N, D) = target_x.shape
        latent = torch.stack(
            [
                torch.stack([latent_distribution[i].rsample() for _ in range(N)])
                for i in range(B)
            ]
        ).to(target_x.device)
        mesh_vector = mesh_vector.unsqueeze(1)
        mesh_vector = mesh_vector.expand(B, N, self.POINT_NET_ENCODING_SIZE)

        input_vector = torch.cat([target_x, deterministic, latent, mesh_vector], dim=2)
        if self.args.use_obj_prop:
            obj_data = obj_data.unsqueeze(1)
            obj_data = obj_data.expand(B, N, obj_data.shape[2]).to(target_x.device)
            input_vector = torch.cat([input_vector, obj_data], dim=2)

        input_vector = self.layers(input_vector)

        mean = self.mean(input_vector)
        var = self.var(input_vector)

        output = torch.zeros(
            B, N, self.OUTPUT_SIZE, self.OUTPUT_SIZE, device=target_x.device
        )
        lower_triangular = torch.tril_indices(
            self.OUTPUT_SIZE, self.OUTPUT_SIZE, device=target_x.device
        )

        output[:, :, lower_triangular[0], lower_triangular[1]] = var
        output[:, :, range(self.OUTPUT_SIZE), range(self.OUTPUT_SIZE)] = F.softplus(
            output[:, :, range(self.OUTPUT_SIZE), range(self.OUTPUT_SIZE)]
        )

        return (
            [
                [
                    torch.distributions.MultivariateNormal(
                        mean[i][j], scale_tril=output[i][j]
                    )
                    for j in range(N)
                ]
                for i in range(B)
            ],
            mean,
            output @ output.transpose(-1, -2),
        )
