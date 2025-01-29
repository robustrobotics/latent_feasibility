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
        if not args.no_deterministic:
            self.deterministic_encoder = APNPEncoderDeterministic(
                args, dropout=args.dropout, embed_dimension=args.attention_encoding
            )
        self.decoder = APNPDecoder(
            args,
            d_latents=args.d_latents,
            point_net_encoding_size=self.POINT_NET_ENCODING_SIZE,
            use_mixture=args.use_mixture,
        )
        self.args = args

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
        if not self.args.no_deterministic:
            deterministic = self.deterministic_encoder(context_x, context_y, target_x)
        else:
            deterministic = None
        # print(torch.max(deterministic[0][0] - deterministic[0][1]))

        if self.args.guess_obj:
            # guess = self.decoder(target_x, deterministic, qzs, mesh_vector, obj)
            guess = torch.stack([
                torch.stack([
                    qzs[i].mean for _ in range(target_x.shape[1])
                ]) for i in range(len(qzs))
            ])
            loss, loss_x, loss_y, loss_z = self.guess_obj_data(guess, obj)
            return (
                loss,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(
                    context_x.shape[0], context_x.shape[1], context_x.shape[2], 4
                ),
                torch.zeros(
                    context_x.shape[0], context_x.shape[1], context_x.shape[2], 4, 4
                ),
                torch.zeros(1),
                (loss_x, loss_y, loss_z),
            )

        if self.args.use_mixture:
            distributions, mu, sigma, mix = self.decoder(
                target_x, deterministic, qzs, mesh_vector, obj
            )
            bce_loss = self.bce_loss_mixture(distributions, target_y, mix)
            distance = self.average_distance(distributions, target_y, mix)
        else:
            distributions, mu, sigma = self.decoder(
                target_x, deterministic, qzs, mesh_vector, obj
            )
            bce_loss = self.bce_loss_fn(distributions, target_y, mu)
            distance = self.average_distance(distributions, target_y)
        kl_loss = None
        entropy = None
        total_loss = bce_loss

        if mode == "train":
            kl_loss = self.kl_loss_fn(qzs, qz_contexts)
            total_loss = kl_loss + total_loss
        else:  # We assume this is validation.
            entropy = torch.stack([d.entropy() for d in qzs]).mean()

        # print(total_loss, bce_loss)

        return total_loss, bce_loss, kl_loss, mu, sigma, distance, entropy

    # def sample_multiple(self, context_x, context_y, target_x, mesh, obj): 
    #     mesh_vector, _ = self.point_net_encoder(mesh.transpose(1, 2)) 
    #     qzs = self.latent_encoder(context_x, context_y, mesh_vector)

    #     # Now we need to sample multiple from qzs, we get 50 for each one

    #     samples = [qz.sample(sample_shape=torch.Size([target_x.shape[-1], 50])) for qz in qzs]


    def guess_obj_data(self, y, obj):
        # print(y.shape)
        # print(obj.shape)
        # print(obj.shape, y.shape)
        obj = obj.unsqueeze(1)
        obj = obj.expand(y.shape) 
        # print(y)
        # print(obj) 

        x = F.mse_loss(y[2], obj[2], reduction="mean") 
        y1 = F.mse_loss(y[3], obj[3], reduction="mean")
        z = F.mse_loss(y[4], obj[4], reduction="mean")
        return F.mse_loss(y[2:5], obj[2:5], reduction="mean"), x, y1, z

    def bce_loss_mixture(self, distributions, target_y, mix):
        loss = 0
        for i in range(target_y.shape[0]):
            for j in range(target_y.shape[1]):
                probs = torch.stack(
                    [distributions[k][i][j].log_prob(target_y[i][j]) for k in range(5)]
                )
                probs += torch.log(mix[i][j])
                loss -= torch.logsumexp(probs, dim=0)
        return loss / target_y.shape[0] / target_y.shape[1]

    def bce_loss_fn(self, distributions, target_y, mu):
        if self.args.regression:
            loss = 0
            for i in range(target_y.shape[0]): #O(batch size*n_pushes)
                for j in range(target_y.shape[1]):
                    loss += torch.dist(mu[i][j], target_y[i][j])
            return loss / target_y.shape[0] / target_y.shape[1]
        else:
            loss = 0
            for i in range(target_y.shape[0]):
                for j in range(target_y.shape[1]):
                    # if torch.rand(1) < 0.0001:
                    #     print(
                    #         torch.exp(distributions[i][j].log_prob(target_y[i][j])),
                    #         distributions[i][j].mean,
                    #         target_y[i][j],
                    #     )
                    loss -= distributions[i][j].log_prob(target_y[i][j]) 

                    # loss = log_p(x|z) = \log \prod p(x_i|z) = \sum \log p(x_i|z) 
            # return loss / target_y.shape[0] / target_y.shape[1]
            return loss / target_y.shape[0] 

    def kl_loss_fn(self, qzs, qz_contexts):
        loss = 0
        for i in range(len(qzs)):
            loss += torch.distributions.kl_divergence(qzs[i], qz_contexts[i]) # O(batch size)
        return loss / len(qzs)

    def average_distance(self, distributions, target_y, mixture=None):
        distance = 0
        for i in range(target_y.shape[0]):
            for j in range(target_y.shape[1]):
                if not self.args.use_mixture:
                    distance += torch.sqrt(
                        F.mse_loss(
                            distributions[i][j].mean[:3],
                            target_y[i][j][:3],
                            reduction="sum",
                        )
                    )
                else:
                    means = torch.stack([distributions[k][i][j].mean for k in range(5)])
                    means = torch.mul(means, mixture[i][j].unsqueeze(1))
                    means = means.sum(dim=0)
                    distance += torch.sqrt(
                        F.mse_loss(
                            means[:3],
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
        # var = 0.1 + 0.9 * self.sigmoid(log_var)

        var = 0.01 + 0.99 * self.sigmoid(log_var)

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
        use_mixture=False,
        point_cloud=False,
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
        self.X_DIMENSION = 3 if args.no_contact else 9 # If we don't do offset then we may need to get rid of this, can't calculate it  
        self.OBJ_PROPERTIES = 5 if args.use_obj_prop else 0  # mass, friction, com

        # print("SELF OBJ PROP", self.OBJ_PROPERTIES)
        self.INPUT_SIZE = (
            (0 if args.no_deterministic else embed_dimension)
            + d_latents
            + (point_net_encoding_size if not args.point_cloud else 0)
            + self.X_DIMENSION
            + self.OBJ_PROPERTIES
        )
        self.OUTPUT_SIZE = output_size
        self.POINT_NET_ENCODING_SIZE = point_net_encoding_size

        # self.layer_sizes = [512, 1024, 512]
        self.layer_sizes = [512, 256, 128]
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, self.layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.layer_sizes[0], self.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.layer_sizes[1], self.layer_sizes[2]),
            nn.ReLU(),
            # nn.Linear(self.layer_sizes[1], self.layer_sizes[2]),
            # nn.ReLU(),
        )
        self.use_mixture = use_mixture

        if use_mixture:
            self.mean = nn.ModuleList(
                nn.Linear(self.layer_sizes[-1], output_size) for _ in range(5)
            )
            self.var = nn.ModuleList(
                nn.Linear(self.layer_sizes[-1], output_size * (output_size + 1) // 2)
                for _ in range(5)
            )
            self.mixture = nn.Linear(self.layer_sizes[-1], 5)
        elif args.guess_obj:
            self.obj = nn.Linear(self.layer_sizes[-1], 5)
        else:
            self.mean = nn.Linear(self.layer_sizes[-1], output_size)
            self.var = nn.Linear(
                self.layer_sizes[-1], output_size * (output_size + 1) // 2
            )

    def forward(
        self, target_x, deterministic, latent_distribution, mesh_vector, obj_data
    ):
        (B, N, D) = target_x.shape
        if self.args.latent_samp == -1:
            if latent_distribution is not None:
                latent = torch.stack(
                    [
                        torch.stack([latent_distribution[i].rsample() for _ in range(N)])
                        for i in range(B)
                    ]
                ).to(target_x.device)
            else: 
                latent = None
            if mesh_vector is not None:
                mesh_vector = mesh_vector.unsqueeze(1)
                mesh_vector = mesh_vector.expand(B, N, self.POINT_NET_ENCODING_SIZE)
        else:
            latent = torch.stack(
                [
                    [
                        [
                            torch.stack(
                                [
                                    latent_distribution[i].rsample()
                                    for _ in range(self.args.latent_samp)
                                ]
                            )
                        ]
                        for __ in range(N)
                    ]
                    for i in range(B)
                ]
            )
            mesh_vector = mesh_vector.unsqueeze(1)
            mesh_vector = mesh_vector.expand(B, N, self.POINT_NET_ENCODING_SIZE)
            mesh_vector = mesh_vector.unsqueeze(2)
            mesh_vector = mesh_vector.expand(
                B, N, self.args.latent_samp, self.POINT_NET_ENCODING_SIZE
            )

            target_x = target_x.unsqueeze(2)
            target_x = target_x.expand(B, N, self.args.latent_samp, target_x.shape[-1])

            deterministic = deterministic.unsqueeze(2)
            deterministic = deterministic.expand(
                B, N, self.args.latent_samp, deterministic.shape[-1]
            )

            # if self.args.use_obj_prop:
            #     obj_data = obj_data.unqueeze

        if self.args.point_cloud: 
            obj_data = obj_data.unsqueeze(1)
            obj_data = obj_data.expand(B, N, obj_data.shape[2]).to(target_x.device)
            input_vector = torch.cat([target_x, obj_data], dim=2)
            # print(input_vector.shape)
        else: 
            if self.args.no_deterministic:
                input_vector = torch.cat([target_x, latent, mesh_vector], dim=2)
            else:
                input_vector = torch.cat(
                    [target_x, deterministic, latent, mesh_vector], dim=2
                )
            if self.args.use_obj_prop:
                obj_data = obj_data.unsqueeze(1)
                obj_data = obj_data.expand(B, N, obj_data.shape[2]).to(target_x.device)
                input_vector = torch.cat([input_vector, obj_data], dim=2)

        input_vector = self.layers(input_vector)

        if self.args.latent_samp != -1:
            mean = self.mean(input_vector)
            var = self.var(input_vector)

            # output = torch.zeros(B, N, self.args.latent_sample self.OUTPUT_SIZE, self.OUTPUT_SIZE, device=target_x.device)

            output = torch.zeros(
                B,
                N,
                self.args.latent_sample,
                self.OUTPUT_SIZE,
                self.OUTPUT_SIZE,
                device=target_x.device,
            )
            lower_triangular = torch.tril_indices(
                self.OUTPUT_SIZE, self.OUTPUT_SIZE, device=target_x.device
            )

            output[:, :, :, lower_triangular[0], lower_triangular[1]] = var
            output[:, :, :, range(self.OUTPUT_SIZE), range(self.OUTPUT_SIZE)] = (
                F.softplus(
                    output[:, :, :, range(self.OUTPUT_SIZE), range(self.OUTPUT_SIZE)]
                )
            )
            return (
                [
                    [
                        [
                            torch.distributions.MultivariateNormal(
                                mean[b][n][i], scale_tril=output[b][n][i]
                            )
                            for i in range(self.args.latent_sample)
                        ]
                        for n in range(N)
                    ]
                    for b in range(B)
                ],
                mean,
                output @ output.transpose(-1, -2),
            )

        if self.args.guess_obj:
            # print (self.obj(input_vector).shape)
            return self.obj(input_vector)

        if self.use_mixture:
            distributions = []
            variances = []
            means = []
            for i in range(5):
                mean = self.mean[i](input_vector)
                var = self.var[i](input_vector)

                output = torch.zeros(
                    B, N, self.OUTPUT_SIZE, self.OUTPUT_SIZE, device=target_x.device
                )
                lower_triangular = torch.tril_indices(
                    self.OUTPUT_SIZE, self.OUTPUT_SIZE, device=target_x.device
                )

                output[:, :, lower_triangular[0], lower_triangular[1]] = var
                output[:, :, range(self.OUTPUT_SIZE), range(self.OUTPUT_SIZE)] = (
                    F.softplus(
                        output[:, :, range(self.OUTPUT_SIZE), range(self.OUTPUT_SIZE)]
                    )
                )

                distributions.append(
                    [
                        [
                            torch.distributions.MultivariateNormal(
                                mean[b][j], scale_tril=output[b][j]
                            )
                            for j in range(N)
                        ]
                        for b in range(B)
                    ],
                )
                means.append(mean)
                variances.append(output @ output.transpose(-1, -2))

            mix = self.mixture(input_vector)
            mix = F.softmax(mix, dim=2)
            return (
                distributions,
                torch.stack(means, dim=0),
                torch.stack(variances, dim=0),
                mix,
            )
        else:
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
                            mean[b][j], scale_tril=output[b][j]
                        )
                        for j in range(N)
                    ]
                    for b in range(B)
                ],
                mean,
                output @ output.transpose(-1, -2),
            )
