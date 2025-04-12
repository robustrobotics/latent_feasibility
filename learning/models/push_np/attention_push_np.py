import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import pickle


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
        if not args.no_pointnet:
            self.point_net_encoder = PointNetRegressor(
                self.N_MESH_FEATURES,
                self.POINT_NET_ENCODING_SIZE,
                n_geometric_features=self.N_GEOM_FEATURES,
                use_stn=False,
                use_batch_norm=False,
            )
        else:
            self.point_net_encoder = None
            self.POINT_NET_ENCODING_SIZE = 0  # Set to 0 when PointNet is disabled
            
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

        if hasattr(self.args, 'use_regression_model') and self.args.use_regression_model:
            path = os.path.join("learning", "data", "pushing", args.dataset, args.regression_model)
            weights = torch.load(os.path.join(path, "best_model.pth"))
            regression_args = pickle.load(open(os.path.join(path, "args.pkl"), "rb"))
            regression_args.use_regression_model = False
            regression_model = AttentionPushNP(regression_args)
            regression_model.load_state_dict(weights)
            self.regression_model = regression_model
            for param in self.regression_model.parameters():
                param.requires_grad = False  # freeze all the weights

    def forward(
        self, context_x, context_y, target_x, target_y, mesh, obj, mode="train"
    ):
        """
        Forward pass of the AttentionPushNP model.
        
        Args:
            context_x: Input features for context points
            context_y: Output features for context points
            target_x: Input features for target points
            target_y: Output features for target points (only used during training)
            mesh: Mesh features
            obj: Object features
            mode: Mode of operation ('train' or 'eval')
            
        Returns:
            A tuple containing losses, predictions, and other metrics
        """
        if self.point_net_encoder is not None:
            mesh_vector, _ = self.point_net_encoder(mesh.transpose(1, 2))
        else:
            mesh_vector = None
        
        # Get latent distributions for both context and target
        if mode == "train":
            qz = self.latent_encoder(target_x, target_y, mesh_vector)
            qz_context = self.latent_encoder(context_x, context_y, mesh_vector)
        else:
            qz = self.latent_encoder(context_x, context_y, mesh_vector)
            qz_context = None
        
        # Get deterministic encoding if enabled
        if not self.args.no_deterministic:
            deterministic = self.deterministic_encoder(context_x, context_y, target_x)
        else:
            deterministic = None

        # Handle the guess_obj branch
        if self.args.guess_obj:
            # Sample from the latent distribution and expand to match target shape
            guess = qz.sample().unsqueeze(1).expand(-1, target_x.shape[1], -1)
            loss, lx, ly, lz = self.guess_obj_data(guess, obj)
            return (
                loss,
                torch.zeros(1, device=context_x.device),
                torch.zeros(1, device=context_x.device),
                torch.zeros(context_x.shape[0], context_x.shape[1], context_x.shape[2], 4, device=context_x.device),
                torch.zeros(context_x.shape[0], context_x.shape[1], context_x.shape[2], 4, 4, device=context_x.device),
                torch.zeros(1, device=context_x.device),
                (lx, ly, lz),
            )
        
        # Sample from the latent distribution and prepare for decoder
        latent_tensor = qz.sample().unsqueeze(1).expand(-1, target_x.shape[1], -1)
        
        # Use the decoder to get predictions
        if self.args.use_mixture:
            means, scale_trils, covariances, mix = self.decoder(target_x, deterministic, latent_tensor, mesh_vector, obj)
            bce_loss = self.bce_loss_mixture(means, scale_trils, mix, target_y)
            distance = self.average_distance_mixture(means, mix, target_y)
        elif not (hasattr(self.args, 'use_regression_model') and self.args.use_regression_model):
            dist, mu, sigma = self.decoder(target_x, deterministic, latent_tensor, mesh_vector, obj)
            if mode != "test":
                bce_loss = self.bce_loss_fn(dist, target_y, regression=self.args.regression, mu=mu)
                distance = self.average_distance(dist, target_y)
            else: 
                bce_loss = 0
                distance = 0
        else:
            # Regression model branch
            dist, mu, sigma = self.decoder(target_x, deterministic, latent_tensor, mesh_vector, obj)
            mu_regression = self.regression_model(context_x, context_y, target_x, target_y, mesh, obj, mode=mode)[3]
            bce_loss, _ = self.only_train_distrib(mu_regression, sigma, target_y)
            distance = self.average_distance(dist, target_y)
            mu = mu_regression

        # Training mode: compute KL divergence
        if mode == "train":
            kl_loss = torch.distributions.kl_divergence(qz, qz_context).mean()
            total_loss = kl_loss + bce_loss
            entropy = None
        else:
            kl_loss = None
            entropy = qz.entropy().mean()
            total_loss = bce_loss

        return total_loss, bce_loss, kl_loss, mu, sigma, distance, (entropy if mode != "train" else None)

    def guess_obj_data(self, y, obj):
        obj = obj.unsqueeze(1)
        obj = obj.expand(y.shape)
        
        x = F.mse_loss(y[2], obj[2], reduction="mean")
        y1 = F.mse_loss(y[3], obj[3], reduction="mean")
        z = F.mse_loss(y[4], obj[4], reduction="mean")
        return F.mse_loss(y[2:5], obj[2:5], reduction="mean"), x, y1, z

    def bce_loss_fn(self, distribution, target_y, regression=False, mu=None):
        """
        For non-mixture (or regression) cases, compute negative log likelihood.
        If regression mode is enabled, we use an L2 loss.
        """
        if regression:
            # Mean L2 distance over all pushes.
            return torch.norm(mu - target_y, dim=-1).mean()
        else:
            # Check if distribution is a batched MultivariateNormal or list of distributions
            if isinstance(distribution, torch.distributions.Distribution):
                # Batched version
                # distribution is a batched MultivariateNormal.
                log_prob = distribution.log_prob(target_y)  # vectorized over [B, N]
                return -log_prob.mean()
            else:
                # Original non-batched version (list of lists of distributions)
                loss = 0
                for i in range(target_y.shape[0]):
                    for j in range(target_y.shape[1]):
                        loss -= distribution[i][j].log_prob(target_y[i][j])
                return loss / target_y.shape[0]

    def bce_loss_mixture(self, means, scale_trils, mixture_probs, target_y):
        """
        Compute the negative log likelihood for a mixture of Gaussians in a vectorized manner.
        Args:
            means: Tensor of shape [B, N, 5, OUTPUT_SIZE]
            scale_trils: Tensor of shape [B, N, 5, OUTPUT_SIZE, OUTPUT_SIZE]
            mixture_probs: Tensor of shape [B, N, 5]
            target_y: Tensor of shape [B, N, OUTPUT_SIZE]
        """
        B, N, num_comp, _ = means.shape
        log_probs = []
        for i in range(num_comp):
            # Build a batched multivariate distribution for each component.
            dist_i = torch.distributions.MultivariateNormal(means[:, :, i, :],
                                                            scale_tril=scale_trils[:, :, i, :, :])
            log_prob_i = dist_i.log_prob(target_y)  # shape [B, N]
            log_probs.append(log_prob_i)
        log_probs = torch.stack(log_probs, dim=2)  # [B, N, 5]
        log_mix = torch.log(mixture_probs + 1e-10)  # avoid numerical issues
        combined_log_probs = torch.logsumexp(log_probs + log_mix, dim=2)  # [B, N]
        return -combined_log_probs.mean()

    def average_distance(self, distribution, target_y):
        """
        Computes the average distance (Euclidean over the first three dimensions)
        between the predicted means and target_y.
        """
        # Check if distribution is a batched MultivariateNormal or list of distributions
        if isinstance(distribution, torch.distributions.Distribution):
            # Batched version
            pred_mean = distribution.mean  # shape: [B, N, OUTPUT_SIZE]
            diff = pred_mean[..., :3] - target_y[..., :3]
            return torch.sqrt((diff ** 2).sum(dim=-1)).mean()
        else:
            # Original non-batched version
            distance = 0
            for i in range(target_y.shape[0]):
                for j in range(target_y.shape[1]):
                    distance += torch.sqrt(
                        F.mse_loss(
                            distribution[i][j].mean[:3],
                            target_y[i][j][:3],
                            reduction="sum",
                        )
                    )
            return distance / target_y.shape[0] / target_y.shape[1]

    def average_distance_mixture(self, means, mixture_probs, target_y):
        """
        For a mixture of Gaussians, compute the distance using the weighted average of the means.
        """
        weighted_mean = (means * mixture_probs.unsqueeze(-1)).sum(dim=2)  # [B, N, OUTPUT_SIZE]
        diff = weighted_mean[..., :3] - target_y[..., :3]
        return torch.sqrt((diff ** 2).sum(dim=-1)).mean()

    def kl_loss_fn(self, qzs, qz_contexts):
        """
        Compute an average KL divergence between two sets of (batched) distributions.
        """
        if self.args.no_kl:
            return torch.zeros(1).to(qzs[0].mean.device)
        loss = 0
        for i in range(len(qzs)):
            loss += torch.distributions.kl_divergence(qzs[i], qz_contexts[i])
        return loss / len(qzs)

    def only_train_distrib(self, regression_mu, sigma, target_y):
        """
        Combine the regression model's mean with the sigma from this model
        to produce a batched multivariate distribution, then compute the loss.
        """
        # Assume regression_mu has shape [B, N, OUTPUT_SIZE] and sigma is compatible.
        B, N = regression_mu.shape[0], regression_mu.shape[1]
        
        # Build a batched distribution using the regression model's mean and current model's scale_tril
        dist = torch.distributions.MultivariateNormal(regression_mu, scale_tril=sigma)
        loss = self.bce_loss_fn(dist, target_y)
        return loss, dist

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
            self.X_DIMENSION = 3 if args.no_contact else 9
            self.PUSH_DIMENSION = self.X_DIMENSION + 4 + self.POINT_NET_ENCODING_SIZE

        self.EMBED_DIMENSION = embed_dimension
        self.HEADS = h
        self.d_latents = d_latents

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
        
        # Check if there is any context (all zeros)
        if torch.all(context_x == 0) and torch.all(context_y == 0):
            # Return standard normal distribution when there is no context
            mean = torch.zeros(B, self.d_latents, device=context_x.device)
            scale_tril = torch.eye(self.d_latents, device=context_x.device).unsqueeze(0).expand(B, -1, -1)
            # print("NO CONTEXT")
            return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
            
        if mesh_encoding is not None:
            mesh_encoding = mesh_encoding.unsqueeze(1)
            mesh_encoding = mesh_encoding.expand(B, N, self.POINT_NET_ENCODING_SIZE)
            pushes = torch.cat([context_x, context_y, mesh_encoding], dim=2)
        else:
            pushes = torch.cat([context_x, context_y], dim=2)

        # Ensure the input dimension matches PUSH_DIMENSION
        if pushes.shape[2] != self.PUSH_DIMENSION:
            # If we're missing dimensions, pad with zeros
            if pushes.shape[2] < self.PUSH_DIMENSION:
                padding = torch.zeros(B, N, self.PUSH_DIMENSION - pushes.shape[2], device=pushes.device)
                pushes = torch.cat([pushes, padding], dim=2)
            # If we have extra dimensions, truncate
            else:
                pushes = pushes[:, :, :self.PUSH_DIMENSION]

        pushes = self.first(pushes)
        for i in range(2):
            pushes = self.multihead[i](pushes, pushes, pushes)[0]

        pushes = pushes.mean(dim=1)
        pushes = self.penultimate(pushes)
        mean = self.mean(pushes)
        log_var = self.log_var(pushes)
        var = 0.01 + 0.99 * self.sigmoid(log_var)

        # Create batched diagonal covariance matrix
        scale_tril = torch.diag_embed(torch.sqrt(var))
        
        # Create a single batched MultivariateNormal distribution
        dist = torch.distributions.MultivariateNormal(
            loc=mean,
            scale_tril=scale_tril
        )

        return dist

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
        self.args = args
        self.X_DIMENSION = 3 if args.no_contact else 9
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
        # Check if there is any context (all zeros)
        if torch.all(context_x == 0) and torch.all(context_y == 0):
            # print("NO CONTEXT")
            # Return zero tensor of appropriate shape when there is no context
            B, N, _ = target_x.shape
            return torch.zeros(B, N, self.FINAL, device=target_x.device)

        # Ensure input dimensions match expected dimensions
        if self.args.no_contact and context_x.shape[-1] > 3:
            # If no_contact is True but we have more than 3 dimensions, take only the first 3
            context_x = context_x[..., :3]
            target_x = target_x[..., :3]
        elif not self.args.no_contact and context_x.shape[-1] < 9:
            # If no_contact is False but we have less than 9 dimensions, pad with zeros
            padding = torch.zeros(*context_x.shape[:-1], 9 - context_x.shape[-1], device=context_x.device)
            context_x = torch.cat([context_x, padding], dim=-1)
            target_x = torch.cat([target_x, padding], dim=-1)

        context = torch.cat([context_x, context_y], dim=2)
        context = self.to_embed(context)

        for i in range(2):
            context = self.self_attention_heads[i](context, context, context)[0]

        keys = self.to_key(context_x)
        queries = self.to_key(target_x)

        for i in range(2):
            queries = self.cross_attention_heads[i](
                query=queries, key=keys, value=context
            )[0]

        queries = F.relu(queries)
        queries = self.shorten(queries)
        return queries

class APNPDecoder(nn.Module):
    def __init__(self, args, embed_dimension=8, d_latents=5,
                 point_net_encoding_size=512, output_size=4,
                 use_mixture=False, point_cloud=False):
        """
        The decoder for the model which takes the latent variable and 
        the deterministic encoding for the target_x and predicts target_y.
        """
        super().__init__()

        self.args = args
        self.OUTPUT_SIZE = output_size  # output dimension per push
        self.X_DIMENSION = 3 if args.no_contact else 9  
        self.OBJ_PROPERTIES = 5 if args.use_obj_prop else 0

        # Determine input size based on settings.
        self.INPUT_SIZE = (
            (0 if args.no_deterministic else embed_dimension)
            + d_latents
            + (point_net_encoding_size if not args.point_cloud else 0)
            + self.X_DIMENSION
            + self.OBJ_PROPERTIES
        )

        # Example MLP layers (feel free to adjust)
        layer_sizes = [512, 256, 128]
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
        )
        self.use_mixture = use_mixture

        # Single Gaussian branch
        if not use_mixture and not args.guess_obj:
            self.mean = nn.Linear(layer_sizes[-1], output_size)
            # The covariance is parameterized as a flat vector that will be
            # converted into a lower triangular matrix. Its size is:
            self.var = nn.Linear(layer_sizes[-1], output_size * (output_size + 1) // 2)

        # Mixture branch (vectorized over mixture components)
        if use_mixture:
            self.mean = nn.ModuleList([
                nn.Linear(layer_sizes[-1], output_size) for _ in range(5)
            ])
            self.var = nn.ModuleList([
                nn.Linear(layer_sizes[-1], output_size * (output_size + 1) // 2) for _ in range(5)
            ])
            self.mixture = nn.Linear(layer_sizes[-1], 5)

        if args.guess_obj:
            self.obj = nn.Linear(layer_sizes[-1], 5)

    def construct_scale_tril(self, var_flat):
        """
        Converts a flat vector of lower-triangular entries into a full scale_tril matrix,
        applying softplus nonlinearity to the diagonal.
        
        var_flat: tensor of shape [*, M] where M = OUTPUT_SIZE*(OUTPUT_SIZE+1)//2
        Returns: tensor of shape [*, OUTPUT_SIZE, OUTPUT_SIZE]
        """
        # Assume var_flat shape: [B, N, M]
        B, N, _ = var_flat.shape
        output_size = self.OUTPUT_SIZE
        scale_tril = torch.zeros(B, N, output_size, output_size, device=var_flat.device)
        tril_indices = torch.tril_indices(output_size, output_size, device=var_flat.device)
        scale_tril[..., tril_indices[0], tril_indices[1]] = var_flat
        # Apply softplus to the diagonal to ensure positive definiteness.
        diag_indices = torch.arange(output_size, device=var_flat.device)
        scale_tril[..., diag_indices, diag_indices] = F.softplus(scale_tril[..., diag_indices, diag_indices])
        return scale_tril

    def forward(self, target_x, deterministic, latent, mesh_vector, obj_data):
        """
        Decodes the input (including target_x, deterministic and latent) and returns
        either:
          - a batched MultivariateNormal distribution, its mean and covariance (non-mixture), or 
          - if using a mixture, stacked means, scale_trils and mixture probabilities.
        """
        B, N, _ = target_x.shape

        # Ensure mesh_vector and latent have shape [B, N, feature_dim]
        if mesh_vector is not None and len(mesh_vector.shape) == 2:  # [B, feature_dim]
            mesh_vector = mesh_vector.unsqueeze(1).expand(-1, N, -1)
            
        # Handle case where latent is None and d_latents is 0
        if latent is None:
            # Create a zero tensor of appropriate shape
            latent = torch.zeros(B, N, 0, device=target_x.device)
        elif len(latent.shape) == 2:  # [B, feature_dim]
            latent = latent.unsqueeze(1).expand(-1, N, -1)

        # Prepare input vector.
        if self.args.point_cloud:
            # Use point cloud information with object data.
            obj_data = obj_data.unsqueeze(1).expand(B, N, obj_data.shape[-1])
            input_vector = torch.cat([target_x, obj_data], dim=2)
        else:
            # print(target_x.shape, deterministic.shape, latent.shape, mesh_vector.shape)
            if self.args.no_deterministic:
                if mesh_vector is not None:
                    input_vector = torch.cat([target_x, latent, mesh_vector], dim=2)
                else:
                    input_vector = torch.cat([target_x, latent], dim=2)
            else:
                if mesh_vector is not None:
                    input_vector = torch.cat([target_x, deterministic, latent, mesh_vector], dim=2)
                else:
                    input_vector = torch.cat([target_x, deterministic, latent], dim=2)
            if self.args.use_obj_prop:
                obj_data = obj_data.unsqueeze(1).expand(B, N, obj_data.shape[-1])
                input_vector = torch.cat([input_vector, obj_data], dim=2)

        # Ensure input dimensions match expected dimensions
        if input_vector.shape[-1] != self.INPUT_SIZE:
            if input_vector.shape[-1] < self.INPUT_SIZE:
                # Pad with zeros if input is smaller than expected
                padding = torch.zeros(*input_vector.shape[:-1], self.INPUT_SIZE - input_vector.shape[-1], device=input_vector.device)
                input_vector = torch.cat([input_vector, padding], dim=-1)
            else:
                # Truncate if input is larger than expected
                input_vector = input_vector[..., :self.INPUT_SIZE]

        # Pass through shared layers.
        h = self.layers(input_vector)

        # Latent sampling branch (if using extra latent samples).
        if self.args.latent_samp != -1:
            # In this case h is assumed to have shape [B, N, latent_samp, D]
            mean = self.mean(h)  # shape [B, N, latent_samp, OUTPUT_SIZE]
            var_flat = self.var(h)  # shape [B, N, latent_samp, (OUTPUT_SIZE*(OUTPUT_SIZE+1)//2)]
            scale_tril = self.construct_scale_tril(var_flat)  # shape [B, N, latent_samp, OUTPUT_SIZE, OUTPUT_SIZE]
            cov = scale_tril @ scale_tril.transpose(-1, -2)
            dist = torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
            return dist, mean, cov

        # Mixture branch.
        elif self.use_mixture:
            # Compute mixture weights.
            mixture_logits = self.mixture(h)  # shape [B, N, 5]
            mixture_probs = F.softmax(mixture_logits, dim=-1)  # shape [B, N, 5]
            
            means = []
            scale_trils = []
            for i in range(5):
                mean_i = self.mean[i](h)  # shape [B, N, OUTPUT_SIZE]
                var_flat_i = self.var[i](h)  # shape [B, N, (OUTPUT_SIZE*(OUTPUT_SIZE+1)//2)]
                scale_tril_i = self.construct_scale_tril(var_flat_i)  # shape [B, N, OUTPUT_SIZE, OUTPUT_SIZE]
                means.append(mean_i)
                scale_trils.append(scale_tril_i)
            # Stack mixture components along a new dimension.
            means = torch.stack(means, dim=2)           # [B, N, 5, OUTPUT_SIZE]
            scale_trils = torch.stack(scale_trils, dim=2)   # [B, N, 5, OUTPUT_SIZE, OUTPUT_SIZE]
            covariances = scale_trils @ scale_trils.transpose(-1, -2)
            return means, scale_trils, covariances, mixture_probs

        # Non-mixture, non-guess object branch.
        elif not self.args.guess_obj:
            mean = self.mean(h)  # shape [B, N, OUTPUT_SIZE]
            var_flat = self.var(h)  # shape [B, N, (OUTPUT_SIZE*(OUTPUT_SIZE+1)//2)]
            scale_tril = self.construct_scale_tril(var_flat)  # shape [B, N, OUTPUT_SIZE, OUTPUT_SIZE]
            cov = scale_tril @ scale_tril.transpose(-1, -2)
            dist = torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
            return dist, mean, cov

        # Guess object branch.
        else:
            return self.obj(h)
