import numpy as np
import torch
import torch.nn as nn

from learning.models.pointnet import PointNetRegressor, PointNetClassifier


def _get_identity_transform(batch_size):
    batched_identity = torch.eye(3).unsqueeze(0).repeat([batch_size, 1, 1])
    if torch.cuda.is_available():
        batched_identity = batched_identity.cuda()

    return batched_identity


class CustomGraspNeuralProcess(nn.Module):

    def __init__(self, d_latents, input_features, d_object_mesh_enc, d_grasp_mesh_enc):
        """
        :param d_latents: # of latent variables for the encoder to output.
        :param input_features: Dictionary describing which features to pass as input to the networks.
        :param d_object_mesh_enc: Dimensionality of object mesh encoding.
        "param d_grasp_mesh_enc: Dimensionality of grasp mesh encoding.
        """
        super(CustomGraspNeuralProcess, self).__init__()
        
        # Compute number of features being used.
        n_mesh_features = 3
        n_in_decoder = 6 + 1 + d_object_mesh_enc  # grasp_points, force, object_mesh
        n_in_encoder = 6 + 1 + 1 + 1 + d_object_mesh_enc  # grasp_point, force, label, c_size, object_mesh
        if input_features['mesh_normals']:
            n_mesh_features += 3
        if input_features['grasp_normals']:
            n_in_decoder += 6
            n_in_encoder += 6
        if input_features['mesh_curvatures']:
            n_mesh_features += 6
        if input_features['grasp_curvatures']:
            n_in_decoder += 12
            n_in_encoder += 12
        if input_features['grasp_mesh']:
            n_in_encoder += d_grasp_mesh_enc
            n_in_decoder += d_grasp_mesh_enc
        if input_features['object_properties']:
            n_in_decoder += 5
        else:
            n_in_decoder += d_latents

        self.encoder = CustomGNPEncoder(
            n_in=n_in_encoder,
            d_latents=d_latents,
            input_features=input_features
        )

        self.decoder = CustomGNPDecoder(
            n_in=n_in_decoder,
            d_latents=d_latents,
            input_features=input_features
        )

        n_geometric_feats = 1
        if input_features['mesh_normals']:
            n_geometric_feats += 1

        self.mesh_encoder = PointNetRegressor(
            n_in=n_mesh_features,
            n_geometric_features=n_geometric_feats,
            n_out=d_object_mesh_enc,
            use_batch_norm=False,
            use_stn=True
        )
        self.grasp_geom_encoder = PointNetRegressor(
            n_in=n_mesh_features,
            n_geometric_features=n_geometric_feats,
            n_out=d_grasp_mesh_enc,
            use_batch_norm=False,
            use_stn=False
        )

        self.d_latents = d_latents
        self.input_features = input_features
        self.d_object_mesh_enc = d_object_mesh_enc
        self.d_grasp_mesh_enc = d_grasp_mesh_enc

    def forward(self, contexts, target_xs, object_data):
        meshes, object_properties = object_data

        q_z, mesh_enc, global_transform = self.forward_until_latents(contexts, meshes)
        # Replace True properties with latent samples.
        target_geoms, target_grasp_points, target_curvatures, target_normals, \
            target_mids, target_forces = target_xs
        if self.input_features['object_properties']:
            z = object_properties
        else:
            z = q_z.rsample()
        
        n_batch, n_grasp, n_feat, n_pts = target_geoms.shape
        geoms = target_geoms.view(-1, n_feat, n_pts)
        if self.input_features['grasp_mesh']:
            geoms_enc = self.grasp_geom_encoder(
                geoms, override_transform=_get_identity_transform(n_batch*n_grasp)
            )[0].view(n_batch, n_grasp, -1)
        else:
            geoms_enc = None

        y_pred = self.decoder(
            geoms_enc,
            target_grasp_points, target_curvatures, target_normals, target_mids, target_forces,
            z, mesh_enc,
            override_transform=global_transform
        )
        return y_pred, q_z

    def forward_until_latents(self, contexts, meshes):
        mesh_enc, global_transform = self.mesh_encoder(meshes)

        context_geoms, context_grasp_points, \
            context_curvatures, context_normals, \
            context_midpoints, context_forces, \
            context_labels, context_sizes = contexts
        # print(context_geoms.shape, context_grasp_points.shape, context_curvatures.shape, context_normals.shape, context_forces.shape)

        n_batch, n_grasp, n_feat, n_geom_pts = context_geoms.shape
        geoms = context_geoms.view(-1, n_feat, n_geom_pts)
        if self.input_features['grasp_mesh']:
            geoms_enc = self.grasp_geom_encoder(
                geoms, override_transform=_get_identity_transform(n_batch*n_grasp)
            )[0].view(n_batch, n_grasp, -1)
        else:
            geoms_enc = None

        mu, sigma = self.encoder(
            geoms_enc,
            context_grasp_points, context_curvatures, context_normals, context_midpoints, context_forces, context_labels, context_sizes,
            mesh_enc,
            override_transform=global_transform
        )

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)
        return q_z, mesh_enc, global_transform

    def conditional_forward(self, target_xs, meshes, zs):
        """ Forward function that specifies the latents (i.e., no encoder is used). """
        mesh_enc, global_transform = self.mesh_encoder(meshes)
        # mesh_enc = torch.zeros_like(mesh_enc)

        target_geoms, target_grasp_points, target_curvatures, target_normals, target_mids, target_forces = target_xs
        n_batch, n_grasp, n_feat, n_pts = target_geoms.shape
        geoms = target_geoms.view(-1, n_feat, n_pts)

        if self.input_features['grasp_mesh']:
            geoms_enc = self.grasp_geom_encoder(
                geoms, override_transform=_get_identity_transform(n_batch*n_grasp)
            )[0].view(n_batch, n_grasp, -1)
        else:
            geoms_enc = None

        y_pred = self.decoder(
            geoms_enc,
            target_grasp_points, target_curvatures, target_normals, target_mids, target_forces,
            zs, mesh_enc,
            override_transform=global_transform
        )
        return y_pred


class CustomGNPDecoder(nn.Module):

    def __init__(self, n_in, d_latents, input_features):
        super(CustomGNPDecoder, self).__init__()

        n_geometric_feats = 2
        if input_features['grasp_normals']:
            n_geometric_feats += 2
        self.pointnet = PointNetClassifier(
            n_in=n_in,
            n_geometric_features=n_geometric_feats,
            use_stn=False
        )

        self.n_in = n_in
        self.d_latents = d_latents
        self.input_features = input_features

    def forward(self, target_geoms, target_grasp_points, target_curvatures, target_normals, target_midpoints, target_forces, zs,
                meshes, override_transform=None):
        """
        :param target geoms: (batch_size, n_grasps, 3, n_points)
        :param target_midpoint: (batch_size, n_grasps, 3)
        :param target_forces: (batch_size, n_grasps)
        :param zs: (batch_size, d_latents)
        """
        n_batch, n_grasp = target_forces.shape
        zs_broadcast = zs[:, None, :].expand(n_batch, n_grasp, -1)
        meshes_broadcast = meshes[:, None, :].expand(n_batch, n_grasp, -1)

        target_grasp_points_flat = target_grasp_points.flatten(start_dim=2)
        xs_with_latents = [target_grasp_points_flat]
        if self.input_features['grasp_normals']:
            target_normals_flat = target_normals.flatten(start_dim=2)
            xs_with_latents += [target_normals_flat]
        if self.input_features['grasp_curvatures']:
            target_curvatures_flat = target_curvatures.flatten(start_dim=2)
            xs_with_latents += [target_curvatures_flat]
        if self.input_features['grasp_mesh']:
            xs_with_latents += [target_geoms]
        xs_with_latents += [
            target_forces[:, :, None],
            zs_broadcast,
            meshes_broadcast
        ]
        xs_with_latents = torch.cat(xs_with_latents, dim=2)
        xs = xs_with_latents.view(-1, self.n_in)[:, :, None]

        if override_transform is not None:
            override_transform = override_transform[:, None, :, :].expand(n_batch, n_grasp, 3, 3).reshape(-1, 3, 3)

        xs, _ = self.pointnet(xs, None, override_transform=override_transform)
        return xs.view(n_batch, n_grasp, 1)


class CustomGNPEncoder(nn.Module):

    def __init__(self, n_in, d_latents, input_features):
        super(CustomGNPEncoder, self).__init__()
        # Used to encode local geometry.
        n_geometric_feats = 2
        if input_features['grasp_normals']:
            n_geometric_feats += 2
        self.pn_grasp = PointNetRegressor(
            n_in=n_in,
            n_out=d_latents * 2,
            use_batch_norm=False,
            n_geometric_features=n_geometric_feats,
            use_stn=False
        )
        self.d_latents = d_latents
        self.input_features = input_features

    def forward(self, geoms_enc, context_grasp_points, context_curvatures, context_normals, context_midpoints, context_forces,
                context_labels, context_sizes, meshes, override_transform=None):
        """
        :param context_geoms: (batch_size, n_grasps, 3, n_points)
        :param context_midpoints: (batch_size, n_grasps, 3)
        :param context_labels: (batch_size, n_grasps, 1)
        """
        n_batch, n_grasp = context_labels.shape

        # Build input feature vector.
        meshes = meshes[:, None, :].expand(n_batch, n_grasp, -1)  # expand single object global mesh encodings for all grasps
        context_grasp_points_flat = context_grasp_points.flatten(start_dim=2)

        grasp_input = [context_grasp_points_flat]
        if self.input_features['grasp_normals']:
            context_normals_flat = context_normals.flatten(start_dim=2)
            grasp_input += [context_normals_flat]
        if self.input_features['grasp_curvatures']:
            context_curvatures_flat = context_curvatures.flatten(start_dim=2)
            grasp_input += [context_curvatures_flat]
        if self.input_features['grasp_mesh']:
            grasp_input += [geoms_enc]
        grasp_input += [
            context_forces[:, :, None],
            context_labels[:, :, None],
            context_sizes[:, :, None],
            meshes]
        grasp_input = torch.cat(grasp_input, dim=2).swapaxes(1, 2)

        # Get latent distribution.
        x, _ = self.pn_grasp(grasp_input, override_transform=override_transform)
        mu, log_sigma = x[..., :self.d_latents], x[..., self.d_latents:]
        # sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)
        sigma = 0.0001 + torch.exp(log_sigma)
        return mu, sigma
