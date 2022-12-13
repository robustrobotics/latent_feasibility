import numpy as np
import torch
import torch.nn as nn

from learning.models.pointnet import PointNetEncoder, PointNetRegressor, PointNetClassifier, PointNetPerPointClassifier


class CustomGraspNeuralProcess(nn.Module):

    def __init__(self, d_latents):
        super(CustomGraspNeuralProcess, self).__init__()
        d_mesh = 3
        n_out_geom = 1
        self.encoder = CustomGNPEncoder(d_latents=d_latents, d_mesh=d_mesh)
        self.decoder = CustomGNPDecoder(n_in=3+1+n_out_geom+d_latents+d_mesh, d_latents=d_latents)
        self.mesh_encoder = PointNetRegressor(n_in=3, n_out=d_mesh)
        self.grasp_geom_encoder = PointNetRegressor(n_in=3, n_out=n_out_geom)

        self.d_latents = d_latents

    def forward(self, contexts, target_xs, meshes):
        mesh_enc = self.mesh_encoder(meshes)
        # mesh_enc = torch.zeros_like(mesh_enc)

        context_geoms, context_midpoints, context_forces, context_labels = contexts
        n_batch, n_grasp, _, n_geom_pts = context_geoms.shape
        geoms = context_geoms.view(-1, 3, n_geom_pts)
        geoms_enc = self.grasp_geom_encoder(geoms).view(n_batch, n_grasp, -1)

        mu, sigma = self.encoder(
            geoms_enc,
            context_midpoints, context_forces, context_labels,
            mesh_enc
        )

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)

        # Replace True properties with latent samples.
        target_geoms, target_mids, target_forces = target_xs
        n_batch, n_grasp, _, n_pts = target_geoms.shape
        z = q_z.rsample()

        geoms = target_geoms.view(-1, 3, n_pts)
        geoms_enc = self.grasp_geom_encoder(geoms).view(n_batch, n_grasp, -1)
        y_pred = self.decoder(geoms_enc, target_mids, target_forces, z, mesh_enc)
        return y_pred, q_z

    def conditional_forward(self, target_xs, meshes, zs):
        """ Forward function that specifies the latents (i.e., no encoder is used). """
        mesh_enc = self.mesh_encoder(meshes)
        # mesh_enc = torch.zeros_like(mesh_enc)
        target_geoms, target_mids, target_forces = target_xs
        n_batch, n_grasp, _, n_pts = target_geoms.shape
        geoms = target_geoms.reshape(-1, 3, n_pts)
        geoms_enc = self.grasp_geom_encoder(geoms).view(n_batch, n_grasp, -1)

        y_pred = self.decoder(geoms_enc, target_mids, target_forces, zs, mesh_enc)
        return y_pred

class CustomGNPDecoder(nn.Module):

    def __init__(self, n_in, d_latents):
        super(CustomGNPDecoder, self).__init__()
        self.pointnet = PointNetClassifier(n_in=n_in)
        self.n_in = n_in
        self.d_latents = d_latents

    def forward(self, target_geoms, target_midpoints, target_forces, zs, meshes):
        """
        :param target geoms: (batch_size, n_grasps, 3, n_points)
        :param target_midpoint: (batch_size, n_grasps, 3)
        :param target_forces: (batch_size, n_grasps)
        :param zs: (batch_size, d_latents)
        """
        n_batch, n_grasp, n_feats = target_geoms.shape
        zs_broadcast = zs[:, None, :].expand(n_batch, n_grasp, -1)
        # midpoints_broadcast = target_midpoints[:, :, :, None].expand(n_batch, n_grasp, 3, n_pts)
        meshes_broadcast = meshes[:, None, :].expand(n_batch, n_grasp, -1)
        xs_with_latents = torch.cat([
            target_midpoints,
            target_geoms,
            target_forces[:, :, None],
            zs_broadcast,
            meshes_broadcast
        ], dim=2)

        zs_grasp_broadcast = zs[:, None, :].expand(n_batch, n_grasp, self.d_latents)
        xs = xs_with_latents.view(-1, self.n_in)[:, :, None]
        # import IPython; IPython.embed()
        xs = self.pointnet(xs, zs_grasp_broadcast.reshape(-1, self.d_latents))
        #  import IPython; IPython.embed()
        return xs.view(n_batch, n_grasp, 1)


class CustomGNPEncoder(nn.Module):

    def __init__(self, d_latents, d_mesh):
        super(CustomGNPEncoder, self).__init__()

        # Used to encode local geometry.
        n_out_geom = 1
        self.pn_grasp = PointNetRegressor(n_in=3+1+1+n_out_geom+d_mesh, n_out=d_latents*2)
        self.d_latents = d_latents

    def forward(self, geoms_enc, context_midpoints, context_forces, context_labels, meshes):
        """
        :param context_geoms: (batch_size, n_grasps, 3, n_points)
        :param context_midpoints: (batch_size, n_grasps, 3)
        :param context_labels: (batch_size, n_grasps, 1)
        """
        n_batch, n_grasp = context_labels.shape
        meshes = meshes[:, None, :].expand(n_batch, n_grasp, -1)
        grasp_input = torch.cat([
            context_midpoints,
            context_forces[:, :, None],
            context_labels[:, :, None],
            geoms_enc,
            meshes
        ], dim=2).swapaxes(1, 2)
        x = self.pn_grasp(grasp_input)
        mu, log_sigma = x[..., :self.d_latents], x[..., self.d_latents:]
        # sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)
        sigma = 0.0001 + torch.exp(log_sigma)
        return mu, sigma
