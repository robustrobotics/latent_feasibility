import numpy as np
import torch
import torch.nn as nn

from learning.models.pointnet import PointNetEncoder, PointNetRegressor, PointNetClassifier, PointNetPerPointClassifier


class CustomGraspNeuralProcess(nn.Module):

    def __init__(self, d_latents, use_local_point_clouds=True):
        super(CustomGraspNeuralProcess, self).__init__()
        d_mesh = 3
        n_out_geom = 1

        self.encoder = CustomGNPEncoder(d_latents=d_latents, d_mesh=d_mesh,
                                        use_local_point_clouds=use_local_point_clouds)

        # need to adjust input if we are only using curvature features
        self.use_local_point_clouds = use_local_point_clouds
        if self.use_local_point_clouds:
            self.decoder = CustomGNPDecoder(n_in=6 + 1 + n_out_geom + d_latents + d_mesh,
                                            d_latents=d_latents, use_local_point_clouds=use_local_point_clouds)
        else:
            self.decoder = CustomGNPDecoder(n_in=6 + 12 + 1 + d_latents + d_mesh, d_latents=d_latents,
                                            use_local_point_clouds=use_local_point_clouds)
        self.mesh_encoder = PointNetRegressor(n_in=3, n_out=d_mesh)

        # currently the local geom encoder is still here so that we do not need to modify the logger for saving/loading
        self.grasp_geom_encoder = PointNetRegressor(n_in=3, n_out=n_out_geom)

        self.d_latents = d_latents

    def forward(self, contexts, target_xs, meshes):
        q_z, mesh_enc = self.forward_until_latents(contexts, meshes)

        # Replace True properties with latent samples.
        target_geoms, target_grasp_points, target_curvatures, target_mids, target_forces = target_xs
        z = q_z.rsample()

        n_batch, n_grasp, _, n_pts = target_geoms.shape
        geoms = target_geoms.view(-1, 3, n_pts)
        if self.use_local_point_clouds:
            geoms_enc = self.grasp_geom_encoder(geoms).view(n_batch, n_grasp, -1)
        else:
            geoms_enc = None
        y_pred = self.decoder(geoms_enc,
                              target_grasp_points, target_curvatures, target_mids, target_forces,
                              z, mesh_enc)
        return y_pred, q_z

    def forward_until_latents(self, contexts, meshes):
        mesh_enc = self.mesh_encoder(meshes)

        context_geoms, \
            context_grasp_points, \
            context_curvatures, \
            context_midpoints, \
            context_forces, \
            context_labels = contexts

        n_batch, n_grasp, _, n_geom_pts = context_geoms.shape
        print(n_batch, n_grasp)
        geoms = context_geoms.view(-1, 3, n_geom_pts)
        if self.use_local_point_clouds:
            geoms_enc = self.grasp_geom_encoder(geoms).view(n_batch, n_grasp, -1)
        else:
            geoms_enc = None
        mu, sigma = self.encoder(
            geoms_enc,
            context_grasp_points, context_curvatures, context_midpoints, context_forces, context_labels,
            mesh_enc
        )

        # Sample via reparameterization trick.
        q_z = torch.distributions.normal.Normal(mu, sigma)
        return q_z, mesh_enc

    def conditional_forward(self, target_xs, meshes, zs):
        """ Forward function that specifies the latents (i.e., no encoder is used). """
        mesh_enc = self.mesh_encoder(meshes)
        # mesh_enc = torch.zeros_like(mesh_enc)

        target_geoms, target_grasp_points, target_curvatures, target_mids, target_forces = target_xs
        n_batch, n_grasp, _, n_pts = target_geoms.shape
        geoms = target_geoms.reshape(-1, 3, n_pts)

        if self.use_local_point_clouds:
            geoms_enc = self.grasp_geom_encoder(geoms).view(n_batch, n_grasp, -1)
        else:
            geoms_enc = None

        y_pred = self.decoder(geoms_enc,
                              target_grasp_points, target_curvatures, target_mids, target_forces,
                              zs, mesh_enc)
        return y_pred


class CustomGNPDecoder(nn.Module):

    def __init__(self, n_in, d_latents, use_local_point_clouds=True):
        super(CustomGNPDecoder, self).__init__()
        self.pointnet = PointNetClassifier(n_in=n_in)
        self.n_in = n_in
        self.d_latents = d_latents
        self.use_local_point_clouds = use_local_point_clouds

    def forward(self, target_geoms, target_grasp_points, target_curvatures, target_midpoints, target_forces, zs,
                meshes):
        """
        :param target geoms: (batch_size, n_grasps, 3, n_points)
        :param target_midpoint: (batch_size, n_grasps, 3)
        :param target_forces: (batch_size, n_grasps)
        :param zs: (batch_size, d_latents)
        """
        n_batch, n_grasp = target_forces.shape
        zs_broadcast = zs[:, None, :].expand(n_batch, n_grasp, -1)
        # midpoints_broadcast = target_midpoints[:, :, :, None].expand(n_batch, n_grasp, 3, n_pts)
        meshes_broadcast = meshes[:, None, :].expand(n_batch, n_grasp, -1)

        target_grasp_points_flat = target_grasp_points.flatten(start_dim=2)
        if self.use_local_point_clouds:
            xs_with_latents = torch.cat([
                target_grasp_points_flat,
                target_geoms,
                target_forces[:, :, None],
                zs_broadcast,
                meshes_broadcast
            ], dim=2)
        else:
            target_curvatures_flat = target_curvatures.flatten(start_dim=2)

            xs_with_latents = torch.cat([
                target_grasp_points_flat,
                target_curvatures_flat,
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

    def __init__(self, d_latents, d_mesh, use_local_point_clouds):
        super(CustomGNPEncoder, self).__init__()

        # Used to encode local geometry.
        self.use_local_point_clouds = use_local_point_clouds
        if self.use_local_point_clouds:
            n_out_geom = 1
            self.pn_grasp = PointNetRegressor(n_in=6 + 1 + 1 + n_out_geom + d_mesh, n_out=d_latents * 2)
        else:  # grasp points + curvatures + midpoints + forces + labels
            self.pn_grasp = PointNetRegressor(n_in=6 + 12 + 1 + 1 + d_mesh, n_out=d_latents * 2)
        self.d_latents = d_latents

    def forward(self, geoms_enc, context_grasp_points, context_curvatures, context_midpoints, context_forces,
                context_labels, meshes):
        """
        :param context_geoms: (batch_size, n_grasps, 3, n_points)
        :param context_midpoints: (batch_size, n_grasps, 3)
        :param context_labels: (batch_size, n_grasps, 1)
        """
        n_batch, n_grasp = context_labels.shape

        # expand single object global mesh encodings for all grasps
        meshes = meshes[:, None, :].expand(n_batch, n_grasp, -1)

        context_grasp_points = context_grasp_points.flatten(start_dim=2)
        if self.use_local_point_clouds:
            grasp_input = torch.cat([
                context_grasp_points,
                context_forces[:, :, None],
                context_labels[:, :, None],
                geoms_enc,
                meshes
            ], dim=2).swapaxes(1, 2)
        else:
            # adjust curvature and grasp point format to single vectors
            context_curvatures = context_curvatures.flatten(start_dim=2)

            grasp_input = torch.cat([
                context_grasp_points,
                context_curvatures,
                context_forces[:, :, None],
                context_labels[:, :, None],
                meshes
            ], dim=2).swapaxes(1, 2)
        x = self.pn_grasp(grasp_input)
        mu, log_sigma = x[..., :self.d_latents], x[..., self.d_latents:]
        # sigma = 0.01 + 0.99 * torch.sigmoid(log_sigma)
        sigma = 0.0001 + torch.exp(log_sigma)
        return mu, sigma
