"""
Copyright 2020 Massachusetts Insititute of Technology

Izzy Brand
"""
import argparse
from learning.domains.grasping.grasp_data import GraspDataset, GraspParallelDataLoader
from learning.domains.towers.tower_data import ParallelDataLoader, TowerDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader

from actions import make_platform_world, plan_action
from agents.panda_agent import PandaAgent
from agents.teleport_agent import TeleportAgent
from base_class import BeliefBase
from block_utils import get_adversarial_blocks, get_com_ranges, \
    Environment, ParticleDistribution
from learning.domains.grasping.active_utils import get_fit_object
from learning.models.grasp_np.create_gnp_data import process_geometry
from learning.models.grasp_np.dataset import CustomGNPGraspDataset, custom_collate_fn
from filter_utils import create_uniform_particles, create_gaussian_particles, sample_and_wiggle, \
    sample_particle_distribution
from learning.models.grasp_np.train_grasp_np import check_to_cuda


class ParticleBelief(BeliefBase):
    def __init__(self, block, noise, N=200, plot=False, vis_sim=False):
        self.block = deepcopy(block)
        self.plot = plot  # plot the particles
        self.vis_sim = vis_sim  # display the pybullet simulator

        self.TRUE_OBS_COV = noise * np.eye(3)  # covariance used when add noise to observations
        self.OBS_MODEL_COV = noise * np.eye(3)  # covariance used in observation model
        self.N = N  # number of particles
        self.D = 3  # dimensions of a single particle

        self.setup()
        if self.plot:
            plt.ion()
            fig = plt.figure()
            fig.set_size_inches((4, 4))
            self.ax = Axes3D(fig)
            self.setup_ax(self.ax, self.block)
            self.plot_particles(self.ax, self.particles.particles, self.particles.weights)

    def setup(self):
        self.com_ranges = get_com_ranges(self.block)
        self.particles = create_uniform_particles(self.N, self.D, self.com_ranges)
        self.experience = []
        self.estimated_coms = []

    def setup_ax(self, ax, obj):
        ax.clear()
        halfdim = max(obj.dimensions) / 2
        ax.set_xlim(-halfdim, halfdim)
        ax.set_ylim(-halfdim, halfdim)
        ax.set_zlim(-halfdim, halfdim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(obj.name + ' Center of Mass')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    def plot_particles(self, ax, particles, weights, t=None, true_com=None):
        for particle, weight in zip(particles, weights):
            alpha = 0.25 + 0.75 * weight
            ax.scatter(*particle, s=10, color=(0, 0, 1), alpha=alpha)

        ax.scatter(*self.block.com, s=10, color=(1, 0, 0), label='True CoM')
        ax.legend()

        plt.draw()
        plt.pause(0.1)

    def update(self, observation):
        # observation is a tuple (action, rot, timesteps, pose)

        # resample the distribution
        self.particles = sample_and_wiggle(self.particles, self.experience[-5:],
                                           self.OBS_MODEL_COV, self.block, self.com_ranges)

        self.experience.append(observation)
        action, T, end_pose = observation

        particle_blocks = [deepcopy(self.block) for particle in self.particles.particles]
        for (com, particle_block) in zip(self.particles.particles, particle_blocks):
            particle_block.com = com
        particle_worlds = [make_platform_world(pb, action) for pb in particle_blocks]
        env = Environment(particle_worlds, vis_sim=self.vis_sim)
        for _ in range(T):
            env.step(action=action)

        # update all particle weights
        new_weights = []

        for pi, (particle_world, old_weight) in enumerate(zip(particle_worlds, self.particles.weights)):
            particle_end_pose = particle_world.get_pose(particle_world.objects[1])
            obs_model = multivariate_normal.pdf(end_pose.pos,
                                                mean=particle_end_pose.pos,
                                                cov=self.OBS_MODEL_COV)
            new_weight = old_weight * obs_model
            new_weights.append(new_weight)

        # normalize particle weights
        new_weights = np.array(new_weights) / np.sum(new_weights)
        # and update the particle distribution with the new weights
        self.particles = ParticleDistribution(self.particles.particles, new_weights)

        if self.plot:
            # visualize particles (it's very slow)
            self.setup_ax(self.ax, self.block)
            self.plot_particles(self.ax, self.particles.particles, new_weights)

        mean = np.array(self.particles.particles).T @ np.array(self.particles.weights)

        self.estimated_coms.append(mean)

        env.disconnect()
        env.cleanup()

        return self.particles, self.estimated_coms


class DiscreteLikelihoodParticleBelief(BeliefBase):
    """
    This particle belief represents a belief. It is assumed that the 
    observations will come from a Bernoulli distribution.

    This is meant to be compatible with a LatentEnsemble observation
    model.

    The prior distribution for this belief is N(0, 1). 
    """

    def __init__(self, block, D, N=200, likelihood=None, plot=False):
        self.block = deepcopy(block)
        self.block_id = block.get_id()
        self.plot = plot  # plot the particles

        self.N = N  # number of particles
        self.D = D  # dimensions of a single particle
        self.likelihood = likelihood  # LatentEnsemble object that outputs [0, 1]

        self.setup()
        if self.plot:
            plt.ion()
            fig = plt.figure()
            fig.set_size_inches((4, 4))
            self.ax = Axes3D(fig)
            self.setup_ax(self.ax)
            self.plot_particles(self.ax, self.particles.particles, self.particles.weights)

    def setup(self):
        self.particles = create_gaussian_particles(N=self.N,
                                                   D=self.D,
                                                   means=[0.] * self.D,
                                                   stds=[2.] * self.D)
        self.experience = []
        self.estimated_coms = []

    def setup_ax(self, ax):
        ax.clear()
        halfdim = 5
        ax.set_xlim(-halfdim, halfdim)
        ax.set_ylim(-halfdim, halfdim)
        ax.set_zlim(-halfdim, halfdim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Latent ParticleBelief')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    def plot_particles(self, ax, particles, weights, t=None, true_com=None):
        for particle, weight in zip(particles, weights):
            alpha = 0.25 + 0.75 * weight
            ax.scatter(*particle[1:], s=10, color=(0, 0, 1), alpha=alpha)
        plt.draw()
        plt.pause(0.1)

    def sample_and_wiggle(self, distribution, experience):
        N, D = distribution.particles.shape
        # NOTE(izzy): note sure if this is an ok way to get the covariance matrix...
        # If the weights has collapsed onto a single particle, then the covariance
        # will collapse too and we won't perturb the particles very much after we
        # sample them. Maybe this should be a uniform covariance with the magnitude
        # being equal to the largest variance?
        # NOTE: (mike): I added a small noise term and a M-H update step which hopefully
        # prevents complete collapse. The M-H update is useful so that we don't sample
        # something completely unlikely by chance. It's okay for the noise term to be larger 
        # as the M-H step should reject bad particles - it may be inefficient if too large (and not accept often).
        cov = np.cov(distribution.particles, rowvar=False, aweights=distribution.weights + 1e-3) + np.eye(D) * 0.5
        particles = sample_particle_distribution(distribution, num_samples=N)

        mean = np.mean(particles, axis=0)
        proposed_particles = np.random.multivariate_normal(mean=mean, cov=cov, size=N)

        # TODO: Update M-H update to be compatible with our model.
        # The commented out code block does M-H update. 
        if True:
            # Old particles and new particles.
            likelihoods = np.zeros((N, 2))
            # Compute likelihood of particles over history so far.
            n_correct = np.zeros((N, 2))
            for observation in experience:
                bern_probs_particles = self.get_particle_likelihoods(particles, observation)
                bern_probs_proposed = self.get_particle_likelihoods(proposed_particles, observation)

                # sim_poses = simulate(np.concatenate([particles, proposed_particles], axis=0),
                #                     action,
                #                     T,
                #                     true_block)
                for k in observation.keys():
                    if observation[k]['towers'].shape[0] != 0:
                        label = observation[k]['labels'][0]
                for ix in range(N):
                    # print(bern_probs_particles[ix], bern_probs_particles[ix] > 0.5, label, bern_probs_particles[ix] > 0.5 == label)
                    if (float(bern_probs_particles[ix] > 0.5) == label):
                        n_correct[ix, 0] += 1
                    if (float(bern_probs_proposed[ix] > 0.5) == label):
                        n_correct[ix, 1] += 1
                    likelihood_part = label * bern_probs_particles[ix] + (1 - label) * (1 - bern_probs_particles[ix])
                    likelihood_prop = label * bern_probs_proposed[ix] + (1 - label) * (1 - bern_probs_proposed[ix])
                    likelihoods[ix, 0] += np.log(likelihood_part + 1e-8)
                    likelihoods[ix, 1] += np.log(likelihood_prop + 1e-8)
                    # likelihoods[ix,0] += np.log(multivariate_normal.pdf(true_pose.pos,
                    #                                                     mean=sim_poses[ix, :],
                    #                                                     cov=obs_model_cov)+1e-8)
                    # likelihoods[ix,1] += np.log(multivariate_normal.pdf(true_pose.pos,
                    #                                                     mean=sim_poses[N+ix,:],
                    #                                                     cov=obs_model_cov)+1e-8)
            # print(np.round(np.exp(likelihoods[0:10, :]), 2))
            print('Correct of ALL Samples:')
            # print(len(experience))
            # print(n_correct/len(experience))
            print((n_correct / len(experience)).mean())
            # if len(experience) > 0:
            #     print((bern_probs_particles > 0.5).any())
            # Calculate M-H acceptance prob.
            prop_probs = np.zeros((N, 2))
            for ix in range(N):
                prop_probs[ix, 0] = np.log(multivariate_normal.pdf(particles[ix, :], mean=mean, cov=cov) + 1e-8)
                prop_probs[ix, 1] = np.log(
                    multivariate_normal.pdf(proposed_particles[ix, :], mean=mean, cov=cov) + 1e-8)

            p_accept = likelihoods[:, 1] + prop_probs[:, 0] - (likelihoods[:, 0] + prop_probs[:, 1])
            # p_accept = likelihoods[:,0]+prop_probs[:,1] - (likelihoods[:,1]+prop_probs[:,0])
            accept = np.zeros((N, 2))
            accept[:, 0] = p_accept
            accept = np.min(accept, axis=1)
            # Keep particles based on acceptance probability.
            u = np.random.uniform(size=N)
            indices = np.argwhere(u > 1 - np.exp(accept)).flatten()
            print('Accept Rate:', len(indices) / N)
            particles[indices] = proposed_particles[indices]
        else:
            particles = proposed_particles

        weights = np.ones(N) / float(N)  # weights become uniform again
        return ParticleDistribution(particles, weights)

    def get_particle_likelihoods(self, particles, observation):
        """
         
        """
        dataset = TowerDataset(tower_dict=observation,
                               augment=False)
        dataloader = ParallelDataLoader(dataset=dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        n_dataloaders=1)
        bernoulli_probs = []  # Contains one prediction for each particle.

        latent_samples = torch.Tensor(particles)  # (N, 4)
        if torch.cuda.is_available():
            latent_samples = latent_samples.cuda()
        for set_of_batches in dataloader:
            towers, block_ids, _ = set_of_batches[0]
            if torch.cuda.is_available():
                towers = towers.cuda()
                block_ids = block_ids.cuda()

            for ix in range(0, latent_samples.shape[0] // 10):
                pred = self.likelihood.forward(towers=towers[:, :, 4:],
                                               block_ids=block_ids.long(),
                                               N_samples=10,
                                               collapse_latents=True,
                                               collapse_ensemble=True,
                                               keep_latent_ix=self.block_id,
                                               latent_samples=latent_samples[ix * 10:(ix + 1) * 10, :]).squeeze()
                bernoulli_probs.append(pred.cpu().detach().numpy())
        return np.concatenate(bernoulli_probs)

    def update(self, observation):
        """
        :param observation: tower_dict format for a single tower.
        """
        # Resample the distribution
        self.particles = self.sample_and_wiggle(self.particles, self.experience)

        # Append the current observation to the dataset of all observations so far.
        self.experience.append(observation)

        # Forward simulation using the LatentEnsemble likelihood.
        bernoulli_probs = self.get_particle_likelihoods(self.particles.particles, observation)

        for k in observation.keys():
            if observation[k]['towers'].shape[0] != 0:
                label = observation[k]['labels'][0]
        n_correct = ((bernoulli_probs > 0.5).astype('float32') == label).sum()
        print('Correct for CURRENT sample:', n_correct / len(bernoulli_probs), len(bernoulli_probs))
        # TODO: Replace below using the likelihood defined by the NN.
        # update all particle weights
        new_weights = []

        for pi, (bern_prob, old_weight) in enumerate(zip(bernoulli_probs, self.particles.weights)):
            obs_model = bern_prob * label + (1 - bern_prob) * (1 - label)
            new_weight = old_weight * obs_model
            new_weights.append(new_weight)

        # normalize particle weights
        new_weights = np.array(new_weights) / np.sum(new_weights)
        # and update the particle distribution with the new weights
        self.particles = ParticleDistribution(self.particles.particles, new_weights)

        if self.plot:
            # visualize particles (it's very slow)
            self.setup_ax(self.ax)
            self.plot_particles(self.ax, self.particles.particles, new_weights)

        mean = np.array(self.particles.particles).T @ np.array(self.particles.weights)
        print(mean)
        self.estimated_coms.append(mean)

        return self.particles, self.estimated_coms


class GraspingDiscreteLikelihoodParticleBelief(BeliefBase):
    """
    This particle belief represents a belief. It is assumed that the 
    observations will come from a Bernoulli distribution.

    This is meant to be compatible with a LatentEnsemble observation
    model.

    The prior distribution for this belief is N(0, 1). 
    """

    def __init__(self, object_set, d_latents, n_particles, likelihood=None, resample=False, plot=False):
        """
        Maintain a particle distribution over the latent properties for an object.

        :param object_set: A dictionary of form: {'object_names': [...], 'object_properties': [...]}
            where the last element in each list correponds to the fitting object.
        :param d_latents: Dimensionality of belief space.
        :param n_particles: Number of particles to represent belief.
        :param likelihood: Model that predicts Bernoulli likelihood of data.
        :param resample: If True, sample particles after each interaction.
        :param plot: If True, plot first 3 dimensions of latent space during interactions.
        """
        super().__init__()
        object_name, object_properties, object_ix = get_fit_object(object_set)
        print(f'Fitting: {object_name} {object_properties}')
        self.object_name = object_name
        self.object_properties = object_properties
        self.object_ix = object_ix
        self.resample = resample
        self.plot = plot  # plot the particles

        self.N = n_particles  # number of particles
        self.D = d_latents  # dimensions of a single particle
        self.likelihood = likelihood  # LatentEnsemble object that outputs [0, 1]

        self.setup()
        if self.plot:
            plt.ion()
            fig = plt.figure()
            fig.set_size_inches((4, 4))
            self.ax = Axes3D(fig)
            self.setup_ax(self.ax)
            self.plot_particles(self.ax, self.particles.particles, self.particles.weights)

    def setup(self):
        self.particles = create_gaussian_particles(N=self.N,
                                                   D=self.D,
                                                   means=[0.] * self.D,
                                                   stds=[2.] * self.D)
        if not self.resample:
            self.particles = ParticleDistribution(self.particles.particles, np.ones(self.N))

        self.experience = []
        self.estimated_coms = []

    def setup_ax(self, ax):
        ax.clear()
        halfdim = 5
        ax.set_xlim(-halfdim, halfdim)
        ax.set_ylim(-halfdim, halfdim)
        ax.set_zlim(-halfdim, halfdim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Latent ParticleBelief')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    def plot_particles(self, ax, particles, weights, t=None, true_com=None):
        for particle, weight in zip(particles, weights):
            alpha = 0.25 + 0.75 * weight / np.sum(weights)
            ax.scatter(*particle[:3], s=10, color=(0, 0, 1), alpha=alpha)
        plt.draw()
        plt.pause(0.1)

    def sample_and_wiggle(self, distribution, experience):
        N, D = distribution.particles.shape
        # NOTE(izzy): note sure if this is an ok way to get the covariance matrix...
        # If the weights has collapsed onto a single particle, then the covariance
        # will collapse too and we won't perturb the particles very much after we
        # sample them. Maybe this should be a uniform covariance with the magnitude
        # being equal to the largest variance?
        # NOTE: (mike): I added a small noise term and a M-H update step which hopefully
        # prevents complete collapse. The M-H update is useful so that we don't sample
        # something completely unlikely by chance. It's okay for the noise term to be larger 
        # as the M-H step should reject bad particles - it may be inefficient if too large (and not accept often).

        # PROPOSAL 1: Fit mean and cov to current particles and sample from those.
        # particles = distribution.particles
        cov = np.cov(distribution.particles, rowvar=False, aweights=distribution.weights + 1e-3) + np.eye(D) * 0.5
        particles = sample_particle_distribution(distribution, num_samples=N)
        mean = np.mean(particles, axis=0)
        # # print('Proposal:')
        # # print(mean)
        # # print(np.diag(cov))

        # particles = distribution.particles
        # mean = np.zeros((3,))
        # cov = np.eye(3)
        proposed_particles = np.random.multivariate_normal(mean=mean, cov=cov, size=N)

        # PROPOSAL 2: Resample then wiggle.
        # particles = distribution.particles
        # resampled_particles = sample_particle_distribution(distribution, num_samples=N)     
        # proposed_particles = resampled_particles + np.random.multivariate_normal(mean=np.zeros((D,)), cov=np.eye(D)*0.5, size=N)
        # ----------

        # proposed_particles = particles
        # particles = distribution.particles
        # TODO: Update M-H update to be compatible with our model.
        # The commented out code block does M-H update. 
        if True:
            # Old particles and new particles.
            likelihoods = np.zeros((N, 2))
            # Compute likelihood of particles over history so far.
            n_correct = np.zeros((N, 2))
            for observation in experience:
                bern_probs_particles = self.get_particle_likelihoods(particles, observation)
                bern_probs_proposed = self.get_particle_likelihoods(proposed_particles, observation)

                label = observation['grasp_data']['labels'][0]
                for ix in range(N):
                    # print(bern_probs_particles[ix], bern_probs_particles[ix] > 0.5, label, bern_probs_particles[ix] > 0.5 == label)
                    if (float(bern_probs_particles[ix] > 0.5) == label):
                        n_correct[ix, 0] += 1
                    if (float(bern_probs_proposed[ix] > 0.5) == label):
                        n_correct[ix, 1] += 1
                    likelihood_part = label * bern_probs_particles[ix] + (1 - label) * (1 - bern_probs_particles[ix])
                    likelihood_prop = label * bern_probs_proposed[ix] + (1 - label) * (1 - bern_probs_proposed[ix])
                    likelihoods[ix, 0] += np.log(likelihood_part + 1e-8)
                    likelihoods[ix, 1] += np.log(likelihood_prop + 1e-8)
            print('EXP:', (n_correct / len(experience)).max())

            print('Correct of ALL Samples:')
            print((n_correct[:, 0] / len(experience)).mean())

            # Calculate M-H acceptance prob. Uncomment if using a non-symmetric proposal distribution.
            prop_probs = np.zeros((N, 2))
            for ix in range(N):
                prop_probs[ix, 0] = np.log(multivariate_normal.pdf(particles[ix, :], mean=mean, cov=cov) + 1e-8)
                prop_probs[ix, 1] = np.log(
                    multivariate_normal.pdf(proposed_particles[ix, :], mean=mean, cov=cov) + 1e-8)
            p_accept = likelihoods[:, 1] + prop_probs[:, 0] - (likelihoods[:, 0] + prop_probs[:, 1])

            # p_accept = likelihoods[:,1] - (likelihoods[:,0])
            accept = np.zeros((N, 2))
            accept[:, 0] = p_accept
            # print(np.concatenate([likelihoods, prop_probs, accept, n_correct], axis=1)[:20, :])
            accept = np.min(accept, axis=1)

            # Keep particles based on acceptance probability.
            u = np.random.uniform(size=N)
            indices = np.argwhere(u > 1 - np.exp(accept)).flatten()
            print('Accept Rate:', len(indices) / N)
            particles[indices] = proposed_particles[indices]
        else:
            particles = proposed_particles

        weights = np.ones(N) / float(N)  # weights become uniform again
        return ParticleDistribution(particles, weights)

    def get_particle_likelihoods(self, particles, observation):
        """
         
        """
        self.likelihood.eval()
        dataset = GraspDataset(data=observation, grasp_encoding='per_point')
        dataloader = GraspParallelDataLoader(dataset=dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             n_dataloaders=1)
        bernoulli_probs = []  # Contains one prediction for each particle.

        latent_samples = torch.Tensor(particles)  # (N, 4)
        if torch.cuda.is_available():
            latent_samples = latent_samples.cuda()
        for set_of_batches in dataloader:
            grasps, object_ixs, _ = set_of_batches[0]
            if torch.cuda.is_available():
                grasps = grasps.cuda()
                object_ixs = object_ixs.cuda()

            for ix in range(0, latent_samples.shape[0] // 20):
                pred = self.likelihood.forward(X=grasps[:, :-5, :],
                                               object_ids=object_ixs.long(),
                                               N_samples=20,
                                               collapse_latents=True,
                                               collapse_ensemble=True,
                                               pf_latent_ix=self.object_ix,
                                               latent_samples=latent_samples[ix * 20:(ix + 1) * 20, :]).squeeze()
                bernoulli_probs.append(pred.cpu().detach().numpy())
        return np.concatenate(bernoulli_probs)

    def get_label_from_observation(self, observation):
        return observation['grasp_data']['labels'][0]

    def update(self, observation):
        """
        :param observation: tower_dict format for a single tower.
        """
        # Resample the distribution
        if len(self.experience) > 0 and self.resample:
            self.particles = self.sample_and_wiggle(self.particles, self.experience)

        # Append the current observation to the dataset of all observations so far.
        self.experience.append(observation)

        # Forward simulation using the LatentEnsemble likelihood.
        if isinstance(self.likelihood, nn.Module):
            bernoulli_probs = self.get_particle_likelihoods(self.particles.particles, observation)
        else:
            bernoulli_probs = self.likelihood.get_particle_likelihoods(self.particles.particles, observation)

        label = self.get_label_from_observation(observation)

        n_correct = ((bernoulli_probs > 0.5).astype('float32') == label).sum()
        print('Correct for CURRENT sample:', n_correct / len(bernoulli_probs), len(bernoulli_probs))

        new_weights = []
        for pi, (bern_prob, old_weight) in enumerate(zip(bernoulli_probs, self.particles.weights)):
            # print(pi, bern_prob, old_weight)
            obs_model = bern_prob * label + (1 - bern_prob) * (1 - label)
            new_weight = old_weight * obs_model
            new_weights.append(new_weight)

        # normalize particle weights
        new_weights = np.array(new_weights)
        # If resampling, normalize weights as they will be used as probabilities.
        if self.resample:
            new_weights /= np.sum(new_weights)

        # and update the particle distribution with the new weights
        self.particles = ParticleDistribution(self.particles.particles, new_weights)

        if self.plot:
            # visualize particles (it's very slow)
            self.setup_ax(self.ax)
            self.plot_particles(self.ax, self.particles.particles, new_weights)

        part_probs = np.array(self.particles.weights)
        part_probs /= part_probs.sum()
        mean = np.array(self.particles.particles).T @ part_probs
        print('Particle Mean:', mean)
        print('True:', self.object_properties)
        self.estimated_coms.append(mean)

        return self.particles, self.estimated_coms


class AmortizedGraspingDiscreteLikelihoodParticleBelief(GraspingDiscreteLikelihoodParticleBelief):
    """
    A ParticleBelief that is compatible with a GraspNeuralProcess object.
    """

    def __init__(self, object_set, d_latents, n_particles, likelihood=None, resample=False, plot=False,
                 data_is_in_gnp_format=False):
        super().__init__(object_set, d_latents, n_particles, likelihood=likelihood, resample=resample, plot=plot)
        self.data_is_in_gnp_format = data_is_in_gnp_format

    def get_label_from_observation(self, observation):
        if self.data_is_in_gnp_format:
            return float(list(observation['grasp_data']['labels'].values())[0][0])
        else:
            return observation['grasp_data']['labels'][0]

    def get_particle_likelihoods(self, particles, observation, batch_size=1000):
        """
        Compute the likelihood of an obervation for each particle.
        :param particles: NxD matrix of particles.
        :param observation: A grasp dictionary containing a single datapoint/grasp.
            { 'grasp_data': {} , 'object_data: {} , 'metadata': {} }
        """
        self.likelihood.eval()

        if self.data_is_in_gnp_format:
            gnp_observation = observation
        else:
            gnp_observation = process_geometry(
                observation,
                radius=0.03,
                skip=1,
                verbose=False
            )

        # Note the context data is irrelevant here as we are only using the decoder.
        dataset = CustomGNPGraspDataset(
            data=gnp_observation,
            context_data=gnp_observation
        )
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=custom_collate_fn,
            batch_size=1,
            shuffle=False
        )

        bernoulli_probs = []
        latent_samples = torch.Tensor(particles)
        if torch.cuda.is_available():
            latent_samples = latent_samples.cuda()
            self.likelihood.cuda()

        for (_, target_data, meshes) in dataloader:
            t_grasp_geoms, t_grasp_points, t_curvatures, t_midpoints, t_forces, _ = check_to_cuda(target_data)

            if torch.cuda.is_available():
                meshes = meshes.cuda()

            t_grasp_geoms = t_grasp_geoms.expand(batch_size, -1, -1, -1)
            t_grasp_points = t_grasp_points.expand(batch_size, -1, -1, -1)
            t_curvatures = t_curvatures.expand(batch_size, -1, -1, -1)
            t_midpoints = t_midpoints.expand(batch_size, -1, -1)
            t_forces = t_forces.expand(batch_size, -1)

            for ix in range(0, latent_samples.shape[0] // batch_size):
                preds = self.likelihood.conditional_forward(
                    target_xs=(t_grasp_geoms, t_grasp_points, t_curvatures, t_midpoints, t_forces),
                    meshes=meshes,
                    zs=latent_samples[ix * batch_size:(ix + 1) * batch_size]
                ).squeeze()
                bernoulli_probs.append(preds.cpu().detach().numpy())

        return np.concatenate(bernoulli_probs)


# =============================================================
"""
A script that tests the particle filter by reporting the error on
the CoM estimate as we get more observations.
"""


def plot_com_error(errors_random, errors_var):
    for tx in range(0, len(errors_var[0][0])):
        err_rand, err_var = 0, 0
        for bx in range(0, len(errors_var)):
            true = np.array(errors_var[bx][1])
            guess_rand = errors_random[bx][0][tx]
            guess_var = errors_var[bx][0][tx]
            err_var += np.linalg.norm(true - guess_var)
            err_rand += np.linalg.norm(true - guess_rand)
        plt.scatter(tx, err_rand / len(errors_var), c='r')
        plt.scatter(tx, err_var / len(errors_var), c='b')
    plt.show()


"""
Notes on tuning the particle filter.
- When plot=True for ParticleBelief, we want to see the particles (blue) become 
  more tightly distributed around the true CoM (red). 
- Make sure some particles are initialized near the true CoM.
- Check resampling-step if particles don't converge to true CoM.
-- Are M-H steps being accepted? Are we removing unlikely samples?
- I have observed that for PlaceAction, n_particles=200, and n_actions=10, 
  the distribution converges pretty tightly for all adversarial blocks.
- If the particles jump around too much, the true noise might be too large.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['teleport', 'panda'], default='teleport')
    parser.add_argument('--n-particles', type=int, default=10)
    parser.add_argument('--n-actions', type=int, default=2)
    args = parser.parse_args()
    NOISE = 0.00005

    # get a bunch of random blocks
    blocks = get_adversarial_blocks()

    if args.agent == 'teleport':
        agent = TeleportAgent(blocks, NOISE)
    else:
        agent = PandaAgent(blocks, NOISE, teleport=False)

    # construct a world containing those blocks
    for b_ix, block in enumerate(blocks):
        # new code
        print('Running filter for', block.name, block.dimensions)
        belief = ParticleBelief(block,
                                N=args.n_particles,
                                plot=True,
                                vis_sim=False,
                                noise=NOISE)
        for interaction_num in range(args.n_actions):
            print('----------')
            # print(belief.particles.particles[::4, :])
            print("Interaction number: ", interaction_num)
            action = plan_action(belief, exp_type='reduce_var', action_type='place')
            observation = agent.simulate_action(action, b_ix, T=50, vis_sim=False)
            belief.update(observation)
            block.com_filter = belief.particles

            est = belief.estimated_coms[-1]
            true = np.array(block.com)
            error = np.linalg.norm(est - true)
            print('Estimated CoM:', est)
            print('True:', true)
            print('Error:', error)
