from learning.domains.grasping.active_utils import sample_unlabeled_data
import numpy as np
import pickle
import multiprocessing

from block_utils import ParticleDistribution
from learning.domains.grasping.active_utils import sample_unlabeled_data
from learning.domains.grasping.generate_grasp_datasets import vector_from_graspablebody, graspablebody_from_vector
from pb_robot.planners.antipodalGraspPlanner import Grasp, GraspableBodySampler, GraspStabilityChecker, \
    GraspSimulationClient


# this is a helper function to verify grasp stability from a single pybullet instance
# (and will be parallelized)
def get_label(body, cand_grasp):
    labeler = GraspStabilityChecker(body, grasp_noise=0.0025, recompute_inertia=True)
    label = labeler.get_label(cand_grasp)
    labeler.disconnect()
    return label


class PBLikelihood:

    def __init__(self, object_name, n_samples, batch_size, n_processes=20):
        self.object_name = object_name
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.bodies_for_particles = None
        self.n_processes = n_processes

    def eval(self):
        pass

    def init_particles(self, N):
        graspable_bodies = []
        graspable_vectors = []
        for px in range(N):
            graspable_bodies.append(GraspableBodySampler.sample_random_object_properties(self.object_name))
            graspable_vectors.append(vector_from_graspablebody(graspable_bodies[-1]))

        self.bodies_for_particles = graspable_bodies
        particle_dist = ParticleDistribution(np.array(graspable_vectors), np.ones(len(graspable_vectors)))
        return particle_dist

    def particle_distribution_from_graspable_vectors(self, graspable_vectors):
        graspable_bodies = []
        for gsp_vect in graspable_vectors:
            body = graspablebody_from_vector(self.object_name, gsp_vect)

            # create a simulation client to create a temporary urdf with given particle properties
            client = GraspSimulationClient(body, False, recompute_inertia=True)
            client.disconnect()

            graspable_bodies.append(body)
        self.bodies_for_particles = graspable_bodies
        return ParticleDistribution(np.array(graspable_vectors), np.ones(len(graspable_vectors)))

    def get_particle_likelihoods(self, particles, observation):
        worker_pool = multiprocessing.Pool(self.n_processes, maxtasksperchild=1)

        if self.bodies_for_particles is None:
            print('[ERROR] Need to initialize particles first.')
            return

        tgrasp = observation['grasp_data']['raw_grasps'][0]

        labels = []
        n_batches = int(np.ceil(len(particles) / self.batch_size))
        for bx in range(n_batches):
            bodies = self.bodies_for_particles[bx * self.batch_size:(bx + 1) * self.batch_size]
            grasps = []
            for body in bodies:
                g = Grasp(body, tgrasp.pb_point1, tgrasp.pb_point2, tgrasp.pitch, tgrasp.roll, tgrasp.ee_relpose,
                          tgrasp.force)
                grasps.append(g)

            batch_labels = []
            for sx in range(self.n_samples):
                # parallelizing labeling process for all grasps for one object
                object_labels = worker_pool.starmap(get_label, zip(bodies, grasps))
                batch_labels.append(object_labels)

            batch_labels = np.array(batch_labels).mean(axis=0).tolist()
            labels += batch_labels

        # gracefully close the pool to free up threads to prevent system lockup
        worker_pool.close()
        return np.array(labels)


if __name__ == '__main__':
    with open('learning/data/grasping/train-sn100-test-sn10/objects/test_geo_test_props.pkl', 'rb') as handle:
        all_object_set = pickle.load(handle)['object_data']

    object_ix = 0
    object_set = {
        'object_names': [all_object_set['object_names'][object_ix]],
        'object_properties': [all_object_set['object_properties'][object_ix]],
    }

    object_name = object_set['object_names'][object_ix]
    likelihood = PBLikelihood(object_name=object_name, n_samples=5, batch_size=50)
    particles = likelihood.init_particles(100)

    grasp = sample_unlabeled_data(1, object_set)
    probs = likelihood.get_particle_likelihoods(particles.particles, grasp)
    print(probs)
