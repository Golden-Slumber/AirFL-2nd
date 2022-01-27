import numpy
import sys
from Algorithms.CrossEntropy.Executor.cross_entropy_executor import CrossEntropyExecutor
from Algorithms.CrossEntropy.Solver.cross_entropy_solver import CrossEntropySolver
from Utils.DCA import sparse_optimization_dca, feasibility_detection_dca
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY, GBMA
from tqdm import tqdm

home_dir = '../../../'
sys.path.append(home_dir)

eta_list = 1 / (2 ** numpy.arange(0, 10))

class FedGDCrossEntropySolver(CrossEntropySolver):

    def fit(self, x_mat, y_vec, data_size_list, shards):
        """
        Partition X and y to self.m blocks.
        If s is not given, then we set s=n/m and the partition has no overlap.
        """
        n, self.d = x_mat.shape
        self.data_size_list = data_size_list
        self.n = sum(data_size_list)
        self.s = self.n // self.m

        for i in range(self.m):
            idx = shards[i]
            x_blk = x_mat[idx, :]
            y_blk = y_vec[idx, :].reshape(self.s, 1)
            executor = CrossEntropyExecutor(x_blk, y_blk, self.num_class)
            self.executor_list.append(executor)

    def update(self, w, is_search, t):
        # initialization
        p_vec_list = list()
        grad = numpy.zeros((self.num_class * self.d, 1))
        optimization_scaling_factor = 0

        # local computation
        for i in tqdm(range(self.m)):
            data_size = self.executor_list[i].get_data_size()
            local_grad = self.executor_list[i].compute_gradient()
            self.executor_list[i].update_p(local_grad)
            p_vec_list.append(data_size * local_grad)
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  self.executor_list[i].get_p()))
            self.h_mat[t, :, i] = self.h_mat[t, :, i] / (
                    (data_size / 1) * numpy.linalg.norm(self.executor_list[i].get_p()))
        self.h_mat[t, :, :] = optimization_scaling_factor * self.h_mat[t, :, :]

        # system optimization
        theta = numpy.sqrt(10)
        v = sparse_optimization_dca(self.h_mat[t], theta)
        self.selected_set, self.a = feasibility_detection_dca(v, self.h_mat[t], theta)
        self.a = numpy.multiply(optimization_scaling_factor, self.a)

        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            grad += p_vec_list[j]
        grad /= total_data_size

        if self.opt_mode != PERFECT_AGGREGATION:
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k, self.num_class * self.d) + 1j * numpy.random.randn(self.k,
                                                                                                            self.num_class * self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.num_class * self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.H, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            grad = numpy.real(numpy.add(grad, noise_vec))

        for executor in self.executor_list:
            executor.update_p(grad)

        eta = 0.001
        grad = numpy.multiply(eta, grad)
        for executor in self.executor_list:
            executor.update_p(grad)

        for executor in self.executor_list:
            executor.update_w()
        w = numpy.subtract(w, grad)
        return w
