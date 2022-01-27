import numpy
import sys
from Algorithms.CrossEntropy.Executor.DANE_cross_entropy_executor import DANECrossEntropyExecutor
from Algorithms.CrossEntropy.Solver.cross_entropy_solver import CrossEntropySolver
from Utils.DCA import dca_solver
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY
from tqdm import tqdm

home_dir = '../../../'
sys.path.append(home_dir)


class DANECrossEntropySolver(CrossEntropySolver):
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

            executor = DANECrossEntropyExecutor(x_blk, y_blk, self.num_class)
            self.executor_list.append(executor)

    def update(self, w, is_search, t):
        # first communication round
        # initialization
        grad_list = list()
        grad = numpy.zeros((self.num_class * self.d, 1))
        optimization_scaling_factor = 0

        # local computation
        current_h_mat = numpy.copy(self.h_mat[t])
        for i in tqdm(range(self.m)):
            data_size = self.executor_list[i].get_data_size()
            local_g = self.executor_list[i].compute_gradient()
            self.executor_list[i].update_g(local_g)
            grad_list.append(data_size * local_g)
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  local_g))
            current_h_mat[:, i] = current_h_mat[:, i] / (
                    (data_size / 1) * numpy.linalg.norm(local_g))
        current_h_mat = optimization_scaling_factor * current_h_mat

        # system optimization
        self.selected_set = range(self.m)
        self.a = optimization_scaling_factor * dca_solver(self.selected_set, current_h_mat)

        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            grad += grad_list[j]
        grad /= total_data_size
        if self.opt_mode != PERFECT_AGGREGATION:
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k,
                                                                  self.num_class * self.d) + 1j * numpy.random.randn(
                self.k,
                self.num_class * self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.num_class * self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.H, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            grad = numpy.real(numpy.add(grad, noise_vec))
        for executor in self.executor_list:
            executor.update_global_gradient(grad)

        # second communication round
        # initialization
        h_vec = numpy.zeros((self.num_class * self.d, 1))
        h_vec_list = list()
        optimization_scaling_factor = 0

        # local computation
        current_h_mat = numpy.copy(self.h_mat[t])
        for i in tqdm(range(self.m)):
            data_size = self.executor_list[i].get_data_size()
            local_h = self.executor_list[i].gradient_descent_solver()
            h_vec_list.append(data_size * local_h)
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  local_h))
            current_h_mat[:, i] = current_h_mat[:, i] / (
                    (data_size / 1) * numpy.linalg.norm(local_h))
        current_h_mat = optimization_scaling_factor * current_h_mat

        # system optimization
        self.selected_set = range(self.m)
        self.a = optimization_scaling_factor * dca_solver(self.selected_set, current_h_mat)

        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            h_vec += h_vec_list[j]
        h_vec /= total_data_size
        if self.opt_mode != PERFECT_AGGREGATION:
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k,
                                                                  self.num_class * self.d) + 1j * numpy.random.randn(
                self.k,
                self.num_class * self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.num_class * self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.H, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            h_vec = numpy.real(numpy.add(h_vec, noise_vec))

        # print(h_vec)
        for executor in self.executor_list:
            executor.set_w(w + h_vec)

        return w + h_vec
