import numpy
import sys
from Algorithms.CrossEntropy.Executor.Fedsplit_cross_entropy_executor import FedSplitCrossEntropyExecutor
from Algorithms.CrossEntropy.Solver.cross_entropy_solver import CrossEntropySolver
from Utils.DCA import dca_solver
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY, THRESHOLD
from tqdm import tqdm

home_dir = '../../../'
sys.path.append(home_dir)


class FedSplitCrossEntropySolver(CrossEntropySolver):
    def fit(self, x_mat, y_vec, data_size_list, shards):
        """
        Partition X and y to self.m blocks.
        If s is not given, then we set s=n/m and the partition has no overlap.
        """
        n, self.d = x_mat.shape
        initial_w = numpy.zeros((self.num_class * self.d, 1))

        self.data_size_list = data_size_list
        self.n = sum(data_size_list)
        self.s = self.n // self.m

        for i in range(self.m):
            idx = shards[i]
            x_blk = x_mat[idx, :]
            y_blk = y_vec[idx, :].reshape(self.s, 1)

            executor = FedSplitCrossEntropyExecutor(x_blk, y_blk, self.num_class, initial_w)
            self.executor_list.append(executor)

    def update(self, w, is_search, t):
        # initialization
        w_vec_list = list()
        w_vec = numpy.zeros((self.num_class * self.d, 1))
        optimization_scaling_factor = 0

        # local computation
        for i in tqdm(range(self.m)):
            data_size = self.executor_list[i].get_data_size()
            local_w = self.executor_list[i].compute_w_proximal()
            w_vec_list.append(data_size * local_w)
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  local_w))
            self.h_mat[t, :, i] = self.h_mat[t, :, i] / (
                    (data_size / 1) * numpy.linalg.norm(local_w))
        self.h_mat[t, :, :] = optimization_scaling_factor * self.h_mat[t, :, :]

        # system optimization
        self.selected_set = list()
        for i in range(self.m):
            h_vec = self.h_mat[t, :, i]
            if numpy.linalg.norm(h_vec) >= 1 / (2 * self.m):
                self.selected_set.append(i)
        self.a = optimization_scaling_factor * dca_solver(self.selected_set, self.h_mat[t])

        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            w_vec += w_vec_list[j]
        w_vec /= total_data_size
        if self.opt_mode != PERFECT_AGGREGATION:
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k,
                                                                  self.num_class * self.d) + 1j * numpy.random.randn(
                self.k,
                self.num_class * self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.num_class * self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.H, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            w_vec = numpy.real(numpy.add(w_vec, noise_vec))

        for executor in self.executor_list:
            executor.update_global_w(w_vec)

        return w_vec
