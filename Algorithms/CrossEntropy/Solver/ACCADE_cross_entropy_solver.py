import numpy
import sys
from Algorithms.CrossEntropy.Executor.ACCADE_cross_entropy_executor import ACCADECrossEntropyExecutor
from Algorithms.CrossEntropy.Solver.cross_entropy_solver import CrossEntropySolver
from Algorithms.System.system_optimization_solver import SystemOptimizationSolver
from Utils.DCA import dca_solver, sdr_solver
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY
from tqdm import tqdm

home_dir = '../../../'
sys.path.append(home_dir)

eta_list = 1 / (4 ** numpy.arange(0, 10))


class ACCADECrossEntropySolver(CrossEntropySolver):
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

            executor = ACCADECrossEntropyExecutor(x_blk, y_blk, self.num_class)
            self.executor_list.append(executor)

    def update(self, w, is_search, t):
        # initialization
        grad_list = list()
        p_vec_list = list()
        grad = numpy.zeros((self.num_class * self.d, 1))
        p_vec = numpy.zeros((self.num_class * self.d, 1))
        optimization_scaling_factor = 0

        # local computation
        for i in tqdm(range(self.m)):
            data_size = self.executor_list[i].get_data_size()
            grad_list.append(data_size * self.executor_list[i].compute_gradient())
            self.executor_list[i].update_p(self.executor_list[i].compute_newton())
            p_vec_list.append(data_size * self.executor_list[i].get_p())
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  self.executor_list[i].get_p()))
            self.h_mat[t, :, i] = self.h_mat[t, :, i] / (
                    (data_size / 1) * numpy.linalg.norm(self.executor_list[i].get_p()))
        self.h_mat[t, :, :] = optimization_scaling_factor * self.h_mat[t, :, :]

        # system optimization
        if self.opt_mode == GS_DCA or self.opt_mode == GS_SDR:
            opt_solver = SystemOptimizationSolver(self.p, self.m, self.lam, self.delta, self.p_var_bound,
                                                  self.gradient_bound, self.d, self.s, self.k, self.tau,
                                                  self.h_mat[t], self.data_size_list, optimization_scaling_factor)
            opt_res = opt_solver.gibbs_sampling_based_device_selection(self.opt_mode, beta=100, rho=0.9)
            self.a, self.selected_set = opt_res[0], opt_res[1]
            print('device selection', self.selected_set)
        elif self.opt_mode == PERFECT_AGGREGATION:
            self.a = numpy.ones((self.k, 1))
            self.selected_set = range(self.m)
        elif self.opt_mode == DCA_ONLY:
            self.selected_set = range(self.m)
            self.a = dca_solver(self.selected_set, self.h_mat[t])
            # print('initial beamforming', self.a)
        elif self.opt_mode == SDR_ONLY:
            self.selected_set = range(self.m)
            self.a = sdr_solver(self.selected_set, self.h_mat[t])
        self.a = numpy.multiply(optimization_scaling_factor, self.a)

        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            p_vec += p_vec_list[j]
            grad += grad_list[j]
        p_vec /= total_data_size
        grad /= total_data_size
        if self.opt_mode != PERFECT_AGGREGATION:
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k,
                                                                  self.num_class * self.d) + 1j * numpy.random.randn(
                self.k,
                self.num_class * self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.num_class * self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.T, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            p_vec = numpy.real(numpy.add(p_vec, noise_vec))

        for executor in self.executor_list:
            executor.update_p(p_vec)

        if is_search:
            pg = -0.1 * numpy.sum(numpy.multiply(p_vec, grad))
            objective_value_vec = numpy.zeros(self.num_etas + 1)
            for executor in self.executor_list:
                objective_value_vec += executor.search_objective_val()

            eta = 0
            objective_value_old = objective_value_vec[-1]
            for i in range(self.num_etas):
                objective_value_new = objective_value_vec[i]
                eta = self.eta_list[i]
                if objective_value_new < objective_value_old + pg * eta:
                    break

            p_vec = numpy.multiply(eta, p_vec)
        for executor in self.executor_list:
            executor.update_p(p_vec)

        for executor in self.executor_list:
            executor.update_w()
        w = numpy.subtract(w, p_vec)
        return w
