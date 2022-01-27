import numpy
import sys
from Algorithms.Logistic.Executor.GIANT_logistic_executor import GIANTLogisticExecutor
from Algorithms.Logistic.Solver.logistic_solver import LogisticSolver
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY
from Utils.DCA import dca_solver, sdr_solver

class GIANTLogisticSolver(LogisticSolver):
    def __init__(self, m=None, h_mat=None, tau=None, p=None, x_test=None, y_test=None, opt_mode=None):
        super().__init__(m, h_mat, tau, p, x_test, y_test, opt_mode)
        self.eta_list = 1 / (2 ** numpy.arange(0, 15))

    def fit(self, x_mat, y_vec, data_size_list):
        """
        Partition X and y to self.m blocks.
        If s is not given, then we set s=n/m and the partition has no overlap.
        """
        n, self.d = x_mat.shape
        # perm = numpy.random.permutation(n)
        # self.x_mat = x_mat[perm, :]
        # self.y_vec = y_vec[perm, :]
        #
        # self.s = int(numpy.floor(n / self.m))
        # self.n = int(self.s * self.m)
        #
        # i_begin = 0
        # for i in range(self.m):
        #     idx = range(i_begin, i_begin + self.s)
        #     i_begin += self.s
        #     x_blk = x_mat[idx, :]
        #     y_blk = y_vec[idx, :].reshape(self.s, 1)
        #
        #     executor = GIANTLogisticExecutor(x_blk, y_blk)
        #     self.executor_list.append(executor)
        self.data_size_list = data_size_list
        i_begin = 0
        for i in range(self.m):
            idx = range(i_begin, i_begin + data_size_list[i])
            i_begin += data_size_list[i]
            x_blk = x_mat[idx, :]
            y_blk = y_vec[idx, :].reshape(data_size_list[i], 1)
            self.n += data_size_list[i]

            executor = GIANTLogisticExecutor(x_blk, y_blk)
            self.executor_list.append(executor)
        self.s = int(numpy.floor(self.n / self.m))

    def update(self, w, is_search, t):
        grad_list = list()
        grad = numpy.zeros((self.d, 1))
        optimization_scaling_factor = 0

        # local computation
        current_h_mat = numpy.copy(self.h_mat[t])
        for i in range(self.m):
            data_size = self.executor_list[i].get_data_size()
            local_grad = self.executor_list[i].compute_gradient()
            # print(local_grad)
            self.executor_list[i].update_g(local_grad)
            grad_list.append(data_size * local_grad)
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  local_grad))
            current_h_mat[:, i] = current_h_mat[:, i] / (
                    (data_size / 1) * numpy.linalg.norm(local_grad))
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
            # noise = []
            # for i in range(self.k):
            #     noise.append(numpy.random.normal(0, self.tau, (1, 2 * self.d)).view(numpy.complex128))
            # noise = numpy.concatenate(noise)
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k, self.d) + 1j * numpy.random.randn(self.k,
                                                                                                            self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / self.d * self.p) / total_data_size
            noise_vec = numpy.multiply(coefficient, numpy.dot(self.a.H, noise).T)
            # print(noise_vec)
            grad = numpy.real(numpy.add(grad, noise_vec))
        # print(grad)

        for executor in self.executor_list:
            executor.set_global_gradient(grad)


        optimization_scaling_factor = 0
        p_vec_list = list()
        p_vec = numpy.zeros((self.d, 1))

        # local computation
        current_h_mat = numpy.copy(self.h_mat[t])
        for i in range(self.m):
            data_size = self.executor_list[i].get_data_size()
            local_p = self.executor_list[i].compute_newton()
            self.executor_list[i].update_p(local_p)
            p_vec_list.append(data_size * local_p)
            current_h_mat[:, i] = current_h_mat[:, i] / (
                    (data_size / 1) * numpy.linalg.norm(local_p))
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  local_p))
        current_h_mat = optimization_scaling_factor * current_h_mat

        # system optimization
        self.selected_set = range(self.m)
        self.a = optimization_scaling_factor * dca_solver(self.selected_set, current_h_mat)

        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            p_vec += p_vec_list[j]
        p_vec /= total_data_size
        if self.opt_mode != PERFECT_AGGREGATION:
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k, self.d) + 1j * numpy.random.randn(self.k,
                                                                                                            self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.d * self.p)) / total_data_size
            noise_vec = numpy.multiply(coefficient, numpy.dot(self.a.H, noise).T)
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
