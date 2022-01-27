import numpy
import sys
from Algorithms.Logistic.Executor.logistic_executor import LogisticExecutor
from Algorithms.Logistic.Solver.logistic_solver import LogisticSolver
from Utils.DCA import sparse_optimization_dca, feasibility_detection_dca, dca_solver
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY

home_dir = '../../../'
sys.path.append(home_dir)

# eta_list = 1 / (2 ** numpy.arange(0, 10))
eta_list = [10, 1, 0.1, 0.01, 0.001, 1e-4]

class FedGDLogisticSolver(LogisticSolver):

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
        #     executor = LogisticExecutor(x_blk, y_blk)
        #     self.executor_list.append(executor)
        self.data_size_list = data_size_list
        i_begin = 0
        for i in range(self.m):
            idx = range(i_begin, i_begin + data_size_list[i])
            i_begin += data_size_list[i]
            x_blk = x_mat[idx, :]
            y_blk = y_vec[idx, :].reshape(data_size_list[i], 1)
            self.n += data_size_list[i]

            executor = LogisticExecutor(x_blk, y_blk)
            self.executor_list.append(executor)
        self.s = int(numpy.floor(self.n / self.m))

    def update(self, w, is_search, t):
        p_vec_list = list()
        grad = numpy.zeros((self.d, 1))
        optimization_scaling_factor = 0

        # local computation
        for i in range(self.m):
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
        # print(v)
        self.selected_set, self.a = feasibility_detection_dca(v, self.h_mat[t], theta)
        # self.selected_set = range(self.m)
        # self.a = dca_solver(self.selected_set, self.h_mat[t])
        self.a = numpy.multiply(optimization_scaling_factor, self.a)

        print(self.selected_set)
        # aggregation
        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            grad += p_vec_list[j]
        grad /= total_data_size

        if self.opt_mode != PERFECT_AGGREGATION:
            # noise = []
            # for i in range(self.k):
            #     noise.append(numpy.random.normal(0, self.tau, (1, 2 * self.d)).view(numpy.complex128))
            # noise = numpy.concatenate(noise)
            #
            # coefficient = numpy.sqrt(p_var / self.p) / len(self.selected_set)
            # noise_vec = numpy.multiply(coefficient, numpy.dot(self.a.T, noise).T)
            # p_vec = numpy.real(numpy.add(p_vec, noise_vec))
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k, self.d) + 1j * numpy.random.randn(self.k,
                                                                                                            self.d) * numpy.sqrt(
                self.tau / 2)
            coefficient = numpy.sqrt(1 / (self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.H, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            grad = numpy.real(numpy.add(grad, noise_vec))

        for executor in self.executor_list:
            executor.update_p(grad)

        if is_search:
            pg = -0.1 * numpy.sum(numpy.multiply(grad, grad))
            objective_value_vec = numpy.zeros(self.num_etas + 1)
            for executor in self.executor_list:
                objective_value_vec += executor.search_objective_val()

            eta = 0
            objective_value_old = objective_value_vec[-1]
            for i in range(self.num_etas):
                objective_value_new = objective_value_vec[i]
                eta = self.eta_list[i]
                # print('eta: ', eta)
                # print(objective_value_new)
                # print(objective_value_old)
                # print(pg * eta)
                if objective_value_new < objective_value_old + pg * eta:
                    break

            eta = 2
            # eta = 1.3
            # eta = 0.8
            # eta = 10
            grad = numpy.multiply(eta, grad)
            for executor in self.executor_list:
                executor.update_p(grad)

        for executor in self.executor_list:
            executor.update_w()
        w = numpy.subtract(w, grad)
        return w
