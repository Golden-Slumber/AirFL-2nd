import numpy
import sys
from Resources.data_loader import load_data
from Algorithms.Logistic.Executor.ACCADE_logistic_executor import ACCADELogisticExecutor
from Utils.DCA import dca_solver, sdr_solver
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY
from Algorithms.System.system_optimization_solver import SystemOptimizationSolver

home_dir = '../../../'
sys.path.append(home_dir)

eta_list = 1 / (4 ** numpy.arange(0, 10))


class ACCADELogisticSolver:
    def __init__(self, m=None, h_mat=None, tau=None, p=None, x_test=None, y_test=None, opt_mode=None):
        self.m = m
        self.h_mat = h_mat
        self.k = h_mat.shape[1]
        self.tau = tau
        self.p = p
        self.x_test = x_test
        self.y_test = y_test
        self.opt_mode = opt_mode

        self.eta_list = eta_list
        self.num_etas = len(eta_list)
        self.executor_list = list()
        self.n = 0
        self.d = None
        self.s = None
        self.a = None
        self.selected_set = None
        self.w = None
        self.data_size_list = None

        self.lam = 0.1
        self.delta = 0.01
        self.p_var_bound = 5
        self.gradient_bound = 100

    def fit(self, x_mat, y_vec, data_size_list):
        """
        Partition X and y to self.m blocks.
        If s is not given, then we set s=n/m and the partition has no overlap.
        """
        n, self.d = x_mat.shape
        self.data_size_list = data_size_list
        i_begin = 0
        for i in range(self.m):
            idx = range(i_begin, i_begin + data_size_list[i])
            i_begin += data_size_list[i]
            x_blk = x_mat[idx, :]
            y_blk = y_vec[idx, :].reshape(data_size_list[i], 1)
            self.n += data_size_list[i]

            executor = ACCADELogisticExecutor(x_blk, y_blk)
            self.executor_list.append(executor)
        self.s = int(numpy.floor(self.n / self.m))

    def train(self, gamma, w_opt, max_iter=20, is_search=True, newton_tol=1e-10, newton_max_iter=20):
        error_list = list()
        accuracy_list = list()
        w = numpy.zeros((self.d, 1))

        for executor in self.executor_list:
            executor.set_param(gamma, newton_tol, newton_max_iter, is_search, self.eta_list)

        err_initial = self.objective_function(w) - self.objective_function(w_opt)
        error_list.append(err_initial)
        acc_initial = self.accuracy(w)
        accuracy_list.append(acc_initial)

        for t in range(max_iter):
            w = self.update(w, is_search, t)

            err = self.objective_function(w) - self.objective_function(w_opt)
            error_list.append(err)
            acc = self.accuracy(w)
            accuracy_list.append(acc)
            print('iter ' + str(t) + ': error is ' + str(err) + ' , accuracy is ' + str(acc))

        self.w = w
        return error_list, accuracy_list

    def accuracy(self, w):
        num = self.x_test.shape[0]
        count = 0
        idx = 0
        for row in self.x_test:
            if numpy.sign(numpy.dot(row, w.reshape(self.d, 1)))[0] == self.y_test[idx, 0]:
                count += 1
            idx += 1
        return count / num

    def objective_function(self, w):
        loss = 0
        for executor in self.executor_list:
            loss += executor.get_data_size() * executor.objective_function(w)
        loss /= self.n
        return loss

    def update(self, w, is_search, t):

        grad_list = list()
        p_vec_list = list()
        p_vec = numpy.zeros((self.d, 1))
        grad = numpy.zeros((self.d, 1))

        optimization_scaling_factor = 0
        for i in range(self.m):
            data_size = self.executor_list[i].get_data_size()
            grad_list.append(data_size * self.executor_list[i].compute_gradient())
            self.executor_list[i].update_p(self.executor_list[i].compute_newton())
            p_vec_list.append(data_size * self.executor_list[i].get_p())
            # print(self.h_mat[t, :, i])
            optimization_scaling_factor = max(optimization_scaling_factor,
                                              1e3 * (data_size / 1) * numpy.linalg.norm(
                                                  self.executor_list[i].get_p()))
            self.h_mat[t, :, i] = self.h_mat[t, :, i] / (
                    (data_size / 1) * numpy.linalg.norm(self.executor_list[i].get_p()))
            # print(self.h_mat[t, :, i])
        self.h_mat[t, :, :] = optimization_scaling_factor * self.h_mat[t, :, :]
        # print('optimization scaling factor', optimization_scaling_factor)

        # print(self.h_mat[t])
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
            # self.selected_set = list()
            # for device in range(self.m-2):
            #     self.selected_set.append(device+2)
            # self.a = dca_solver(self.selected_set, self.h_mat[t])
        self.a = numpy.multiply(optimization_scaling_factor, self.a)

        total_data_size = 0
        for j in self.selected_set:
            total_data_size += self.executor_list[j].get_data_size()
            p_vec += p_vec_list[j]
            grad += grad_list[j]
        p_vec /= total_data_size
        grad /= total_data_size

        if self.opt_mode != PERFECT_AGGREGATION:
            # noise = []
            # for i in range(self.k):
            #     noise.append(numpy.random.normal(0, self.tau, (1, 2 * self.d)).view(numpy.complex128))
            # noise = numpy.concatenate(noise)
            noise = numpy.sqrt(self.tau / 2) * numpy.random.randn(self.k, self.d) + 1j * numpy.random.randn(self.k,
                                                                                                            self.d) * numpy.sqrt(
                self.tau / 2)
            # coefficient = numpy.sqrt(1 / self.p) * self.n / total_data_size
            coefficient = numpy.sqrt(1 / (self.d * self.p)) / total_data_size
            noise_vec = numpy.dot(self.a.T, noise).T
            noise_vec = numpy.multiply(coefficient, noise_vec)
            # print('beamforming', self.a)
            # print('noise', noise_vec)
            p_vec = numpy.real(numpy.add(p_vec, noise_vec))

        for executor in self.executor_list:
            executor.update_p(p_vec)

        if is_search:
            pg = -0.1 * numpy.sum(numpy.multiply(p_vec, grad))
            objective_value_vec = numpy.zeros(self.num_etas + 1)
            for executor in self.executor_list:
                objective_value_vec += executor.get_data_size() * executor.search_objective_val()

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


if __name__ == '__main__':
    demo = ACCADELogisticSolver(m=20)
    data_name = 'covtype'
    x_mat, y_vec = load_data(data_name=data_name, file_path=home_dir + 'Resources/' + data_name + '.npz')
    x_mat = numpy.mat(x_mat)
    y_vec = numpy.mat(y_vec).T
    n, d = x_mat.shape
    s = int(numpy.floor(n / 20))
    demo.fit(x_mat, y_vec)
