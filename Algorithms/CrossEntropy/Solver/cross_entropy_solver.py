import numpy
import sys
from constants import GS_DCA, GS_SDR, PERFECT_AGGREGATION, DCA_ONLY, SDR_ONLY, THRESHOLD, GBMA, DC_FRAMEWORK
from Algorithms.System.system_optimization_solver import SystemOptimizationSolver
from Utils.CrossEntropy import stable_softmax
from Utils.DCA import dca_solver, sdr_solver, sparse_optimization_dca, feasibility_detection_dca

home_dir = '../../../'
sys.path.append(home_dir)

eta_list = 1 / (4 ** numpy.arange(0, 15))


class CrossEntropySolver:
    def __init__(self, m=None, h_mat=None, tau=None, p=None, x_test=None, y_test=None, opt_mode=None, num_class=None):
        self.m = m
        self.h_mat = h_mat
        self.k = h_mat.shape[1]
        self.tau = tau
        self.p = p
        self.x_test = x_test
        self.y_test = y_test
        self.opt_mode = opt_mode
        self.num_class = num_class

        self.eta_list = eta_list
        self.num_etas = len(eta_list)
        self.executor_list = list()
        self.n = None
        self.d = None
        self.s = None
        self.a = None
        self.selected_set = None
        self.w = None
        self.x_mat = None
        self.y_vec = None
        self.data_size_list = None

        self.lam = 0.1
        self.delta = 0.01
        self.p_var_bound = 5
        self.gradient_bound = 10

    def train(self, gamma, w_opt, max_iter=20, is_search=True, newton_tol=1e-10, newton_max_iter=20):
        error_list = list()
        accuracy_list = list()
        w = numpy.random.randn(self.num_class * self.d, 1) * 0.01

        for executor in self.executor_list:
            executor.set_param(gamma, newton_tol, newton_max_iter, is_search, self.eta_list)

        err = self.objective_function(w) - self.objective_function(w_opt)
        error_list.append(err)
        acc = self.accuracy(w)
        accuracy_list.append(acc)

        for t in range(max_iter):
            # print(self.opt_mode)
            # if self.opt_mode == GS_DCA or self.opt_mode == GS_SDR:
            #     opt_solver = SystemOptimizationSolver(self.p, self.m, self.lam, self.delta, self.p_var_bound,
            #                                           self.gradient_bound, self.d, self.s, self.k, self.tau,
            #                                           self.h_mat[t])
            #     opt_res = opt_solver.gibbs_sampling_based_device_selection(self.opt_mode, beta=0.9, rho=0.8)
            #     self.a, self.selected_set = opt_res[0], opt_res[1]
            # elif self.opt_mode == PERFECT_AGGREGATION:
            #     self.a = numpy.ones((self.k, 1))
            #     self.selected_set = range(self.m)
            # elif self.opt_mode == DCA_ONLY:
            #     self.selected_set = range(self.m)
            #     self.a = dca_solver(self.selected_set, self.h_mat[t])
            # elif self.opt_mode == SDR_ONLY:
            #     self.selected_set = range(self.m)
            #     self.a = sdr_solver(self.selected_set, self.h_mat[t])
            # elif self.opt_mode == THRESHOLD:
            #     self.a = numpy.ones((self.k, 1))
            #     self.selected_set = list()
            #     for i in range(self.m):
            #         h_vec = self.h_mat[t, :, i]
            #         if numpy.linalg.norm(h_vec) >= 1 / (2 * self.m):
            #             self.selected_set.append(i)
            #     print(self.selected_set)
            # elif self.opt_mode == GBMA:
            #     self.a = numpy.ones((self.k, 1))
            #     self.selected_set = range(self.m)
            # elif self.opt_mode == DC_FRAMEWORK:
            #     theta = numpy.sqrt(10)
            #     v = sparse_optimization_dca(self.h_mat[t], theta)
            #     self.selected_set, self.a = feasibility_detection_dca(v, self.h_mat[t], theta)

            w = self.update(w, is_search, t)

            err = self.objective_function(w) - self.objective_function(w_opt)
            error_list.append(err)
            acc = self.accuracy(w)
            accuracy_list.append(acc)
            print('iter ' + str(t) + ': error is ' + str(err) + ' , accuracy is ' + str(acc))
            # print(w)

        self.w = w
        return error_list, accuracy_list

    def accuracy(self, w):
        w_vectors = numpy.split(w, self.num_class, axis=0)
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_test, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)
        p = stable_softmax(w_x)
        pred = numpy.argmax(p, axis=1)

        cnt = 0
        tot = self.y_test.shape[0]
        for i in range(tot):
            if pred[i] == self.y_test[i, 0]:
                cnt += 1
        return cnt / tot

    def objective_function(self, w):
        loss = 0
        for executor in self.executor_list:
            loss += executor.get_data_size() * executor.objective_function(w)
        loss /= self.n
        return loss

    def update(self, w, is_search, t):
        # dummy
        return w
