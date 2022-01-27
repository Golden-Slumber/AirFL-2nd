import numpy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter

from Resources.data_loader import load_data
from Utils.Logistic import LogisticSolver
from sklearn.model_selection import train_test_split
from Algorithms.Logistic.Solver.ACCADE_logistic_solver import ACCADELogisticSolver
from constants import GS_SDR, GS_DCA, PERFECT_AGGREGATION, color_list, marker_list, DCA_ONLY, SDR_ONLY
import sys

home_dir = '../'
sys.path.append(home_dir)


class ACCADEDemo:
    def __init__(self, data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list, heterogeneity):
        self.data_name = data_name
        self.max_iter = max_iter
        self.repeat = repeat
        self.gamma = gamma
        self.sigma = sigma
        self.p = p
        self.m = m
        self.distance_list = distance_list
        self.data_size_list = data_size_list
        self.n = None
        self.d = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.w_opt = None
        self.cond_num = None
        self.heterogeneity = heterogeneity

    def fit(self, x_mat, y_vec):
        x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.2)
        n, self.d = x_train.shape
        self.n = int(numpy.floor(n / self.m)) * self.m

        self.x_train = numpy.mat(x_train)[0:self.n, :]
        self.x_test = numpy.mat(x_test)
        self.y_train = numpy.mat(y_train)[:, 0:self.n]
        self.y_test = numpy.mat(y_test)
        if self.y_train.shape[0] < self.y_train.shape[1]:
            self.y_train = self.y_train.T
            self.y_test = self.y_test.T

        data_size = sum(self.data_size_list)
        self.x_train = self.x_train[0:data_size, :]
        self.y_train = self.y_train[0:data_size, :]
        print(self.x_train.shape)
        print(self.y_train.shape)
        solver = LogisticSolver(self.x_train, self.y_train)
        self.w_opt, self.cond_num = solver.conjugate_newton_simplified(self.gamma)

    def perform_training(self, tau_list, k_list, modes, is_search=True, newton_iter=100):
        for r in range(self.repeat):
            for i in range(len(k_list)):
                for j in range(len(tau_list)):
                    print('repeat ' + str(r) + ' : k = ' + str(k_list[i]) + ' , tau = ' + str(tau_list[j]))

                    # h_mat = numpy.zeros((self.max_iter, k_list[i], self.m), dtype=numpy.complex128)
                    # for it in range(self.max_iter):
                    #     print(numpy.random.normal(loc=0, scale=self.sigma, size=(k_list[i], self.m)).view(numpy.complex128).shape)
                    #     h_mat[it] = numpy.random.normal(loc=0, scale=self.sigma, size=(k_list[i], 2 * self.m)).view(numpy.complex128)
                    # h_mat[it] = self.sigma / 2 * numpy.random.exponential(self.sigma / 2, (k_list[i], self.m))
                    h_mat = numpy.random.randn(self.max_iter, k_list[i], self.m) / numpy.sqrt(
                        2) + 1j * numpy.random.randn(self.max_iter, k_list[i], self.m) / numpy.sqrt(2)
                    for device in range(self.m):
                        PL = (10 ** 2) * ((self.distance_list[device] / 1) ** (-3.76))
                        h_mat[:, :, device] = numpy.sqrt(PL) * h_mat[:, :, device]

                    for mode in modes:
                        tmp_h = numpy.copy(h_mat)
                        solver = ACCADELogisticSolver(m=self.m, h_mat=tmp_h, tau=tau_list[j], p=self.p,
                                                      x_test=self.x_test, y_test=self.y_test, opt_mode=mode)
                        solver.fit(self.x_train, self.y_train, self.data_size_list)
                        err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter, is_search=is_search,
                                                newton_max_iter=newton_iter)
                        out_file_name = home_dir + 'Outputs/ACCADE_demo/ACCADE_demo_' + self.data_name + '_antenna_' + str(
                            k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(
                            r) + '_' + mode + '_' + self.heterogeneity + '.npz'
                        numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)
                        del solver

    def plot_results_versus_iteration(self, data_name, k, tau, modes, repeat, max_iter, heterogeneity):
        err_mat = numpy.zeros((len(modes), repeat, max_iter))
        acc_mat = numpy.zeros((len(modes), repeat, max_iter))
        for i in range(len(modes)):
            for r in range(repeat):
                file_name = home_dir + 'Outputs/ACCADE_demo/ACCADE_demo_' + data_name + '_antenna_' + str(
                    k) + '_tau_' + str(tau) + '_repeat_' + str(r) + '_' + modes[i] + '_' + heterogeneity + '.npz'
                npz_file = numpy.load(file_name)
                err_mat[i][r] = npz_file['err']
                acc_mat[i][r] = npz_file['acc']

        fig = plt.figure(figsize=(9, 8))
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        line_list = []
        for i in range(len(modes)):
            line, = plt.semilogy(numpy.median(err_mat[i], axis=0), color=color_list[i], linestyle='-',
                                 marker=marker_list[i],
                                 markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
            line_list.append(line)
        plt.legend(line_list, modes, fontsize=20)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Training Loss', fontsize=20)
        plt.xlim(0, max_iter - 1)
        plt.tight_layout()

        image_name = home_dir + 'Outputs/ACCADE_demo/ACCADE_demo_err_' + data_name + '_antenna_' + str(
            k) + '_tau_' + str(tau) + '.pdf'
        fig.savefig(image_name, format='pdf', dpi=1200)
        plt.show()

        fig = plt.figure(figsize=(9, 8))
        line_list = []
        for i in range(len(modes)):
            line, = plt.semilogy(numpy.median(acc_mat[i], axis=0), color=color_list[i], linestyle='-',
                                 marker=marker_list[i],
                                 markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
            line_list.append(line)
        plt.legend(line_list, modes, fontsize=20)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Test Accuracy', fontsize=20)
        plt.xlim(0, max_iter - 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()

        image_name = home_dir + 'Outputs/ACCADE_demo/ACCADE_demo_acc_' + data_name + '_antenna_' + str(
            k) + '_tau_' + str(tau) + '.pdf'
        fig.savefig(image_name, format='pdf', dpi=1200)
        plt.show()


if __name__ == '__main__':
    max_iter = 50
    repeat = 1
    gamma = 1e-8
    sigma = 1
    p = 1
    m = 20
    is_search = True
    newton_iter = 50

    # datasets = ['covtype', 'a9a', 'w8a', 'phishing']
    # # datasets = ['phishing']
    # tau_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # k_list = [5, 10]
    # modes = [GS_DCA]
    # heterogeneity = 'Homogeneous'
    #
    # for data_name in datasets:
    #     x_mat, y_vec = load_data(data_name)
    #
    #     # heterogeneity construction for data size and distance
    #     distance_list = numpy.zeros(m)
    #     distance_list = numpy.random.randint(100, 120, size=m)
    #     # distance_list[0: int(m / 10)] = numpy.random.randint(5, 10, size=int(m / 10))
    #     # distance_list[int(m / 10):] = numpy.random.randint(100, 120, size=9 * int(m / 10))
    #     perm = numpy.random.permutation(m)
    #     distance_list = distance_list[perm]
    #     print(distance_list)
    #     data_size_list = numpy.zeros(m, dtype=int)
    #     s = int(numpy.floor(0.8 * x_mat.shape[0] / m))
    #     data_size_list[0:m] = s
    #     print(data_size_list)
    #     perm = numpy.random.permutation(m)
    #     data_size_list = data_size_list[perm]
    #     print(data_size_list)
    #     print(sum(data_size_list))
    #
    #     demo = ACCADEDemo(data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list, heterogeneity)
    #
    #     demo.fit(x_mat, y_vec)
    #     demo.perform_training(tau_list, k_list, modes, is_search=is_search, newton_iter=newton_iter)
    #
    #     for k in k_list:
    #         for tau in tau_list:
    #             plot_results_versus_iteration(data_name, k, tau, modes, repeat, max_iter + 1, heterogeneity)

    # datasets = ['covtype', 'a9a', 'w8a', 'phishing']
    datasets = ['phishing']
    tau_list = [5e-5]
    k_list = [5]
    # modes = [GS_DCA, DCA_ONLY, PERFECT_AGGREGATION, GS_SDR, SDR_ONLY]
    modes = [PERFECT_AGGREGATION]
    heterogeneity = 'Heterogeneous'

    for data_name in datasets:
        x_mat, y_vec = load_data(data_name)

        # heterogeneity construction for data size and distance
        distance_list = numpy.zeros(m)
        distance_list[0: int(m / 10)] = numpy.random.randint(200, 220, size=int(m / 10))
        distance_list[int(m / 10):] = numpy.random.randint(50, 60, size=9 * int(m / 10))
        # perm = numpy.random.permutation(m)
        # distance_list = distance_list[perm]
        # print(distance_list)
        data_size_list = numpy.zeros(m, dtype=int)
        s = int(numpy.floor(0.8 * x_mat.shape[0] / m))
        data_size_list[0: int(m / 10)] = numpy.random.randint(int(0.008 * s), int(0.01 * s + 1), size=int(m / 10))
        data_size_list[int(m / 10):] = numpy.random.randint(int(1.01 * s), int(1.11 * s + 1), size=9 * int(m / 10))
        print(data_size_list)
        # perm = numpy.random.permutation(m)
        # data_size_list = data_size_list[perm]
        # print(data_size_list)
        # print(sum(data_size_list))

        demo = ACCADEDemo(data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list, heterogeneity)

        demo.fit(x_mat, y_vec)
        demo.perform_training(tau_list, k_list, modes, is_search=is_search, newton_iter=newton_iter)

        for k in k_list:
            for tau in tau_list:
                demo.plot_results_versus_iteration(data_name, k, tau, modes, repeat, max_iter + 1, heterogeneity)
