import numpy
import matplotlib.pyplot as plt
import matplotlib
from Resources.data_loader import load_data
from Experiments.demo import Demo
from Algorithms.Logistic.Solver.FedGD_logistic_solver import FedGDLogisticSolver
from Algorithms.Logistic.Solver.Fedsplit_logistic_solver import FedSplitLogisticSolver
from Algorithms.Logistic.Solver.ACCADE_logistic_solver import ACCADELogisticSolver
from constants import DCA_ONLY, GS_DCA, PERFECT_AGGREGATION, color_list, marker_list, first_order_list, DC_FRAMEWORK, \
    THRESHOLD
from matplotlib.ticker import PercentFormatter
import sys

home_dir = '../'
sys.path.append(home_dir)


class WithFirstOrderDemo(Demo):
    def perform_training(self, tau_list, k_list, is_search=True):
        for r in range(self.repeat):
            for i in range(len(k_list)):
                for j in range(len(tau_list)):
                    print('repeat ' + str(r) + ' : k = ' + str(k_list[i]) + ' , tau = ' + str(tau_list[j]))

                    # h_mat = numpy.zeros((self.max_iter, k_list[i], self.m), dtype=numpy.complex128)
                    # for it in range(self.max_iter):
                    #     # h_mat[it] = numpy.random.normal(loc=0, scale=self.sigma, size=(k_list[i], self.m))
                    #     h_mat[it] = numpy.random.normal(loc=0, scale=self.sigma, size=(k_list[i], 2 * self.m)).view(
                    #         numpy.complex128)

                    h_mat = numpy.random.randn(self.max_iter, k_list[i], self.m) / numpy.sqrt(
                        2) + 1j * numpy.random.randn(self.max_iter, k_list[i], self.m) / numpy.sqrt(2)
                    for device in range(self.m):
                        PL = (10 ** 2) * ((self.distance_list[device] / 1) ** (-3.76))
                        h_mat[:, :, device] = numpy.sqrt(PL) * h_mat[:, :, device]

                    solver = FedSplitLogisticSolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                    x_test=self.x_test, y_test=self.y_test, opt_mode=THRESHOLD)
                    solver.fit(self.x_train, self.y_train, self.data_size_list)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter,
                                            is_search=is_search)
                    out_file_name = home_dir + 'Outputs/with_first_order_demo/with_first_order_demo_FedSplit_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    solver = FedGDLogisticSolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                 x_test=self.x_test, y_test=self.y_test, opt_mode=DC_FRAMEWORK)
                    solver.fit(self.x_train, self.y_train, self.data_size_list)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter,
                                            is_search=is_search)
                    out_file_name = home_dir + 'Outputs/with_first_order_demo/with_first_order_demo_FedGD_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    # solver = ACCADELogisticSolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                    #                              x_test=self.x_test, y_test=self.y_test, opt_mode=GS_DCA)
                    # solver.fit(self.x_train, self.y_train, self.data_size_list)
                    # err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter,
                    #                        is_search=is_search)
                    # out_file_name = home_dir + 'Outputs/with_first_order_demo/with_first_order_demo_ACCADE_' + self.data_name + '_antenna_' + str(
                    #     k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '.npz'
                    # numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    del solver

    def plot_results_versus_iteration(self, data_name, k, tau, methods, repeat, max_iter, legends):
        err_mat = numpy.zeros((len(methods), repeat, max_iter))
        acc_mat = numpy.zeros((len(methods), repeat, max_iter))
        # centralized
        for r in range(repeat):
            file_name = home_dir + 'Outputs/centralized_training_demo/centralized_training_demo_' + data_name + '_repeat_' + str(
                r % 5) + '.npz'
            npz_file = numpy.load(file_name)
            err_mat[0][r] = npz_file['err']
            acc_mat[0][r] = npz_file['acc']

        # ACCADE
        for r in range(repeat):
            file_name = home_dir + 'Outputs/ACCADE_demo/ACCADE_demo_' + data_name + '_antenna_' + str(
                k) + '_tau_' + str(tau) + '_repeat_' + str(r) + '_GS-DCA_Homogeneous.npz'
            npz_file = numpy.load(file_name)
            err_mat[1][r] = npz_file['err']
            acc_mat[1][r] = npz_file['acc']

        for i in range(len(methods)-2):
            for r in range(repeat):
                file_name = home_dir + 'Outputs/with_first_order_demo/with_first_order_demo_' + methods[
                    i+2] + '_' + data_name + '_antenna_' + str(
                    k) + '_tau_' + str(tau) + '_repeat_' + str(r) + '.npz'
                npz_file = numpy.load(file_name)
                err_mat[i+2][r] = npz_file['err']
                acc_mat[i+2][r] = npz_file['acc']

        fig = plt.figure(figsize=(9, 8))
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        line_list = []
        for i in range(len(methods)):
            line, = plt.semilogy(numpy.median(err_mat[i], axis=0), color=color_list[i], linestyle='-',
                                 marker=marker_list[i],
                                 markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
            line_list.append(line)
        plt.legend(line_list, legends, fontsize=20)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Training Loss', fontsize=20)
        plt.xlim(0, max_iter - 1)
        plt.tight_layout()
        plt.grid()

        image_name = home_dir + 'Outputs/with_first_order_demo/with_first_order_demo_err_' + data_name + '_antenna_' + str(
            k) + '_tau_' + str(tau) + '.pdf'
        fig.savefig(image_name, format='pdf', dpi=1200)
        plt.show()

        fig = plt.figure(figsize=(9, 8))
        line_list = []
        for i in range(len(methods)):
            line, = plt.plot(numpy.median(acc_mat[i], axis=0), color=color_list[i], linestyle='-',
                             marker=marker_list[i],
                             markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5)
            line_list.append(line)
        plt.legend(line_list, legends, fontsize=20)
        plt.xlabel('Iterations', fontsize=20)
        plt.ylabel('Test Accuracy', fontsize=20)
        plt.xlim(0, max_iter - 1)
        plt.tight_layout()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.grid()

        image_name = home_dir + 'Outputs/with_first_order_demo/with_first_order_demo_acc_' + data_name + '_antenna_' + str(
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

    datasets = ['covtype', 'a9a', 'w8a', 'phishing']
    # datasets = ['phishing']
    tau_list = [1e-5]
    k_list = [5]
    legends = ['Baseline 0', 'Proposed Algorithm', 'Baseline 1', 'Baseline 2']

    for data_name in datasets:
        x_mat, y_vec = load_data(data_name)

        # heterogeneity construction for data size and distance
        distance_list = numpy.random.randint(100, 120, size=int(m / 2))
        # distance_list[0: int(m / 10)] = numpy.random.randint(5, 10, size=int(m / 10))
        # distance_list[int(m / 10):] = numpy.random.randint(100, 120, size=9 * int(m / 10))
        perm = numpy.random.permutation(m)
        distance_list = distance_list[perm]
        data_size_list = numpy.zeros(m, dtype=int)
        s = int(numpy.floor(0.8 * x_mat.shape[0] / m))
        data_size_list[0:m] = s
        # data_size_list[0: int(m / 10)] = numpy.random.randint(int(0.08 * s), int(0.1 * s + 1), size=int(m / 10))
        # data_size_list[int(m / 10):] = numpy.random.randint(int(1 * s), int(1.1 * s + 1), size=9 * int(m / 10))
        perm = numpy.random.permutation(m)
        data_size_list = data_size_list[perm]

        demo = WithFirstOrderDemo(data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list)

        demo.fit(x_mat, y_vec)
        demo.perform_training(tau_list, k_list, is_search=is_search)
        methods = ['centralized', 'ACCADE', 'FedGD', 'FedSplit']
        for k in k_list:
            for tau in tau_list:
                demo.plot_results_versus_iteration(data_name, k, tau, methods, repeat, max_iter + 1, legends)
