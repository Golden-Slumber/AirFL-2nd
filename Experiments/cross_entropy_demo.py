import random

import matplotlib
import numpy

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from Resources.data_loader import load_data
from Utils.CrossEntropy import CrossEntropySolver
from Algorithms.CrossEntropy.Solver.ACCADE_cross_entropy_solver import ACCADECrossEntropySolver
from Algorithms.CrossEntropy.Solver.DANE_cross_entropy_solver import DANECrossEntropySolver
from Algorithms.CrossEntropy.Solver.GIANT_cross_entropy_solver import GIANTCrossEntropySolver
from Algorithms.CrossEntropy.Solver.FedGD_cross_entropy_solver import FedGDCrossEntropySolver
from Algorithms.CrossEntropy.Solver.Fedsplit_cross_entropy_solver import FedSplitCrossEntropySolver
from keras.datasets import fashion_mnist
import sys

from constants import color_list, marker_list, GS_DCA, DCA_ONLY, GS_SDR, SDR_ONLY, PERFECT_AGGREGATION, \
    first_order_list, second_order_list, GBMA, THRESHOLD, DC_FRAMEWORK

home_dir = '../'
sys.path.append(home_dir)


class CrossEntropyDemo(object):
    def __init__(self, data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list):
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
        self.num_class = None
        self.shards = None

    def fit(self, x_train, y_train, shards, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.shards = shards
        self.x_test = x_test
        self.y_test = y_test
        self.n, self.d = self.x_train.shape

        print(self.x_train.shape)
        print(self.y_train.shape)
        self.num_class = numpy.max(self.y_train) + 1
        file_name = home_dir + 'Resources/' + self.data_name + '_optimal.npz'
        npz_file = numpy.load(file_name)
        self.w_opt = npz_file['w_opt']
        print(self.w_opt)
        print(self.w_opt.shape)

    def perform_training(self, tau_list, k_list, modes, is_search=True, newton_iter=100):
        for r in range(self.repeat):
            for i in range(len(k_list)):
                for j in range(len(tau_list)):
                    print('repeat ' + str(r) + ' : k = ' + str(k_list[i]) + ' , tau = ' + str(tau_list[j]))

                    h_mat = numpy.random.randn(self.max_iter, k_list[i], self.m) / numpy.sqrt(
                        2) + 1j * numpy.random.randn(self.max_iter, k_list[i], self.m) / numpy.sqrt(2)
                    for device in range(self.m):
                        PL = (10 ** 2) * ((self.distance_list[device] / 1) ** (-3.76))
                        h_mat[:, :, device] = numpy.sqrt(PL) * h_mat[:, :, device]

                    solver = ACCADECrossEntropySolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                      x_test=self.x_test, y_test=self.y_test,
                                                      opt_mode=DCA_ONLY,
                                                      num_class=self.num_class)
                    solver.fit(self.x_train, self.y_train, self.data_size_list, self.shards)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter, is_search=is_search,
                                            newton_max_iter=newton_iter)
                    out_file_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_ACCADE_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '_GS-DCA.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    solver = FedGDCrossEntropySolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                     x_test=self.x_test, y_test=self.y_test, opt_mode=DC_FRAMEWORK,
                                                     num_class=self.num_class)
                    solver.fit(self.x_train, self.y_train, self.data_size_list, self.shards)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter, is_search=is_search,
                                            newton_max_iter=newton_iter)
                    out_file_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_FedGD_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '_DC_FRAMEWORK.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    solver = FedSplitCrossEntropySolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                        x_test=self.x_test, y_test=self.y_test, opt_mode=THRESHOLD,
                                                        num_class=self.num_class)
                    solver.fit(self.x_train, self.y_train, self.data_size_list, self.shards)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter, is_search=is_search,
                                            newton_max_iter=newton_iter)
                    out_file_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_FedSplit_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '_THRESHOLD.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    solver = DANECrossEntropySolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                    x_test=self.x_test, y_test=self.y_test, opt_mode=DCA_ONLY,
                                                    num_class=self.num_class)
                    solver.fit(self.x_train, self.y_train, self.data_size_list, self.shards)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter, is_search=is_search,
                                            newton_max_iter=newton_iter)
                    out_file_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_DANE_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '_DCA only.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)

                    solver = GIANTCrossEntropySolver(m=self.m, h_mat=h_mat, tau=tau_list[j], p=self.p,
                                                     x_test=self.x_test, y_test=self.y_test, opt_mode=DCA_ONLY,
                                                     num_class=self.num_class)
                    solver.fit(self.x_train, self.y_train, self.data_size_list, self.shards)
                    err, acc = solver.train(self.gamma, self.w_opt, max_iter=self.max_iter, is_search=is_search,
                                            newton_max_iter=newton_iter)
                    out_file_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_GIANT_' + self.data_name + '_antenna_' + str(
                        k_list[i]) + '_tau_' + str(tau_list[j]) + '_repeat_' + str(r) + '_DCA only.npz'
                    numpy.savez(out_file_name, err=err, acc=acc, data_name=self.data_name)
                    del solver

    def plot_results_versus_iteration(self, data_name, k, tau, modes, solvers, repeat, max_iter, legends):
        err_mat = numpy.zeros((len(modes) + 1, repeat, max_iter))
        acc_mat = numpy.zeros((len(modes) + 1, repeat, max_iter))
        # centralized
        for r in range(repeat):
            file_name = home_dir + 'Outputs/centralized_training_demo/centralized_training_demo_' + data_name + '_repeat_' + str(
                r) + '.npz'
            npz_file = numpy.load(file_name)
            err_mat[0][r] = npz_file['err']
            acc_mat[0][r] = npz_file['acc']
        for j in range(len(solvers)):
            for r in range(repeat):
                file_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_' + solvers[
                    j] + '_' + data_name + '_antenna_' + str(
                    k) + '_tau_' + str(tau) + '_repeat_' + str(r) + '_' + modes[j] + '.npz'
                npz_file = numpy.load(file_name)
                # print(npz_file['acc'])
                # print(npz_file['err'])
                err_mat[j+1][r] = npz_file['err']
                acc_mat[j+1][r] = npz_file['acc']

        fig = plt.figure(figsize=(9, 8))
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        line_list = []
        for i in range(len(modes)+1):
            line, = plt.semilogy(numpy.median(err_mat[i], axis=0), color=color_list[i], linestyle='-',
                                 marker=marker_list[i],
                                 markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5)
            line_list.append(line)
        plt.legend(line_list, legends, fontsize=20)
        plt.xlabel('Communication Rounds', fontsize=20)
        plt.ylabel('Training Loss', fontsize=20)
        plt.xlim(0, max_iter - 1)
        plt.ylim(0.25, 2.2)
        plt.tight_layout()
        plt.grid()

        image_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_err_' + data_name + '_antenna_' + str(
            k) + '_tau_' + str(tau) + '.pdf'
        fig.savefig(image_name, format='pdf', dpi=1200)
        plt.show()

        fig = plt.figure(figsize=(9, 8))
        line_list = []
        for i in range(len(modes)+1):
            line, = plt.plot(numpy.median(acc_mat[i], axis=0), color=color_list[i], linestyle='-',
                             marker=marker_list[i],
                             markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
            line_list.append(line)
        plt.legend(line_list, legends, fontsize=20)
        plt.xlabel('Communication Rounds', fontsize=20)
        plt.ylabel('Test Accuracy', fontsize=20)
        plt.xlim(0, max_iter - 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        plt.grid()

        image_name = home_dir + 'Outputs/cross_entropy_demo/cross_entropy_demo_acc_' + data_name + '_antenna_' + str(
            k) + '_tau_' + str(tau) + '.pdf'
        fig.savefig(image_name, format='pdf', dpi=1200)
        plt.show()


def normalization(x_train, x_test):
    mean = numpy.mean(x_train)
    std_ev = numpy.sqrt(numpy.var(x_train))
    normalized_x_train = numpy.divide(numpy.subtract(x_train, mean), std_ev)
    mean = numpy.mean(x_test)
    std_ev = numpy.sqrt(numpy.var(x_test))
    normalized_x_test = numpy.divide(numpy.subtract(x_test, mean), std_ev)
    return normalized_x_train, normalized_x_test


if __name__ == '__main__':
    max_iter = 25
    repeat = 5
    gamma = 1e-8
    sigma = 1
    tau = numpy.sqrt(10)
    k = 5
    p = 1
    m = 10
    is_search = True
    newton_iter = 50

    datasets = ['fashion_mnist']
    tau_list = [1e-9]
    k_list = [5]
    # modes = [GS_DCA, PERFECT_AGGREGATION, DC_FRAMEWORK, THRESHOLD, DCA_ONLY, DCA_ONLY]
    # solvers = ['ACCADE', 'ACCADE', 'FedGD', 'FedSplit', 'GIANT', 'DANE']
    # legends = ['Proposed Algorithm', 'Baseline 0', 'Baseline 1', 'Baseline 2', 'Baseline 3', 'Baseline 4']
    modes = [GS_DCA, DC_FRAMEWORK, THRESHOLD, DCA_ONLY, DCA_ONLY]
    solvers = ['ACCADE', 'FedGD', 'FedSplit', 'GIANT', 'DANE']
    legends = ['Baseline 0', 'Proposed Algorithm', 'Baseline 1', 'Baseline 2', 'Baseline 3', 'Baseline 4']

    for data_name in datasets:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        train_n = x_train.shape[0]
        test_n = x_test.shape[0]
        print(x_train.shape)
        print(y_test.shape)

        x_train = x_train.reshape(train_n, 28 * 28)
        idx = numpy.argsort(y_train)
        # idx = numpy.random.permutation(train_n)
        y_train = numpy.array(y_train).reshape(train_n, 1)
        x_test = x_test.reshape(test_n, 28 * 28)
        y_test = numpy.array(y_test).reshape(test_n, 1)

        x_train, x_test = normalization(x_train, x_test)

        # non-iid data distribution construction
        # print(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]
        shard_size = train_n // (6 * m)
        sub_shards = [range(i, i + shard_size) for i in range(0, 6 * shard_size * m, shard_size)]
        shard_ls = random.sample(range(6 * m), k=6 * m)
        # first_shards = [sub_shards[shard_ls[i]] for i in range(0, 2 * m, 2)]
        # second_shards = [sub_shards[shard_ls[i + 1]] for i in range(0, 2 * m, 2)]
        # shards = [list(sub_shards[shard_ls[i]]) + list(sub_shards[shard_ls[i+1]]) for i in range(0, 2 * m, 2)]
        shards = [list(sub_shards[shard_ls[i]]) + list(sub_shards[shard_ls[i + 1]]) + list(
            sub_shards[shard_ls[i + 2]]) + list(sub_shards[shard_ls[i + 3]]) + list(sub_shards[shard_ls[i + 4]]) + list(
            sub_shards[shard_ls[i + 5]]) for i
                  in range(0, 6 * m, 6)]
        # print(shards[0])

        # heterogeneity construction for data size and distance
        distance_list = numpy.random.randint(100, 120, size=m)
        # distance_list[0: int(m / 10)] = numpy.random.randint(5, 10, size=int(m / 10))
        # distance_list[int(m / 10):] = numpy.random.randint(100, 120, size=9 * int(m / 10))
        perm = numpy.random.permutation(m)
        distance_list = distance_list[perm]
        # print(distance_list)
        data_size_list = numpy.zeros(m, dtype=int)
        data_size_list[0:m] = 6 * shard_size
        # data_size_list[0: int(m / 10)] = numpy.random.randint(int(0.08 * s), int(0.1 * s + 1), size=int(m / 10))
        # data_size_list[int(m / 10):] = numpy.random.randint(int(1 * s), int(1.1 * s + 1), size=9 * int(m / 10))
        perm = numpy.random.permutation(m)
        data_size_list = data_size_list[perm]

        demo = CrossEntropyDemo(data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list)
        demo.fit(x_train, y_train, shards, x_test, y_test)
        demo.perform_training(tau_list, k_list, modes, is_search=is_search, newton_iter=newton_iter)

        for k in k_list:
            for tau in tau_list:
                demo.plot_results_versus_iteration(data_name, k, tau, modes, solvers, repeat, max_iter + 1, legends)
