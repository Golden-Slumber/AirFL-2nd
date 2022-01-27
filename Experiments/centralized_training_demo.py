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

if __name__ == '__main__':
    max_iter = 50
    repeat = 5
    gamma = 1e-8
    sigma = 1
    p = 1
    m = 20
    is_search = True
    newton_iter = 50

    datasets = ['w8a']
    # datasets = ['phishing']
    # tau_list = [1e-5, 1e-7]
    # k_list = [5]
    # modes = [GS_DCA, DCA_ONLY, PERFECT_AGGREGATION, GS_SDR, SDR_ONLY]
    # heterogeneity = 'Heterogeneous'

    for data_name in datasets:
        x_mat, y_vec = load_data(data_name)

        x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.2)
        n, d = x_train.shape
        n = int(numpy.floor(n / m)) * m

        x_train = numpy.mat(x_train)[0:n, :]
        x_test = numpy.mat(x_test)
        y_train = numpy.mat(y_train)[:, 0:n]
        y_test = numpy.mat(y_test)
        if y_train.shape[0] < y_train.shape[1]:
            y_train = y_train.T
            y_test = y_test.T

        for r in range(repeat):
            solver = LogisticSolver(x_train, y_train)
            solver.set_test_data(x_test, y_test)
            err_list, acc_list = solver.centralized_conjugate_newton_simplified(gamma)

            out_file_name = home_dir + 'Outputs/centralized_training_demo/centralized_training_demo_' + data_name + '_repeat_' + str(
                r) + '.npz'
            numpy.savez(out_file_name, err=err_list, acc=acc_list, data_name=data_name)
