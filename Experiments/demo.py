import numpy

from Resources.data_loader import load_data
from Utils.Logistic import LogisticSolver
from sklearn.model_selection import train_test_split
import sys

home_dir = '../'
sys.path.append(home_dir)


class Demo(object):
    def __init__(self, data_name, max_iter, repeat, gamma, sigma, p, m, distance_list, data_size_list):
        self.data_name = data_name
        self.max_iter = max_iter
        self.repeat = repeat
        self.gamma = gamma
        self.sigma = sigma
        self.p = p
        self.m = m
        self.n = None
        self.d = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.w_opt = None
        self.cond_num = None
        self.distance_list = distance_list
        self.data_size_list = data_size_list

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

        print(self.x_train.shape)
        print(self.y_train.shape)
        solver = LogisticSolver(self.x_train, self.y_train)
        self.w_opt, self.cond_num = solver.conjugate_newton_simplified(self.gamma)
