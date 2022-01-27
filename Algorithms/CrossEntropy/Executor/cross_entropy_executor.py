import numpy
from Utils.conjugate_gradient_method import conjugate_solver
from Utils.CrossEntropy import stable_softmax, accelerate_obj
import sys
import numba

home_dir = '../../../'
sys.path.append(home_dir)


class CrossEntropyExecutor(object):
    def __init__(self, x_mat, y_vec, num_class):
        self.s, self.d = x_mat.shape
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.num_class = num_class

        self.w = numpy.zeros((self.num_class * self.d, 1))
        self.p = numpy.zeros((self.num_class * self.d, 1))
        self.g = numpy.zeros((self.num_class * self.d, 1))

        self.gamma = None
        self.g_tol = None
        self.max_iter = None
        self.is_search = None
        self.num_etas = None
        self.eta_list = None

    def set_param(self, gamma, g_tol, max_iter, is_search, eta_list):
        self.gamma = gamma
        self.g_tol = g_tol
        self.max_iter = max_iter
        self.is_search = is_search
        if is_search:
            self.eta_list = eta_list
            self.num_etas = len(eta_list)

    def update_p(self, p):
        self.p = p

    def get_p(self):
        return self.p

    def update_w(self):
        self.w = numpy.subtract(self.w, self.p)

    def update_g(self, g):
        self.g = g

    def get_g(self):
        return self.g

    def get_data_size(self):
        return self.s

    def objective_function(self, w_vec):
        """
        """
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)

        reg = 0
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            reg += (self.gamma / 2) * (numpy.linalg.norm(w_vectors[i]) ** 2)
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        # log_likelihood = 0
        # for i in range(self.s):
        #     log_likelihood = log_likelihood - numpy.log(p[i, self.y_vec[i]])
        log_likelihood = accelerate_obj(p, self.s, self.y_vec)

        obj = log_likelihood / self.s + reg
        return obj

    def search_objective_val(self):
        objective_value_vec = numpy.zeros(self.num_etas + 1)
        for i in range(self.num_etas):
            objective_value_vec[i] = self.objective_function(self.w - self.eta_list[i] * self.p)
        objective_value_vec[-1] = self.objective_function(self.w)
        return objective_value_vec

    def compute_gradient(self):
        w_vectors = numpy.split(self.w, self.num_class, axis=0)

        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        grad = numpy.zeros((self.num_class * self.d, 1))
        for i in range(self.s):
            x = numpy.mat(self.x_mat[i]).reshape(self.d, 1)
            p_i = numpy.mat(p[i]).reshape(self.num_class, 1)
            p_i[self.y_vec[i], 0] = p_i[self.y_vec[i], 0] - 1
            grad = numpy.add(grad, numpy.kron(p_i, x))
        grad = grad / self.s
        # grad = accelerate_gradient(grad, p, self.x_mat, self.y_vec, self.s, self.num_class, self.d)
        return grad + numpy.multiply(self.gamma, self.w)

    def compute_local_statistics(self):
        p_exp = 0
        for i in range(self.num_class * self.d):
            p_exp += self.p[i]
        p_exp /= self.num_class * self.d

        p_var = 0
        for i in range(self.num_class * self.d):
            p_var += (self.p[i] - p_exp) ** 2
        p_var /= self.num_class * self.d

        return p_exp, p_var
