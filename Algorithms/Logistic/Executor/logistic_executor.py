import numpy
from Utils.conjugate_gradient_method import conjugate_solver
import sys

home_dir = '../../../'
sys.path.append(home_dir)


class LogisticExecutor(object):
    def __init__(self, x_mat, y_vec):
        self.s, self.d = x_mat.shape
        self.x_mat = x_mat
        self.y_vec = y_vec

        self.w = numpy.zeros((self.d, 1))
        self.p = numpy.zeros((self.d, 1))
        self.g = numpy.zeros((self.d, 1))

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
        f_j (w) = log (1 + exp(-w dot x_j)) + (gamma/2) * ||w||_2^2
        return the mean of f_j for all local data x_j
        """
        # print(w_vec)
        z_vec = numpy.dot(self.x_mat, w_vec.reshape(self.d, 1))
        z_vec = numpy.multiply(z_vec, self.y_vec)
        # print(z_vec)
        loss_vec = numpy.log(1 + numpy.exp(-z_vec))
        loss = numpy.mean(loss_vec)
        reg = self.gamma / 2 * (numpy.linalg.norm(w_vec) ** 2)
        return loss + reg

    def search_objective_val(self):
        objective_value_vec = numpy.zeros(self.num_etas + 1)
        for i in range(self.num_etas):
            objective_value_vec[i] = self.objective_function(self.w - self.eta_list[i] * self.p)
        objective_value_vec[-1] = self.objective_function(self.w)
        return objective_value_vec

    def compute_gradient(self):
        z_vec = numpy.dot(self.x_mat, self.w)
        z_vec = numpy.multiply(z_vec, self.y_vec)
        exp_z_vec = numpy.exp(z_vec)
        vec_for_grad = numpy.multiply(-1 / (1 + exp_z_vec), self.y_vec)
        grad_term = numpy.dot(self.x_mat.T, vec_for_grad)
        return grad_term / self.s + self.gamma * self.w

    def compute_newton(self):
        z_vec = numpy.dot(self.x_mat, self.w)
        z_vec = numpy.multiply(z_vec, self.y_vec)
        exp_z_vec = numpy.exp(z_vec)
        vec_for_hessian = numpy.sqrt(exp_z_vec) / (1 + exp_z_vec)
        a_mat = numpy.multiply(self.x_mat, (vec_for_hessian / numpy.sqrt(self.s)))

        p_vec = conjugate_solver(a_mat, self.compute_gradient(), self.gamma, tol=self.g_tol, max_iter=self.max_iter)
        self.g_tol *= 0.5

        return p_vec

    def compute_local_statistics(self):
        p_exp = 0
        for i in range(self.d):
            p_exp += self.p[i]
        p_exp /= self.d

        p_var = 0
        for i in range(self.d):
            p_var += (self.p[i] - p_exp) ** 2
        p_var /= self.d

        return p_exp, p_var
