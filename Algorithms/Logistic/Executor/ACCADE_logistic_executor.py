import numpy
import sys
from Utils.conjugate_gradient_method import conjugate_solver

home_dir = '../../../'
sys.path.append(home_dir)


class ACCADELogisticExecutor:
    def __init__(self, x_mat, y_vec):
        self.s, self.d = x_mat.shape
        self.x_mat = x_mat
        self.y_vec = y_vec

        self.w = numpy.zeros((self.d, 1))
        self.p = numpy.zeros((self.d, 1))

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

    def update_w(self):
        self.w = numpy.subtract(self.w, self.p)

    def get_p(self):
        return self.p

    def get_w(self):
        return self.w

    def set_w(self, w_vec):
        self.w = w_vec

    def get_data_size(self):
        return self.s

    def objective_function(self, w_vec):
        """
        f_j (w) = log (1 + exp(-w dot x_j)) + (gamma/2) * ||w||_2^2
        return the mean of f_j for all local data x_j
        """
        z_vec = numpy.dot(self.x_mat, w_vec.reshape(self.d, 1))
        z_vec = numpy.multiply(z_vec, self.y_vec)
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
        """
        Compute the gradient of the objective function using local data
        """
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

    def perform_local_update(self, steps):
        w_old = numpy.copy(self.w)
        for i in range(steps):
            grad = self.compute_gradient()
            self.update_p(self.compute_newton())
            objective_value_vec = self.search_objective_val()

            pg = -0.1 * numpy.sum(numpy.multiply(self.p, grad))
            eta = 0
            objective_value_old = objective_value_vec[-1]
            for j in range(self.num_etas):
                objective_value_new = objective_value_vec[j]
                eta = self.eta_list[j]
                if objective_value_new < objective_value_old + pg * eta:
                    break

            self.update_p(numpy.multiply(eta, self.p))
            self.update_w()

        real_p_vec = w_old - self.w
        self.set_w(w_old)
        return real_p_vec

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

    def compute_local_w_statistics(self):
        w_exp = 0
        for i in range(self.d):
            w_exp += self.w[i]
        w_exp /= self.d

        w_var = 0
        for i in range(self.d):
            w_var += (self.p[i] - w_exp) ** 2
        w_var /= self.d

        return w_exp, w_var
