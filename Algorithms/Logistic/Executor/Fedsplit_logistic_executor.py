import numpy
import sys
from Algorithms.Logistic.Executor.logistic_executor import LogisticExecutor
from Utils.conjugate_gradient_method import conjugate_solver

home_dir = '../../../'
sys.path.append(home_dir)

alpha_list = 1 / (4 ** numpy.arange(0, 10))


class FedSplitLogisticExecutor(LogisticExecutor):
    def __init__(self, x_mat, y_vec, global_w):
        super().__init__(x_mat, y_vec)
        self.global_w = global_w

    def update_global_w(self, global_w):
        self.global_w = global_w

    def compute_gradient_for_w(self, w_vec):
        z_vec = numpy.dot(self.x_mat, w_vec)
        z_vec = numpy.multiply(z_vec, self.y_vec)
        exp_z_vec = numpy.exp(z_vec)
        vec_for_grad = numpy.multiply(-1 / (1 + exp_z_vec), self.y_vec)
        grad_term = numpy.dot(self.x_mat.T, vec_for_grad)
        return grad_term / self.s + self.gamma * w_vec

    def proximal_objective_function(self, u, v):
        """
        s * f(u) + 1/2 ||u - v||^2
        :param u: candidate for zeta^(1/2)
        :param v: 2 * zeta^t - zeta^(t,i)
        :return: objective function for proximal operator
        """
        objective_term = self.objective_function(u)
        proximal_term = numpy.linalg.norm(u - v) ** 2 / 2
        return objective_term + proximal_term

    def approximate_proximal_update(self, e, v, alpha_list, eta):
        """
        compute an approximate zeta^(t+1/2)
        :param e: steps
        :param v: 2 * zeta^t - zeta^(t,i)
        :param alpha_list: step size for approximate
        :param eta: step size for update
        :return:
        """
        if e == 0:
            return v
        u = self.approximate_proximal_update(e - 1, v, alpha_list, eta)
        gradient = self.compute_gradient_for_w(u)

        objective_value_old = self.proximal_objective_function(u, v)
        for alpha in alpha_list:
            objective_value_new = self.proximal_objective_function(u - alpha * eta * gradient + u - v, v)
            if objective_value_new < objective_value_old:
                break
        return u - alpha * eta * gradient + u - v

    def compute_w_proximal(self, steps=6, eta=0.2, rho=0.5, beta=0.3, max_iter=20):
        # 0.1
        # 0.05
        # 0.1
        # 0.32
        eta = 1
        rho = 0.85
        gradient = self.compute_gradient()
        object_value_old = self.objective_function(self.global_w)
        for i in range(max_iter):
            tmp_proximal = self.approximate_proximal_update(steps, 2 * self.global_w - self.w, alpha_list, eta)
            p_vec = 2 * (self.global_w - tmp_proximal)
            objective_value_new = self.objective_function(self.w - p_vec)
            if objective_value_new < object_value_old - beta * numpy.sum(numpy.dot(gradient.T, p_vec)):
                break
            eta *= rho
        # eta = 0.05
        tmp_proximal = self.approximate_proximal_update(steps, 2 * self.global_w - self.w, alpha_list, eta)
        self.w = numpy.add(self.w, 2 * (tmp_proximal - self.global_w))
        return self.w

    def compute_local_statistics(self):
        w_exp = 0
        for i in range(self.d):
            w_exp += self.w[i]
        w_exp /= self.d

        w_var = 0
        for i in range(self.d):
            w_var += (self.w[i] - w_exp) ** 2
        w_var /= self.d

        return w_exp, w_var
