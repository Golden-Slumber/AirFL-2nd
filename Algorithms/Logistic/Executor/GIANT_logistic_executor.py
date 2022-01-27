import numpy
import sys
from Algorithms.Logistic.Executor.logistic_executor import LogisticExecutor
from Utils.conjugate_gradient_method import conjugate_solver

home_dir = '../../../'
sys.path.append(home_dir)


class GIANTLogisticExecutor(LogisticExecutor):
    def __init__(self, x_mat, y_vec):
        super().__init__(x_mat, y_vec)
        self.global_gradient = numpy.zeros((self.d, 1))

    def set_global_gradient(self, g_vec):
        self.global_gradient = g_vec

    def compute_newton(self):
        z_vec = numpy.dot(self.x_mat, self.w)
        z_vec = numpy.multiply(z_vec, self.y_vec)
        exp_z_vec = numpy.exp(z_vec)
        vec_for_hessian = numpy.sqrt(exp_z_vec) / (1 + exp_z_vec)
        a_mat = numpy.multiply(self.x_mat, (vec_for_hessian / numpy.sqrt(self.s)))

        p_vec = conjugate_solver(a_mat, self.global_gradient, self.gamma, tol=self.g_tol, max_iter=self.max_iter)
        self.g_tol *= 0.5

        return p_vec

    def compute_exact_newton(self):
        z_vec = numpy.dot(self.x_mat, self.w)
        z_vec = numpy.multiply(z_vec, self.y_vec)
        exp_z_vec = numpy.exp(z_vec)
        vec_for_hessian = numpy.sqrt(exp_z_vec) / (1 + exp_z_vec)
        a_mat = numpy.multiply(self.x_mat, (vec_for_hessian / numpy.sqrt(self.s)))
        aa = numpy.dot(a_mat.T, a_mat)
        reg_aa = numpy.add(aa, numpy.multiply(self.gamma, numpy.eye(self.d)))
        if numpy.linalg.det(reg_aa) == 0:
            hessian_inv = numpy.linalg.pinv(reg_aa)
        else:
            hessian_inv = numpy.linalg.inv(reg_aa)
        p_vec = numpy.dot(hessian_inv, self.global_gradient)
        return p_vec

    def compute_local_gradient_statistics(self):
        g_exp = 0
        for i in range(self.d):
            g_exp += self.g[i]
        g_exp /= self.d

        g_var = 0
        for i in range(self.d):
            g_var += (self.g[i] - g_exp) ** 2
        g_var /= self.d

        return g_exp, g_var
