import numpy
import sys
from Algorithms.CrossEntropy.Executor.cross_entropy_executor import CrossEntropyExecutor
from Utils.CrossEntropy import stable_softmax

home_dir = '../../../'
sys.path.append(home_dir)

alpha_list = 1 / (4 ** numpy.arange(0, 10))


class FedSplitCrossEntropyExecutor(CrossEntropyExecutor):

    def __init__(self, x_mat, y_vec, num_class, global_w):
        super().__init__(x_mat, y_vec, num_class)
        self.global_w = global_w

    def update_global_w(self, global_w):
        self.global_w = global_w

    def compute_gradient_for_w(self, w_vec):
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)

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
        return grad + numpy.multiply(self.gamma, w_vec)

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

    def compute_w_proximal(self, steps=2, eta=0.001, rho=0.3, beta=0.03, max_iter=10):
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