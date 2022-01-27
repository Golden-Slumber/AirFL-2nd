import numpy
import sys
from Algorithms.CrossEntropy.Executor.cross_entropy_executor import CrossEntropyExecutor
from Utils.CrossEntropy import stable_softmax

home_dir = '../../../'
sys.path.append(home_dir)

eta_list = [0.1, 0.001, 0.0001, 0.000001]


class DANECrossEntropyExecutor(CrossEntropyExecutor):
    def __init__(self, x_mat, y_vec, num_class):
        super().__init__(x_mat, y_vec, num_class)
        self.global_gradient = numpy.zeros((self.d, 1))

    def update_global_gradient(self, g):
        self.global_gradient = g

    def set_w(self, w):
        self.w = w

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

    def compute_batch_gradient(self, x_blk, y_blk, batch_size, w):
        w_vectors = numpy.split(w, self.num_class, axis=0)

        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(x_blk, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        grad = numpy.zeros((self.num_class * self.d, 1))
        for i in range(batch_size):
            x = numpy.mat(x_blk[i]).reshape(self.d, 1)
            p_i = numpy.mat(p[i]).reshape(self.num_class, 1)
            p_i[y_blk[i], 0] = p_i[y_blk[i], 0] - 1
            grad = numpy.add(grad, numpy.kron(p_i, x))
        grad = grad / batch_size
        return grad + numpy.multiply(self.gamma, w)

    def objective_function_for_local_prob(self, h_vec, alpha, mu):
        first_term = self.objective_function(self.w + h_vec)
        second_term = numpy.sum(numpy.dot((self.g - alpha * self.global_gradient).T, h_vec))
        return first_term - second_term + mu / 2 * numpy.linalg.norm(h_vec) ** 2

    def gradient_descent_solver(self, alpha=0.1, rho=0.5, beta=0.5, mu=0.01, max_iter=15):
        """
        solve min f_i(w_t + h) - (g_i(w_t) - alpha * g(w_t)).T * h
        """
        a = alpha
        h_vec = numpy.zeros((self.num_class * self.d, 1))
        for i in range(max_iter):
            grad = self.compute_gradient_for_w(self.w + h_vec) - (
                    self.compute_gradient_for_w(self.w) - alpha * self.global_gradient) + mu * h_vec
            eta = 0.005
            h_vec = numpy.subtract(h_vec, eta * grad)
        return h_vec

    def svrg_solver(self, max_iter=10, alpha=0.0000001):
        batch_size = int(numpy.ceil(self.s / max_iter))
        w = numpy.zeros((self.d, 1))

        for i in range(max_iter):
            w_tilde = numpy.copy(w)

            for j in range(max_iter):
                idx = numpy.random.choice(self.s, batch_size)
                rand_x_blk = self.x_mat[idx, :]
                rand_y_blk = self.y_vec[idx]

                objective_value_old = 1e10
                w_vec = numpy.zeros((self.d, 1))
                for eta in eta_list:
                    full_gradient = self.compute_gradient_for_w(w_tilde) - self.compute_gradient_for_w(
                        self.w) + eta * self.global_gradient + alpha * (w - self.w)
                    batch_gradient = self.compute_batch_gradient(rand_x_blk, rand_y_blk, batch_size,
                                                                 w) - self.compute_batch_gradient(rand_x_blk,
                                                                                                  rand_y_blk,
                                                                                                  batch_size,
                                                                                                  w_tilde) + full_gradient
                    w_new = w - eta * batch_gradient
                    objective_value_new = self.objective_function(w_new)
                    if objective_value_new < objective_value_old:
                        objective_value_old = objective_value_new
                        w_vec = w_new
                w = w_vec

        return w

    def compute_local_statistics_with_info(self, v):
        v_exp = 0
        for i in range(self.num_class * self.d):
            v_exp += v[i]
        v_exp /= self.num_class * self.d

        v_var = 0
        for i in range(self.num_class * self.d):
            v_var += (v[i] - v_exp) ** 2
        v_var /= self.num_class * self.d

        return v_exp, v_var
