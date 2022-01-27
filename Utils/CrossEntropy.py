"""
This module is used to calculate the global optimal solution for the cross entropy loss function
"""
import time
import numpy
from scipy import optimize
from matplotlib import pyplot as plt
from Utils.conjugate_gradient_method import conjugate_solver
from keras.datasets import fashion_mnist
import sys
import numba
from tqdm import tqdm

home_dir = '../'
sys.path.append(home_dir)


@numba.njit()
def stable_softmax(w_x):
    """
    accelerate the process of computing softmax function
    compute exp(xw) / sum_{i=1}^{C}{exp(xw[i])}
    """
    n, c = w_x.shape
    exps = numpy.exp(w_x - numpy.max(w_x))
    sum_exps = numpy.sum(exps.reshape(n, c, 1), axis=1)
    return numpy.divide(exps, sum_exps.reshape(n, 1))


@numba.njit()
def accelerate_obj(p, s, y_vec):
    """
    accelerate the process of computing objective function
    compute sum_{i=1}^{s}{ -log[ exp(xw[y_i]) / sum_{j=1}^{C}{exp(xw[j])} ] }
    """
    log_likelihood = 0
    for i in range(s):
        log_likelihood = log_likelihood - numpy.log(p[i, y_vec[i, 0]])
    return log_likelihood


@numba.njit()
def accelerate_gradient(p, x_mat, y_vec, s, num_class, d):
    """
    accelerate the process of compute gradient
    compute 1/s sum_{i=1}^{s}{ x_i * [ exp(xw[k]) / sum_{j=1}^{C}{exp(xw[j])} - 1_{y_i == k}] }
    """
    # todo: slower than its original implementation, to be optimized.
    grad = numpy.zeros((num_class * d, 1))
    for i in range(s):
        x = x_mat[i].reshape(d, 1)
        p_i = p[i].reshape(num_class, 1)
        idx = y_vec[i, 0]
        p_i[idx, 0] = p_i[idx, 0] - 1
        grad = numpy.add(grad, numpy.kron(p_i, x).reshape(num_class * d, 1))
    grad = grad / s
    return grad


@numba.njit()
def accelerate_hessian(p, x_mat, g_vectors, eye_mat, s, num_class, d):
    """
    accelerate the process of compute hessian
    compute H^-1 * g
    """
    p_vectors = numpy.zeros((num_class, d, 1))
    for i in range(num_class):
        p_i = p[i].reshape(s, 1)
        p_i = p_i - numpy.power(p_i, 2)
        pxx = numpy.dot(x_mat.T, numpy.multiply(x_mat, p_i))
        hessian = numpy.add(pxx / s, eye_mat)
        if numpy.linalg.det(hessian) == 0:
            hessian_inv = numpy.linalg.pinv(hessian)
        else:
            hessian_inv = numpy.linalg.inv(hessian)
        p_vectors[i] = numpy.dot(hessian_inv, g_vectors[i])
    return p_vectors


class CrossEntropySolver:
    def __init__(self, x_mat=None, y_vec=None, num_class=None, x_test=None, y_test=None):
        if (x_mat is not None) and (y_vec is not None):
            self.n, self.d = x_mat.shape
            self.x_mat = x_mat
            self.y_vec = y_vec
            self.num_class = num_class
            self.x_test = x_test
            self.y_test = y_test
            self.y_mat = numpy.zeros((self.n, self.num_class))
            for i in range(self.n):
                self.y_mat[i, self.y_vec[i, 0]] = 1

    def fit(self, x_mat, y_vec):
        self.n, self.d = x_mat.shape
        self.x_mat = x_mat
        self.y_vec = y_vec

    def softmax(self, w_x):
        """
        unstable version, overflow problem exists
        """
        exps = numpy.exp(w_x)
        return numpy.divide(exps, numpy.mat(numpy.sum(exps, axis=1)).reshape(self.n, 1))

    def predication(self, w_vec):
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_test, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)
        p = stable_softmax(w_x)
        pred = numpy.argmax(p, axis=1)

        cnt = 0
        tot = self.y_test.shape[0]
        for i in range(tot):
            if pred[i] == self.y_test[i, 0]:
                cnt += 1
        return cnt / tot

    def obj_fun(self, w_vec, *args):
        gamma = args[0]
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)

        reg = 0
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            reg += (gamma / 2) * (numpy.linalg.norm(w_vectors[i]) ** 2)
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        # log_likelihood = 0
        # for i in range(self.n):
        #     log_likelihood = log_likelihood - numpy.log(p[i, self.y_vec[i]])
        log_likelihood = accelerate_obj(p, self.n, self.y_vec)

        return log_likelihood / self.n + reg

    def grad(self, w_vec, *args):
        gamma = args[0]
        w_vectors = numpy.split(w_vec, self.num_class, axis=0)

        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)

        p = stable_softmax(w_x)
        grad = numpy.zeros((self.num_class * self.d, 1))
        for i in range(self.n):
            x = numpy.mat(self.x_mat[i]).reshape(self.d, 1)
            p_i = numpy.mat(p[i]).reshape(self.num_class, 1)
            p_i[self.y_vec[i], 0] = p_i[self.y_vec[i], 0] - 1
            grad = numpy.add(grad, numpy.kron(p_i, x))
        grad = grad / self.n
        # grad = accelerate_gradient(p.reshape(self.n, self.num_class, 1), self.x_mat.reshape(self.n, self.d, 1),
        #                            self.y_vec, self.n, self.num_class, self.d)
        return grad + numpy.multiply(gamma, w_vec)

    def gradient_descent(self, gamma, max_iter=50, tol=1e-15):
        """
        gradient descent solver for the minimization of cross entropy loss function
        """
        w_vec = numpy.random.randn(self.num_class * self.d, 1) * 0.01
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        args = (gamma,)

        for t in range(max_iter):
            grad = self.grad(w_vec, args)
            grad_norm = numpy.linalg.norm(grad)
            obj = self.obj_fun(w_vec, *args)
            acc = self.predication(w_vec)
            print('Cross Entropy Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(
                grad_norm) + ', objective value = ' + str(obj) + ', predication = ' + str(acc))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(grad, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * grad, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0.5
            w_vec = w_vec - eta * grad
            print(w_vec)

        return w_vec

    def exact_newton(self, gamma, max_iter=50, tol=1e-15):
        """
        exact newton (compute exact newton update direction vector) solver for the minimization of cross entropy loss function
        """
        w_vec = numpy.random.randn(self.num_class * self.d, 1) * 0.01
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)

        for t in range(max_iter):
            print('calculate grad:')
            grad = self.grad(w_vec, *args)
            grad_norm = numpy.linalg.norm(grad)
            obj = self.obj_fun(w_vec, *args)
            acc = self.predication(w_vec)
            print('Cross Entropy Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(
                grad_norm) + ', objective value = ' + str(obj) + ', predication = ' + str(acc))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            w_vectors = numpy.split(w_vec, self.num_class, axis=0)
            w_x = []
            for i in range(self.num_class):
                w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            w_x = numpy.concatenate(w_x, axis=1)

            g_vectors = numpy.array(numpy.split(grad, self.num_class, axis=0))
            p_vectors = numpy.zeros((self.num_class, self.d, 1))
            p = stable_softmax(w_x)
            for i in range(self.num_class):
                p_i = numpy.mat(p.T[i]).reshape(self.n, 1)
                p_i = p_i - numpy.power(p_i, 2)
                pxx = numpy.dot(self.x_mat.T, numpy.multiply(self.x_mat, p_i))
                hessian = numpy.add(pxx / self.n, eye_mat)
                if numpy.linalg.det(hessian) == 0:
                    hessian_inv = numpy.linalg.pinv(hessian)
                else:
                    hessian_inv = numpy.linalg.inv(hessian)
                p_vectors[i] = numpy.dot(hessian_inv, g_vectors[i])
            # p_vectors = accelerate_hessian(p.reshape(self.num_class, self.n, 1), self.x_mat, g_vectors, eye_mat, self.n,
            #                                self.num_class, self.d)
            p_vec = numpy.reshape(p_vectors, (self.num_class * self.d, 1))

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * p_vec, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0
            print(eta)
            w_vec = w_vec - eta * p_vec

        return w_vec

    def centralized_exact_newton(self, gamma, max_iter=25, tol=1e-15):
        w_vec = numpy.random.randn(self.num_class * self.d, 1) * 0.01
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)
        w_vec_list = list()
        err_list = list()
        acc_list = list()

        w_vec_list.append(w_vec)
        err_list.append(self.obj_fun(w_vec, *args))
        acc_list.append(self.predication(w_vec))

        for t in tqdm(range(max_iter)):
            print('calculate grad:')
            grad = self.grad(w_vec, *args)
            grad_norm = numpy.linalg.norm(grad)
            obj = self.obj_fun(w_vec, *args)
            acc = self.predication(w_vec)
            print('Cross Entropy Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(
                grad_norm) + ', objective value = ' + str(obj) + ', predication = ' + str(acc))
            # if grad_norm < tol:
            #     print('The change of objective value is smaller than ' + str(tol))
            #     break

            w_vectors = numpy.split(w_vec, self.num_class, axis=0)
            w_x = []
            for i in range(self.num_class):
                w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            w_x = numpy.concatenate(w_x, axis=1)

            g_vectors = numpy.array(numpy.split(grad, self.num_class, axis=0))
            p_vectors = numpy.zeros((self.num_class, self.d, 1))
            p = stable_softmax(w_x)
            for i in range(self.num_class):
                p_i = numpy.mat(p.T[i]).reshape(self.n, 1)
                p_i = p_i - numpy.power(p_i, 2)
                pxx = numpy.dot(self.x_mat.T, numpy.multiply(self.x_mat, p_i))
                hessian = numpy.add(pxx / self.n, eye_mat)
                if numpy.linalg.det(hessian) == 0:
                    hessian_inv = numpy.linalg.pinv(hessian)
                else:
                    hessian_inv = numpy.linalg.inv(hessian)
                p_vectors[i] = numpy.dot(hessian_inv, g_vectors[i])
            # p_vectors = accelerate_hessian(p.reshape(self.num_class, self.n, 1), self.x_mat, g_vectors, eye_mat, self.n,
            #                                self.num_class, self.d)
            p_vec = numpy.reshape(p_vectors, (self.num_class * self.d, 1))

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * p_vec, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0
            print(eta)
            w_vec = w_vec - eta * p_vec

            w_vec_list.append(w_vec)
            err_list.append(self.obj_fun(w_vec, *args))
            acc_list.append(self.predication(w_vec))

        opt_obj = self.obj_fun(w_vec, *args)
        for t in range(max_iter + 1):
            err_list[t] -= opt_obj
        print(err_list)
        print(acc_list)

        return err_list, acc_list

    def conjugate_newton(self, gamma, max_iter=50, tol=1e-15):
        """
        newton solver (use conjugate gradient method to compute a approximate direction vector) for the minimization of cross entropy loss function
        """
        w_vec = numpy.random.randn(self.num_class * self.d, 1) * 0.01
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        args = (gamma,)

        for t in range(max_iter):
            grad = self.grad(w_vec, *args)
            grad_norm = numpy.linalg.norm(grad)
            obj = self.obj_fun(w_vec, *args)
            acc = self.predication(w_vec)
            print('Cross Entropy Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(
                grad_norm) + ', objective value = ' + str(obj) + ', predication = ' + str(acc))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            w_vectors = numpy.split(w_vec, self.num_class, axis=0)
            w_x = []
            for i in range(self.num_class):
                w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
            w_x = numpy.concatenate(w_x, axis=1)

            g_vectors = numpy.split(grad, self.num_class, axis=0)
            p_vectors = numpy.zeros((self.num_class, self.d, 1))
            p = stable_softmax(w_x)
            for i in range(self.num_class):
                p_i = numpy.mat(p.T[i]).reshape(self.n, 1)
                p_i = p_i - numpy.power(p_i, 2)
                sqrt_p_i = numpy.sqrt(p_i)
                a_mat = numpy.multiply(sqrt_p_i, self.x_mat) / numpy.sqrt(self.n)
                p_vectors[i] = conjugate_solver(a_mat, g_vectors[i], gamma, tol=tol, max_iter=100)
            p_vec = numpy.reshape(p_vectors, (self.num_class * self.d, 1))

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * p_vec, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0
            print(eta)
            w_vec = w_vec - eta * p_vec

        return w_vec


def normalization(x_train, x_test):
    """
    normalization of data
    """
    mean = numpy.mean(x_train)
    std_ev = numpy.sqrt(numpy.var(x_train))
    normalized_x_train = numpy.divide(numpy.subtract(x_train, mean), std_ev)
    mean = numpy.mean(x_test)
    std_ev = numpy.sqrt(numpy.var(x_test))
    normalized_x_test = numpy.divide(numpy.subtract(x_test, mean), std_ev)
    return normalized_x_train, normalized_x_test


if __name__ == '__main__':
    gamma = 1e-8
    repeat = 5
    data_name = 'fashion_mnist'

    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(x_train[i], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(y_train[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    train_n = x_train.shape[0]
    test_n = x_test.shape[0]

    x_train = x_train.reshape(train_n, 28 * 28)
    y_train = numpy.array(y_train).reshape(train_n, 1)
    x_test = x_test.reshape(test_n, 28 * 28)
    y_test = numpy.array(y_test).reshape(test_n, 1)

    x_train, x_test = normalization(x_train, x_test)

    num_class = numpy.max(y_train) + 1
    solver = CrossEntropySolver(x_train, y_train, num_class, x_test, y_test)
    # w_opt = solver.exact_newton(gamma)
    for r in range(repeat):
        err_list, acc_list = solver.centralized_exact_newton(gamma)
        file_name = home_dir + 'Outputs/centralized_training_demo/centralized_training_demo_' + data_name + '_repeat_' + str(
            r) + '.npz'
        numpy.savez(file_name, err=err_list, acc=acc_list, data_name=data_name)
    # print(w_opt)
