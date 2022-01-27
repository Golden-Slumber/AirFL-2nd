"""
This module is used to calculate the global optimal solution for the logistic regression loss function
"""
import numpy
from scipy import optimize
from Utils.conjugate_gradient_method import conjugate_solver
import sys
from tqdm import tqdm

# import numba

home_dir = '../'
sys.path.append(home_dir)


class LogisticSolver:
    def __init__(self, x_mat=None, y_vec=None):
        if (x_mat is not None) and (y_vec is not None):
            self.n, self.d = x_mat.shape
            self.x_mat = x_mat
            self.y_vec = y_vec

    def fit(self, x_mat, y_vec):
        self.n, self.d = x_mat.shape
        self.x_mat = x_mat
        self.y_vec = y_vec

    def obj_fun(self, w_vec, *args):
        gamma = args[0]

        z_vec = numpy.dot(self.x_mat, w_vec.reshape(self.d, 1))
        z_vec = numpy.multiply(z_vec, self.y_vec)
        l_vec = numpy.log(1 + numpy.exp(-z_vec))

        return numpy.mean(l_vec) + (gamma / 2) * (numpy.linalg.norm(w_vec) ** 2)

    def grad(self, w_vec, *args):
        gamma = args[0]

        z_vec = numpy.dot(self.x_mat, w_vec.reshape(self.d, 1))
        z_vec = numpy.multiply(z_vec, self.y_vec)
        exp_z_vec = numpy.exp(z_vec)
        exp_z_vec = 1 + exp_z_vec
        exp_z_vec = -1 / exp_z_vec
        exp_z_vec = numpy.multiply(exp_z_vec, self.y_vec)
        grad = numpy.dot(self.x_mat.T, exp_z_vec)
        grad = grad / self.n + gamma * w_vec.reshape(self.d, 1)
        return grad

    def exact_newton(self, gamma, max_iter=50, tol=1e-15):
        """
        NO ENOUGH MEMORY
        """
        w_vec = numpy.zeros((self.d, 1))
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)

        for t in range(max_iter):
            grad = self.grad(w_vec, args)
            grad_norm = numpy.linalg.norm(grad)
            print('Logistic Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(grad_norm))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            z_vec = numpy.dot(self.x_mat, w_vec)
            z_vec = numpy.multiply(z_vec, self.y_vec)
            exp_z_vec1 = numpy.add(1, numpy.exp(z_vec))
            exp_z_vec2 = numpy.add(1, numpy.exp(numpy.multiply(-1, z_vec)))
            z_mat = 1 / numpy.dot(exp_z_vec1, exp_z_vec2.T)
            xz_mat = numpy.dot(self.x_mat.T, z_mat)
            xz_mat = numpy.dot(xz_mat, self.x_mat)
            hessian = numpy.add(xz_mat, numpy.multiply(gamma, eye_mat))

            if numpy.linalg.det(hessian) == 0:
                hessian_inv = numpy.linalg.pinv(hessian)
            else:
                hessian_inv = numpy.linalg.inv(hessian)
            p_vec = numpy.dot(hessian_inv, grad)
            obj_val = self.obj_fun(w_vec, *args)

            eta = 0
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * p_vec, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0.5
            w_vec = w_vec - eta * p_vec

        sig = numpy.linalg.svd(hessian, compute_uv=False)
        cond_num = sig[0] / sig[-1]
        print('L: ' + sig[0] + ', u: ' + sig[-1] + ', condition number: ' + cond_num)

        return w_vec, cond_num

    def conjugate_newton(self, gamma, max_iter=50, tol=1e-15):
        w_vec = numpy.zeros((self.d, 1))
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)

        for t in range(max_iter):
            grad = self.grad(w_vec, args)
            grad_norm = numpy.linalg.norm(grad)
            print('Logistic Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(grad_norm))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            z_vec = numpy.dot(self.x_mat, w_vec)
            z_vec = numpy.multiply(z_vec, self.y_vec)
            exp_z_vec = numpy.add(1, numpy.exp(z_vec))
            exp_z_vec = numpy.sqrt(numpy.exp(z_vec)) / exp_z_vec
            a_mat = numpy.multiply(self.x_mat, exp_z_vec)
            p_vec = conjugate_solver(a_mat / numpy.sqrt(self.n), grad, gamma, tol=tol, max_iter=100)

            eta = 0
            obj_val = self.obj_fun(w_vec, *args)
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(w_vec - eta * p_vec, *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0.5
            w_vec = w_vec - eta * p_vec

        hessian = numpy.dot(a_mat.T, a_mat) / self.n + eye_mat
        sig = numpy.linalg.svd(hessian, compute_uv=False)
        cond_num = sig[0] / sig[-1]
        print('L: ' + sig[0] + ', u: ' + sig[-1] + ', condition number: ' + cond_num)

        return w_vec, cond_num

    def conjugate_newton_simplified(self, gamma, max_iter=50, tol=1e-15):
        """
        reduce computation complexity
        """
        w_vec = numpy.zeros((self.d, 1))
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)

        for t in range(max_iter):
            z_vec = numpy.dot(self.x_mat, w_vec)
            z_vec = numpy.multiply(z_vec, self.y_vec)
            exp_z_vec = numpy.exp(z_vec)

            loss = numpy.log(1 + 1 / exp_z_vec)
            obj_val = numpy.mean(loss) + (numpy.linalg.norm(w_vec) ** 2) * gamma / 2

            vec_for_grad = numpy.multiply(-1 / (1 + exp_z_vec), self.y_vec)
            grad = numpy.dot(self.x_mat.T, vec_for_grad) / self.n + gamma * w_vec
            grad_norm = numpy.linalg.norm(grad)
            print('Logistic Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(grad_norm))
            if grad_norm < tol:
                print('The change of objective value is smaller than ' + str(tol))
                break

            vec_for_hessian = numpy.sqrt(exp_z_vec) / (1 + exp_z_vec)
            a_mat = numpy.multiply(self.x_mat, vec_for_hessian)
            p_vec = conjugate_solver(a_mat / numpy.sqrt(self.n), grad, gamma, tol=tol, max_iter=100)

            eta = 0
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(numpy.subtract(w_vec, eta * p_vec), *args)
                    if obj_val_new < obj_val + eta * pg:
                        break
            else:
                eta = 0.5
            w_vec = numpy.subtract(w_vec, eta * p_vec)

        hessian = numpy.dot(a_mat.T, a_mat) / self.n + eye_mat
        sig = numpy.linalg.svd(hessian, compute_uv=False)
        cond_num = sig[0] / sig[-1]
        print('L: ' + str(sig[0]) + ', u: ' + str(sig[-1]) + ', condition number: ' + str(cond_num))

        return w_vec, cond_num

    def centralized_conjugate_newton_simplified(self, gamma, max_iter=50, tol=1e-15):
        w_vec = numpy.zeros((self.d, 1))
        eta_list = 1 / (2 ** numpy.arange(0, 10))
        eye_mat = gamma * numpy.eye(self.d)
        args = (gamma,)
        w_vec_list = list()
        err_list = list()
        acc_list = list()

        w_vec_list.append(w_vec)
        err_list.append(self.obj_fun(w_vec, *args))
        acc_list.append(self.accuracy(w_vec))

        for t in tqdm(range(max_iter)):
            z_vec = numpy.dot(self.x_mat, w_vec)
            z_vec = numpy.multiply(z_vec, self.y_vec)
            exp_z_vec = numpy.exp(z_vec)

            loss = numpy.log(1 + 1 / exp_z_vec)
            obj_val = numpy.mean(loss) + (numpy.linalg.norm(w_vec) ** 2) * gamma / 2

            vec_for_grad = numpy.multiply(-1 / (1 + exp_z_vec), self.y_vec)
            grad = numpy.dot(self.x_mat.T, vec_for_grad) / self.n + gamma * w_vec
            grad_norm = numpy.linalg.norm(grad)
            print('Logistic Solver: Iter ' + str(t) + ', L2 norm of gradient = ' + str(grad_norm))
            # if grad_norm < tol:
            #     print('The change of objective value is smaller than ' + str(tol))
            #     break

            vec_for_hessian = numpy.sqrt(exp_z_vec) / (1 + exp_z_vec)
            a_mat = numpy.multiply(self.x_mat, vec_for_hessian)
            p_vec = conjugate_solver(a_mat / numpy.sqrt(self.n), grad, gamma, tol=tol, max_iter=100)

            eta = 0
            if grad_norm > tol:
                pg = - 0.5 * numpy.sum(numpy.multiply(p_vec, grad))
                for eta in eta_list:
                    obj_val_new = self.obj_fun(numpy.subtract(w_vec, eta * p_vec), *args)
                    if obj_val_new < obj_val + eta * pg:
                        # err_list.append(obj_val_new)
                        break

            else:
                eta = 0.5
                # err_list.append(self.obj_fun(numpy.subtract(w_vec, eta * p_vec), *args))
            w_vec = numpy.subtract(w_vec, eta * p_vec)
            w_vec_list.append(w_vec)
            err_list.append(self.obj_fun(w_vec, *args))
            acc_list.append(self.accuracy(w_vec))

        # print(err_list)
        opt_obj = self.obj_fun(w_vec, *args)
        for t in range(max_iter+1):
            err_list[t] -= opt_obj
        print(err_list)
        print(acc_list)
        # for t in tqdm(range(max_iter)):
        #     err = self.obj_fun(w_vec_list[t], *args) - opt_obj
        #     acc = self.accuracy(w_vec_list[t])
        #     err_list.append(err)
        #     acc_list.append(acc)

        return err_list, acc_list

    def set_test_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def accuracy(self, w):
        num = self.x_test.shape[0]
        count = 0
        idx = 0
        for row in self.x_test:
            if numpy.sign(numpy.dot(row, w.reshape(self.d, 1)))[0] == self.y_test[idx, 0]:
                count += 1
            idx += 1
        return count / num
