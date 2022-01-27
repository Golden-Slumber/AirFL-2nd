import numpy
import sys
from Algorithms.CrossEntropy.Executor.cross_entropy_executor import CrossEntropyExecutor
from Utils.CrossEntropy import stable_softmax

home_dir = '../../../'
sys.path.append(home_dir)


class ACCADECrossEntropyExecutor(CrossEntropyExecutor):

    def compute_newton(self):
        eye_mat = self.gamma * numpy.eye(self.d)
        grad = self.compute_gradient()
        w_vectors = numpy.split(self.w, self.num_class, axis=0)
        w_x = []
        for i in range(self.num_class):
            w_x.append(numpy.dot(self.x_mat, w_vectors[i].reshape(self.d, 1)))
        w_x = numpy.concatenate(w_x, axis=1)

        g_vectors = numpy.split(grad, self.num_class, axis=0)
        p_vectors = numpy.zeros((self.num_class, self.d, 1))
        p = stable_softmax(w_x)
        for i in range(self.num_class):
            p_i = numpy.mat(p.T[i]).reshape(self.s, 1)
            p_i = p_i - numpy.power(p_i, 2)
            pxx = numpy.dot(self.x_mat.T, numpy.multiply(self.x_mat, p_i))
            hessian = numpy.add(pxx / self.s, eye_mat)
            if numpy.linalg.det(hessian) == 0:
                hessian_inv = numpy.linalg.pinv(hessian)
            else:
                hessian_inv = numpy.linalg.inv(hessian)
            p_vectors[i] = numpy.dot(hessian_inv, g_vectors[i])
        p_vec = numpy.reshape(p_vectors, (self.num_class * self.d, 1))

        return p_vec
