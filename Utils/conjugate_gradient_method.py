"""
This module contains conjugate gradient method and svrg method
"""
import numpy


def conjugate_solver(A, b, lam, tol=1e-16, max_iter=1000):
    """
    conjugate gradient method
    solve (A^T * A + lam * I) * w = b.
    """
    d = A.shape[1]
    b = b.reshape(d, 1)
    tol = tol * numpy.linalg.norm(b)
    w = numpy.zeros((d, 1))
    A_w = numpy.dot(A.T, numpy.dot(A, w))
    r = numpy.subtract(b, numpy.add(lam * w, A_w))
    p = numpy.copy(r)
    rs_old = numpy.linalg.norm(r) ** 2

    for i in range(max_iter):
        A_p = numpy.dot(A.T, numpy.dot(A, p))
        reg_A_p = lam * p + A_p
        alpha = numpy.sum(rs_old / numpy.dot(p.T, reg_A_p))
        w = numpy.add(w, alpha * p)
        r = numpy.subtract(r, alpha * reg_A_p)
        rs_new = numpy.linalg.norm(r) ** 2
        if numpy.sqrt(rs_new) < tol:
            # print('converged! res = ' + str(rs_new))
            break
        p_vec = numpy.multiply(rs_new / rs_old, p)
        p = numpy.add(r, p_vec)
        rs_old = rs_new

    return w


def svrg_solver(A, b, lam, alpha=0.01, Tol=1e-16, MaxIter=1000, BatchSize=100):
    """
    svrg method
    Solve (A^T * A + lam * I) * w = b.
    """
    s, d = A.shape
    b = b.reshape(d, 1)

    # parameter
    scaling = s / BatchSize
    num_inner_loop = int(numpy.ceil(scaling))

    # initialize
    w = numpy.zeros((d, 1))

    for q in range(MaxIter):
        w_tilde = numpy.copy(w)

        # compute full gradient at w_tilde
        aw = numpy.dot(A, w_tilde)
        grad_full = numpy.dot(A.T, aw) - b + lam * w_tilde

        # mini-batch stochastic gradient
        for j in range(num_inner_loop):
            idx = numpy.random.choice(s, BatchSize)
            A_rand = A[idx, :]

            # the stochastic gradient at w
            aw = numpy.dot(A_rand, w)
            grad1 = numpy.dot(A_rand.T, aw) * scaling - b + lam * w

            # the stochastic gradient at w_tilde
            aw = numpy.dot(A_rand, w_tilde)
            grad2 = numpy.dot(A_rand.T, aw) * scaling - b + lam * w_tilde

            grad_rand = grad1 - grad2 + grad_full
            w = numpy.subtract(w, alpha * grad_rand)

    return w
