"""
This module contains DC related optimization algorithms
"""
import cvxpy
import numpy
import cmath


def sample_spherical(dim):
    vec = numpy.random.randn(dim)
    vec /= numpy.linalg.norm(vec)
    return vec


def rand_a(pre_aa):
    # unsatisfied
    dim = numpy.linalg.matrix_rank(pre_aa)
    sup_vec = sample_spherical(dim)
    eig_values, eig_vectors = numpy.linalg.eig(pre_aa)
    print(eig_values)
    eig_val_mat = numpy.eye(dim)
    for i in range(dim):
        eig_val_mat[i][i] = numpy.sqrt(abs(eig_values[i]))
    candidate_a = numpy.dot(numpy.dot(eig_vectors, eig_val_mat), sup_vec)
    print('candidate_a: ' + str(numpy.dot(candidate_a.T, candidate_a)))
    print('trace aa: ' + str(numpy.trace(pre_aa)))


def rand_b(pre_aa):
    candidates = list()
    k, d = pre_aa.shape
    candidate_a = numpy.zeros((k, 1))

    for i in range(k):
        candidate_a[i] = numpy.sqrt(abs(pre_aa[i][i]))
    candidates.append(candidate_a)

    return candidates


def rand_c(pre_aa, num_candidates=100):
    candidates = list()

    k, d = pre_aa.shape
    # dim = numpy.linalg.matrix_rank(pre_aa)
    eig_values, eig_vectors = numpy.linalg.eig(pre_aa)
    dim = eig_values.shape[0]
    eig_val_mat = numpy.eye(dim)
    for i in range(dim):
        eig_val_mat[i][i] = numpy.sqrt(abs(eig_values[i]))
    mean = numpy.zeros(dim)
    cov = numpy.eye(dim)

    for j in range(num_candidates):
        sup_vec = numpy.random.multivariate_normal(mean, cov)
        candidate_a = numpy.dot(numpy.dot(eig_vectors, eig_val_mat), sup_vec)
        candidates.append(candidate_a.reshape(k, 1))

    return candidates


def scale_to_satisfy_constraints(a, h_mat):
    scale = 1
    k, m = h_mat.shape

    for i in range(m):
        obj = numpy.linalg.norm(numpy.dot(a.T, numpy.mat(h_mat[:, i]).T))
        if obj < 1:
            scale = max(scale, 1 / obj)

    return numpy.multiply(scale, a)


def sdr_solver(selected_set, h_mat, cache_a=None):
    k, m = h_mat.shape
    # print('shape of h_mat: ' + str(h_mat.shape))

    if len(selected_set) == 0:
        return numpy.ones((k, 1))

    h_list = list()
    for i in range(m):
        h_vec = numpy.matrix(h_mat[:, i]).T
        h_list.append(numpy.dot(h_vec, h_vec.H))

    pre_aa = numpy.random.rand(k, k)
    pre_aa = numpy.dot(pre_aa, pre_aa.T)
    pre_aa = numpy.add(pre_aa, pre_aa.T)
    if cache_a is not None:
        pre_aa = numpy.dot(cache_a, cache_a.T)

    aa = cvxpy.Variable((k, k), PSD=True)
    constraints = [aa >> 0]
    for i in range(m):
        if i in selected_set:
            constraints = constraints + [cvxpy.real(cvxpy.trace(aa @ h_list[i])) >= 1]
    obj = cvxpy.Minimize(cvxpy.trace(aa))
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()

    if aa.value is not None and numpy.linalg.matrix_rank(aa.value) == 1:
        pre_aa = aa.value
    else:
        if aa.value is not None:
            pre_aa = aa.value
    del aa, constraints, obj, prob

    candidates = rand_c(pre_aa, num_candidates=5)
    for i in range(len(candidates)):
        candidates[i] = scale_to_satisfy_constraints(candidates[i], h_mat)

    best_candidate_a = min(candidates, key=lambda a: numpy.linalg.norm(a))

    # check_feasibility(selected_set, h_mat, best_candidate_a)
    return best_candidate_a

def dca_solver(selected_set, h_mat, cache_a=None, max_iter=100, tol=1e-10, data_info=None):
    # time = 0
    k, m = h_mat.shape

    # print(h_mat)

    if len(selected_set) == 0:
        return numpy.ones((k, 1))

    h_list = list()
    for i in range(m):
        h_vec = numpy.mat(h_mat[:, i]).T
        h_list.append(numpy.dot(h_vec, h_vec.H))
        # print(numpy.diag(h_list[i]))

    pre_aa = numpy.random.rand(k, k)
    pre_aa = numpy.dot(pre_aa, pre_aa.T)
    pre_aa = numpy.add(pre_aa, pre_aa.T)
    if cache_a is not None:
        pre_aa = numpy.dot(cache_a, cache_a.T)

    # warm_start
    aa = cvxpy.Variable((k, k))
    a_sub_gradient = cvxpy.Parameter((k, k))
    constraints = [aa >> 0, cvxpy.trace(aa) >= 0]
    for j in range(m):
        if j in selected_set:
            constraints = constraints + [cvxpy.real(cvxpy.trace(aa @ h_list[j])) >= 1]
    obj = cvxpy.Minimize(2 * cvxpy.trace(aa) - cvxpy.trace(a_sub_gradient.T @ aa))
    prob = cvxpy.Problem(obj, constraints)

    for i in range(max_iter):
        if abs(numpy.trace(pre_aa) - numpy.linalg.norm(pre_aa, ord=2)) < tol:
            break

        u, s, vh = numpy.linalg.svd(pre_aa)
        um = u[:, 0]
        um = numpy.mat(um).T
        a_sub_gradient.value = numpy.real(numpy.dot(um, um.H))

        prob.solve(verbose=False)
        # time += prob.solver_stats.solve_time
        # print(
        #     'Iter ' + str(i) + ': status ' + str(prob.status) + ', optimal value ' + str(
        #         prob.value))
        if prob.status == cvxpy.INFEASIBLE:
            # print(h_mat)
            print('Error occurred with DCA.')
            # exit()
            return numpy.zeros((k, 1))

        if aa.value is not None:
            pre_aa = aa.value

    eig_values, eig_vectors = numpy.linalg.eig(pre_aa)
    idx = eig_values.argmax()
    a = eig_vectors[:, idx]
    # a = numpy.multiply(numpy.sqrt(abs(eig_values[0])), numpy.matrix(a).T)
    a = numpy.multiply(cmath.sqrt(eig_values[idx]), numpy.matrix(a).T).reshape((k,1))
    return dca_scale(a, h_mat, selected_set)


def dca_scale(a, h_mat, selected_set):
    scale = 1
    # k, m = h_mat.shape

    for i in selected_set:
        obj = numpy.linalg.norm(numpy.dot(a.T, numpy.mat(h_mat[:, i]).T))
        if obj < 1:
            scale = max(scale, 1 / obj)

    return numpy.multiply(scale, a)


def sparse_optimization_dca(h_mat, theta, cache_a=None, max_iter=50, tol=1e-10):
    # time = 0
    k, m = h_mat.shape
    h_list = list()
    for i in range(m):
        h_vec = numpy.mat(h_mat[:, i]).T
        h_list.append(numpy.dot(h_vec, h_vec.H))

    pre_aa = numpy.random.rand(k, k)
    pre_aa = numpy.dot(pre_aa, pre_aa.T)
    pre_aa = numpy.add(pre_aa, pre_aa.T)
    if cache_a is not None:
        pre_aa = numpy.dot(cache_a, cache_a.T)
    V = numpy.random.rand(k)

    # warm_start
    aa = cvxpy.Variable((k, k))
    v = cvxpy.Variable(m)
    a_sub_gradient = cvxpy.Parameter((k, k))
    constraints = [aa >> 0, cvxpy.trace(aa) >= 1]
    for j in range(m):
        constraints = constraints + [cvxpy.trace(aa) - theta * cvxpy.real(cvxpy.trace(aa @ h_list[j])) <= v[j]]
    obj = cvxpy.Minimize(cvxpy.norm(v, 1) + cvxpy.trace(aa) - cvxpy.trace(a_sub_gradient.T @ aa))
    prob = cvxpy.Problem(obj, constraints)

    for i in range(max_iter):
        if abs(numpy.trace(pre_aa) - numpy.linalg.norm(pre_aa, ord=2)) < tol:
            break

        u, s, vh = numpy.linalg.svd(pre_aa)
        um = u[:, 0]
        um = numpy.mat(um).T
        a_sub_gradient.value = numpy.real(numpy.dot(um, um.H))

        prob.solve()
        # time += prob.solver_stats.solve_time
        # print(
        #     'Iter ' + str(i) + ': status ' + str(prob.status) + ', optimal value ' + str(
        #         prob.value))

        if aa.value is not None:
            pre_aa = aa.value
        if v.value is not None:
            V = v.value

    return V


def feasibility_detection_dca(v, h_mat, theta, cache_a=None, max_iter=50, tol=1e-10):
    k, m = h_mat.shape
    h_list = list()
    for i in range(m):
        h_vec = numpy.mat(h_mat[:, i]).T
        h_list.append(numpy.dot(h_vec, h_vec.H))

    pre_aa = numpy.random.rand(k, k)
    pre_aa = numpy.dot(pre_aa, pre_aa.T)
    pre_aa = numpy.add(pre_aa, pre_aa.T)
    if cache_a is not None:
        pre_aa = numpy.dot(cache_a, cache_a.T)
    obj_val = 0
    current_v = range(m)

    sorted_v = sorted(range(len(v)), key=lambda idx: v[idx])
    sorted_v = sorted_v[::-1]

    # warm_start
    aa = cvxpy.Variable((k, k))
    V = cvxpy.Parameter(m)
    a_sub_gradient = cvxpy.Parameter((k, k))
    constraints = [aa >> 0, cvxpy.trace(aa) >= 1]
    for j in range(m):
        # constraints = constraints + [cvxpy.real(cvxpy.trace(aa @ h_list[j])) >= v[j]]
        constraints = constraints + [cvxpy.trace(aa) - theta * cvxpy.real(cvxpy.trace(aa @ h_list[j])) <= 0]
    obj = cvxpy.Minimize(2 * cvxpy.trace(aa) - cvxpy.trace(a_sub_gradient.T @ aa))
    prob = cvxpy.Problem(obj, constraints)

    left = 0
    right = len(sorted_v) - 1
    mid = 0
    while left <= right:
        mid = (left + right) // 2
        current_device_set = sorted_v[0:mid]
        current_v = numpy.zeros(m)
        for i in range(m):
            if i in current_device_set:
                current_v[i] = 1
            else:
                current_v[i] = -1e6

        for i in range(max_iter):
            if abs(numpy.trace(pre_aa) - numpy.linalg.norm(pre_aa, ord=2)) < tol:
                break

            u, s, vh = numpy.linalg.svd(pre_aa)
            um = u[:, 0]
            um = numpy.mat(um).T
            a_sub_gradient.value = numpy.real(numpy.dot(um, um.H))
            V.value = current_v

            prob.solve()
            # time += prob.solver_stats.solve_time
            # print(
            #     'Iter ' + str(i) + ': status ' + str(prob.status) + ', optimal value ' + str(
            #         prob.value))

            if aa.value is not None:
                pre_aa = aa.value
            obj_val = prob.value

        if obj_val > theta:
            right = mid - 1
        elif obj_val < theta:
            left = mid + 1
        else:
            break

    eig_values, eig_vectors = numpy.linalg.eig(pre_aa)
    idx = eig_values.argmax()
    a = eig_vectors[:, idx]
    a = numpy.multiply(cmath.sqrt(eig_values[idx]), numpy.matrix(a).T)

    return sorted_v[0:mid], a


def check_feasibility(selected_set, h_mat, a):
    aa = numpy.dot(a, a.T)
    k, m = h_mat.shape
    h_list = []
    for i in range(m):
        h_vec = numpy.mat(h_mat[:, i]).T
        h_square = numpy.dot(h_vec, h_vec.H)
        h_list.append(h_square)

    print('rank one constraint: ')
    if numpy.linalg.matrix_rank(aa) == 1:
        print('satisfied')
    else:
        print('unsatisfied')

    print('positive semi-definite constraint: ')
    if numpy.all(numpy.linalg.eigvals(aa)) >= 0:
        print('satisfied')
    else:
        print('unsatisfied')

    print('trace value constraint: ')
    if numpy.trace(aa) > 0:
        print('satisfied')
    else:
        print('unsatisfied')

    print('power constraint: ')
    j = 0
    while j < m:
        print(numpy.linalg.norm(numpy.dot(a.T, numpy.mat(h_mat[:, j]).T)))
        if j in selected_set and numpy.linalg.norm(numpy.dot(a.T, numpy.mat(h_mat[:, j]).T)) < 1:
            print('unsatisfied')
        j += 1


if __name__ == '__main__':
    # algorithm examination

    k = 5
    m = 10
    # cache_a = numpy.random.rand(k, k)
    # cache_aa = numpy.dot(cache_aa, cache_aa.T)
    # cache_aa = numpy.add(cache_aa, cache_aa.T)
    # selected_set = list()
    # for i in range(m):
    #     selected_set.append(i)
    #
    # h_mat = numpy.random.normal(loc=0, scale=1, size=(k, m))
    # cache_aa, time = dca_solver(selected_set, h_mat, cache_aa=cache_aa, max_iter=50)
    # print('Execution time: ' + str(time))

    # cache_aa = numpy.random.rand(k, k)
    # cache_aa = numpy.dot(cache_aa, cache_aa.T)
    # cache_aa = numpy.add(cache_aa, cache_aa.T)
    # cache_aa, time = dca_solver_backup(selected_set, h_mat, cache_aa=cache_aa, max_iter=50)
    # print('Execution time: ' + str(time))

    distance_list = numpy.zeros(m)
    distance_list[0: int(m / 2)] = numpy.random.randint(10, 20, size=int(m / 2))
    distance_list[int(m / 2):] = numpy.random.randint(50, 60, size=int(m / 2))
    perm = numpy.random.permutation(m)
    distance_list = distance_list[perm]
    print(distance_list)
    data_size_list = numpy.zeros(m, dtype=int)
    s = int(numpy.floor(80000 / m))
    data_size_list[0: int(m / 2)] = numpy.random.randint(int(1.8 * s), int(1.9 * s + 1), size=int(m / 2))
    data_size_list[int(m / 2):] = numpy.random.randint(int(0.09 * s), int(0.1 * s + 1), size=int(m / 2))
    print(data_size_list)
    perm = numpy.random.permutation(m)
    data_size_list = data_size_list[perm]
    print(data_size_list)
    print(sum(data_size_list))

    selected_set = range(m)
    # h_mat = numpy.random.normal(loc=0, scale=1, size=(k, m))
    h_mat = numpy.random.randn(k, m) / numpy.sqrt(2) + 1j * numpy.random.randn(k, m) / numpy.sqrt(2)
    # print(h_mat)
    # a = dca_solver(selected_set, h_mat)
    # check_feasibility(selected_set, h_mat, a)
    # print('objective value for dca without Data-CSI design: ' + str(numpy.linalg.norm(a)))

    scaling_factor = 1e3

    for device in range(m):
        PL = (10 ** (-3.35)) * ((distance_list[device] / 1) ** (-3.76))
        h_mat[:, device] = scaling_factor * numpy.sqrt(PL) * h_mat[:, device] / (
                data_size_list[device] / sum(data_size_list))
    print(h_mat)

    a = dca_solver(selected_set, h_mat)

    check_feasibility(selected_set, h_mat, a)
    print('objective value for dca with Data-CSI design: ' + str(numpy.linalg.norm(a)))

    # a = sdr_solver(selected_set, h_mat)
    # check_feasibility(selected_set, h_mat, a)
    # print('objective value for sdr: ' + str(numpy.linalg.norm(a)))
