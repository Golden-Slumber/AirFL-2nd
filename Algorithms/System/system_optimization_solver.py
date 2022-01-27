import numpy
from scipy.stats import rv_discrete
from Utils.DCA import dca_solver, sdr_solver, check_feasibility
from constants import GS_DCA, GS_SDR, DCA_ONLY, SDR_ONLY, PERFECT_AGGREGATION


class SystemOptimizationSolver:
    def __init__(self, p, m, lam, delta, p_var_bound, gradient_bound, d, s, k, tau, h_mat, data_size_list,
                 optimization_scaling_factor):
        self.p = p
        self.m = m
        self.lam = lam
        self.delta = delta
        self.p_var_bound = p_var_bound
        self.gradient_bound = gradient_bound
        self.d = d
        self.s = s
        self.k = k
        self.tau = tau
        self.h_mat = h_mat
        self.data_size_list = data_size_list
        self.optimization_scaling_factor = optimization_scaling_factor

    def compute_objective_value(self, a, current_device_set):
        if len(current_device_set) <= 0:
            return 1e6
        elif len(current_device_set) > self.m:
            object_val = (1 / (1 - self.lam)) * (1 + numpy.sqrt(2 * numpy.log(self.m / self.delta))) * numpy.sqrt(
                1 / self.s) * self.gradient_bound
        else:
            data_size = 0
            min_data_size = 1e6
            for i in current_device_set:
                data_size += self.data_size_list[i]
                min_data_size = min(min_data_size, self.data_size_list[i])
            first_term = numpy.sqrt(
                3 * self.d * self.tau / self.p) * self.optimization_scaling_factor * numpy.linalg.norm(
                a) / data_size
            # print('objective first term: ' + str(first_term))
            second_term = numpy.sqrt(24 * (1 - (data_size / (self.m * self.s))) ** 2 / min_data_size + 1 / self.s) * (
                    1 / (1 - self.lam)) * (
                                  1 + numpy.sqrt(2 * numpy.log(1 / self.delta))) * self.gradient_bound
            object_val = first_term + second_term
            # print('objective second term: ' + str(second_term))
        return object_val

    def gibbs_sampling_based_device_selection(self, mode, beta=0.8, rho=0.8, max_iter=100):
        device_entry_vector = numpy.ones(self.m)
        cache_a = None
        cache_device_entry_vectors = list()
        cache_beamforming_list = list()
        cache_objective_list = list()

        for i in range(max_iter):
            index_list = range(self.m)
            probability_list = list()
            beamforming_list = list()
            total_objective_val = 0
            scaling_factor = 0

            for j in range(self.m):
                current_device_entry_vector = numpy.copy(device_entry_vector)
                current_device_entry_vector[j] = 1 - current_device_entry_vector[j]
                # current_device_set = [x for x in current_device_entry_vector if x == 1]
                # print('Gibbs Sampling: iter ' + str(i) + ' neighbor ' + str(j))
                # print('current device entry vector: ' + str(current_device_entry_vector))
                current_device_set = numpy.where(numpy.array(current_device_entry_vector) == 1)[0].tolist()
                # print(current_device_set)
                # print(current_device_set)
                # current_a = numpy.zeros((self.k, 1))
                current_a = None

                cache_index = 0
                while cache_index < len(cache_device_entry_vectors):
                    if numpy.array_equal(cache_device_entry_vectors[cache_index], current_device_entry_vector):
                        break
                    cache_index += 1

                if cache_index != len(cache_device_entry_vectors):
                    current_a = cache_beamforming_list[cache_index]
                    current_objective_val = cache_objective_list[cache_index]
                else:
                    if mode == GS_DCA:
                        current_a = dca_solver(current_device_set, self.h_mat, cache_a=cache_a)
                    elif mode == GS_SDR:
                        current_a = sdr_solver(current_device_set, self.h_mat, cache_a=cache_a)
                    if numpy.linalg.norm(current_a) == 0:
                        current_objective_val = 1e6
                    else:
                        current_objective_val = self.compute_objective_value(current_a, current_device_set)

                    # print(current_a)

                    cache_device_entry_vectors.append(current_device_entry_vector)
                    cache_beamforming_list.append(current_a)
                    cache_objective_list.append(current_objective_val)

                beamforming_list.append(current_a)
                probability_value = 0
                # print(current_objective_val)
                if current_objective_val / beta < 100:
                    probability_value = numpy.exp(-current_objective_val / beta)
                probability_list.append(probability_value)
                # scaling_factor = max(scaling_factor, probability_value)
                total_objective_val += probability_value

            # for j in range(self.m):
            #     probability_list[j] = numpy.exp(-(probability_list[j]/scaling_factor))

            beta = rho * beta
            if total_objective_val == 0:
                continue

            for j in range(self.m):
                probability_list[j] /= total_objective_val

            # print('probability list for neighborhoods: ' + str(probability_list))
            dist = rv_discrete(values=(index_list, probability_list))
            sampled_device_entry = dist.rvs(size=1)
            device_entry_vector[sampled_device_entry[0]] = 1 - device_entry_vector[sampled_device_entry[0]]
            print('Gibbs Sampling Decision iter ' + str(i) + ' :', device_entry_vector)
            cache_a = beamforming_list[sampled_device_entry[0]]

        optimal_device_set = numpy.where(numpy.array(device_entry_vector) == 1)[0].tolist()
        full_device_set = range(self.m)
        optimal_a = numpy.ones((self.k, 1))
        no_selection_a = numpy.copy(optimal_a)
        if mode == GS_DCA:
            optimal_a = dca_solver(optimal_device_set, self.h_mat)
            # check_feasibility(optimal_device_set, self.h_mat, optimal_a)
            no_selection_a = dca_solver(full_device_set, self.h_mat)
            # check_feasibility(full_device_set, self.h_mat, no_selection_a)
        elif mode == GS_SDR:
            optimal_a = sdr_solver(optimal_device_set, self.h_mat)
            no_selection_a = sdr_solver(full_device_set, self.h_mat)

        optimal_objective_val = self.compute_objective_value(optimal_a, optimal_device_set)
        no_selection_objective_val = self.compute_objective_value(no_selection_a, range(self.m))
        if numpy.linalg.norm(optimal_a) == 0:
            print('no feasible solution')
            exit()
        # print('beamforming', optimal_a)
        print('comparison with device selection =>')
        print('with device selection: ' + str(optimal_objective_val))
        print('without device selection: ' + str(no_selection_objective_val))
        if abs(no_selection_objective_val - optimal_objective_val) / no_selection_objective_val > 1:
            # if no_selection_objective_val < optimal_objective_val:
            optimal_a = no_selection_a
            optimal_objective_val = no_selection_objective_val
            optimal_device_set = range(self.m)

        return optimal_a, optimal_device_set, optimal_objective_val, no_selection_objective_val

    def perform_system_optimization(self, mode):
        with_selection_objective = 0
        without_selection_objective = 0

        if mode == GS_DCA or mode == GS_SDR:
            res = self.gibbs_sampling_based_device_selection(mode, beta=100, rho=0.9)
            with_selection_objective, without_selection_objective = res[2], res[3]
        elif mode == PERFECT_AGGREGATION:
            res = self.compute_objective_value(a=None, current_device_set=range(self.m + 1))
            with_selection_objective = without_selection_objective = res

        return with_selection_objective, without_selection_objective

    def perform_system_optimization_wods(self, mode):
        res = 1e6
        if mode == DCA_ONLY:
            a = sdr_solver(range(self.m), self.h_mat)
            res = self.compute_objective_value(a=a, current_device_set=range(self.m))
        elif mode == SDR_ONLY:
            a = dca_solver(range(self.m), self.h_mat)
            res = self.compute_objective_value(a=a, current_device_set=range(self.m))
        return res

if __name__ == '__main__':
    # d = 100
    # sum_a = 0
    # # for i in range(d):
    # #     sum_a += a[i] ** 2
    # # print(sum_a / d)
    # n = 50
    # for i in range(n):
    #     a = numpy.random.normal(loc=0, scale=1, size=(d, 1))
    #     print(numpy.linalg.norm(a) ** 2)
    #     sum_a += numpy.linalg.norm(a) ** 2
    # print(sum_a / n)

    n = 60000
    p = 1
    m = 20
    lam = 0.1
    delta = 0.2
    p_var_bound = 5
    gradient_bound = 5
    d = 10
    s = int(numpy.floor(0.8 * n / m))
    k = 5
    tau = 1e-4
    h_mat = numpy.random.randn(k, m) / numpy.sqrt(
        2) + 1j * numpy.random.randn(k, m) / numpy.sqrt(2)
    optimization_scaling_factor = 1e4
    a = numpy.zeros((k, 1))
    data_size_list = numpy.zeros(m, dtype=int)

    data_size_list[0: int(m / 10)] = numpy.random.randint(int(0.008 * s), int(0.01 * s + 1), size=int(m / 10))
    data_size_list[int(m / 10):] = numpy.random.randint(int(1.01 * s), int(1.11 * s + 1), size=9 * int(m / 10))

    opt_solver = SystemOptimizationSolver(p, m, lam, delta, p_var_bound, gradient_bound, d, s, k, tau, h_mat,
                                          data_size_list,
                                          optimization_scaling_factor)
    set_1 = range(m)
    set_2 = list()
    for i in range(m - 2):
        set_2.append(i + 2)
    print(opt_solver.compute_objective_value(a, set_1))
    print(opt_solver.compute_objective_value(a, set_2))
