import numpy
import matplotlib
import matplotlib.pyplot as plt
from constants import VERSUS_SNR, VERSUS_ANTENNAS, GS_DCA, GS_SDR, PERFECT_AGGREGATION, SDR_ONLY, DCA_ONLY
from Algorithms.System.system_optimization_solver import SystemOptimizationSolver
from Resources.data_loader import load_data
import sys
from tqdm import tqdm

home_dir = '../'
sys.path.append(home_dir)


class SystemOptimizationDemo:
    def __init__(self, repeat, sigma, p, m, lam, delta, p_var_bound, gradient_bound, d, s, data_size_list,
                 optimization_scaling_factor, distance_list):
        self.repeat = repeat
        self.sigma = sigma
        self.p = p
        self.m = m
        self.lam = lam
        self.delta = delta
        self.p_var_bound = p_var_bound
        self.gradient_bound = gradient_bound
        self.d = d
        self.s = s
        self.data_size_list = data_size_list
        self.optimization_scaling_factor = optimization_scaling_factor
        self.distance_list = distance_list

    def system_optimization_antennas(self, k_list, tau, data_name):
        dca_mat = numpy.zeros((self.repeat, len(k_list)))
        no_sel_dca_mat = numpy.copy(dca_mat)
        sdr_mat = numpy.copy(dca_mat)
        no_sel_sdr_mat = numpy.copy(dca_mat)
        perfect_mat = numpy.copy(dca_mat)

        for r in range(self.repeat):
            for i in tqdm(range(len(k_list))):
                print('repeat ' + str(r) + ' k = ' + str(k_list[i]))

                h_mat = numpy.random.randn(k_list[i], self.m) / numpy.sqrt(
                    2) + 1j * numpy.random.randn(k_list[i], self.m) / numpy.sqrt(2)
                for device in range(self.m):
                    PL = (10 ** (-3)) * ((self.distance_list[device] / 1) ** (-3.76))
                    h_mat[:, device] = numpy.sqrt(PL) * h_mat[:, device]
                h_mat = self.optimization_scaling_factor * h_mat
                optimization_solver = SystemOptimizationSolver(self.p, self.m, self.lam, self.delta, self.p_var_bound,
                                                               self.gradient_bound, self.d, self.s, k_list[i], tau,
                                                               h_mat, data_size_list, optimization_scaling_factor)

                dca_mat[r][i], no_sel_dca_mat[r][i] = optimization_solver.perform_system_optimization(GS_DCA)
                sdr_mat[r][i], no_sel_sdr_mat[r][i] = optimization_solver.perform_system_optimization(GS_SDR)
                perfect_mat[r][i], dummy = optimization_solver.perform_system_optimization(PERFECT_AGGREGATION)
        out_file_name = home_dir + 'Outputs/system_optimization_demo/system_optimization_antennas_' + data_name + '_tau_' + str(
            tau) + '.npz'
        numpy.savez(out_file_name, dca=dca_mat, no_sel_dca=no_sel_dca_mat, sdr=sdr_mat, no_sel_sdr=no_sel_sdr_mat,
                    perfect=perfect_mat, k_list=k_list)

    def system_optimization_snr(self, k, tau_list, snr_list, data_name):
        dca_mat = numpy.zeros((self.repeat, len(tau_list)))
        no_sel_dca_mat = numpy.copy(dca_mat)
        sdr_mat = numpy.copy(dca_mat)
        no_sel_sdr_mat = numpy.copy(dca_mat)
        perfect_mat = numpy.copy(dca_mat)

        for r in range(self.repeat):
            for i in tqdm(range(len(tau_list))):
                print('repeat ' + str(r) + ' tau = ' + str(tau_list[i]))

                h_mat = numpy.random.randn(k, self.m) / numpy.sqrt(
                    2) + 1j * numpy.random.randn(k, self.m) / numpy.sqrt(2)
                for device in range(self.m):
                    PL = (10 ** (-3)) * ((self.distance_list[device] / 1) ** (-3.76))
                    h_mat[:, device] = numpy.sqrt(PL) * h_mat[:, device]
                h_mat = self.optimization_scaling_factor * h_mat
                optimization_solver = SystemOptimizationSolver(self.p, self.m, self.lam, self.delta, self.p_var_bound,
                                                               self.gradient_bound, self.d, self.s, k, tau_list[i],
                                                               h_mat, data_size_list, optimization_scaling_factor)

                dca_mat[r][i], no_sel_dca_mat[r][i] = optimization_solver.perform_system_optimization(GS_DCA)
                sdr_mat[r][i], no_sel_sdr_mat[r][i] = optimization_solver.perform_system_optimization(GS_SDR)
                perfect_mat[r][i], dummy = optimization_solver.perform_system_optimization(PERFECT_AGGREGATION)

        out_file_name = home_dir + 'Outputs/system_optimization_demo/system_optimization_' + data_name + '_snr_k_' + str(
            k) + '.npz'
        numpy.savez(out_file_name, dca=dca_mat, no_sel_dca=no_sel_dca_mat, sdr=sdr_mat, no_sel_sdr=no_sel_sdr_mat,
                    perfect=perfect_mat, snr=snr_list)


def plot_results(file_name, data_name, versus, idx):
    npz_file = numpy.load(file_name)
    dca_mat = npz_file['dca'][:, idx]
    no_sel_dca_mat = npz_file['no_sel_dca'][:, idx]
    sdr_mat = npz_file['sdr'][:, idx]
    no_sel_sdr_mat = npz_file['no_sel_sdr'][:, idx]
    perfect_mat = npz_file['perfect'][:, idx]
    x_ticks = list()
    if versus == VERSUS_SNR:
        x_ticks = npz_file['snr'][idx]
        npz_file = numpy.load('../Outputs/system_optimization_demo/system_optimization_covtype_snr_k_5_SDR_ONLY.npz')
        no_sel_sdr_mat = npz_file['sdr'][:, idx]
        npz_file = numpy.load('../Outputs/system_optimization_demo/system_optimization_covtype_snr_2_k_5_SDR_ONLY.npz')
        # print(no_sel_sdr_mat.shape)
        range_i = [1, 2]
        no_sel_sdr_mat[:, range_i] = npz_file['sdr']
    elif versus == VERSUS_ANTENNAS:
        x_ticks = npz_file['k_list'][idx]
        npz_file = numpy.load(
            '../Outputs/system_optimization_demo/system_optimization_antennas_covtype_tau_1_SDR_ONLY.npz')
        no_sel_sdr_mat = npz_file['sdr'][:, idx]
        npz_file = numpy.load(
            '../Outputs/system_optimization_demo/system_optimization_antennas_covtype_tau_1_antenna_5_SDR_ONLY.npz')
        no_sel_sdr_mat[:, 1] = npz_file['sdr']
        npz_file = numpy.load(
            '../Outputs/system_optimization_demo/system_optimization_antennas_covtype_tau_1_wods.npz')
        no_sel_dca_mat = npz_file['dca'][:, idx]

    fig = plt.figure(figsize=(9, 8))
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    line0, = plt.plot(x_ticks, numpy.median(dca_mat, axis=0), color='#0072BD', linestyle='-', marker='o',
                      markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
    line1, = plt.plot(x_ticks, numpy.median(no_sel_dca_mat, axis=0), color='#EDB120', linestyle='-', marker='s',
                      markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
    line2, = plt.plot(x_ticks, numpy.median(sdr_mat, axis=0), color='#7E2F8E', linestyle='-', marker='*',
                      markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
    line3, = plt.plot(x_ticks, numpy.median(no_sel_sdr_mat, axis=0), color='#8B4513', linestyle='-', marker='|',
                      markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)
    line4, = plt.plot(x_ticks, numpy.median(perfect_mat, axis=0), color='#D95319', linestyle='-', marker='d',
                      markerfacecolor='none', ms=7, markeredgewidth=2.5, linewidth=2.5, clip_on=False)

    plt.legend([line0, line1, line2, line3, line4], ['GS+DCA', 'DCA only', 'GS+SDR', 'SDR only', 'Perfect Aggregation'],
               fontsize=20)
    if versus == VERSUS_SNR:
        plt.xlabel('SNR(dB)', fontsize=25)
    elif versus == VERSUS_ANTENNAS:
        plt.xlabel('Number of Antennas', fontsize=25)
    plt.ylabel('Objective Value', fontsize=25)
    plt.xlim(x_ticks[0], x_ticks[-1])
    plt.xticks(x_ticks)
    plt.grid()
    plt.tight_layout()

    image_name = home_dir + 'Outputs/system_optimization_demo/system_optimization_' + data_name + '_' + versus + '.pdf'
    fig.savefig(image_name, format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    repeat = 50
    sigma = 1
    p = 1
    m = 20
    lam = 0.1
    delta = 0.01
    p_var_bound = 5
    gradient_bound = 10

    datasets = ['covtype']
    distance_list = numpy.zeros(m)
    # distance_list[0: int(m / 2)] = numpy.random.randint(100, 120, size=int(m / 2))
    # distance_list[int(m / 2):] = numpy.random.randint(5, 10, size=int(m / 2))
    # distance_list[0:m] = numpy.random.randint(5, 200, size=m)
    distance_list[0: int(m / 10)] = numpy.random.randint(200, 220, size=int(m / 10))
    distance_list[int(m / 10):] = numpy.random.randint(50, 60, size=9 * int(m / 10))

    # tau = 1
    # for data_name in datasets:
    #     X, y = load_data(data_name)
    #     n, d = X.shape
    #     # s = int(numpy.ceil(0.8 * n / m))
    #     data_size_list = numpy.zeros(m, dtype=int)
    #     s = int(numpy.floor(0.8 * n / m))
    #     data_size_list[0: int(m / 10)] = numpy.random.randint(int(0.08 * s), int(0.1 * s + 1), size=int(m / 10))
    #     data_size_list[int(m / 10):] = numpy.random.randint(int(1 * s), int(1.1 * s + 1), size=9 * int(m / 10))
    #     optimization_scaling_factor = 0
    #     for i in range(m):
    #         optimization_scaling_factor = max(optimization_scaling_factor, 1e3 * data_size_list[i])
    #
    #     demo = SystemOptimizationDemo(repeat, sigma, p, m, lam, delta, p_var_bound, gradient_bound, d, s,
    #                                   data_size_list, optimization_scaling_factor, distance_list)
    #
    #     ini = 3
    #     k_list = []
    #     # for j in range(16):
    #     #     k_list.append(ini + j)
    #     demo.system_optimization_antennas(k_list, tau, data_name)
    #     file_name = home_dir + 'Outputs/system_optimization_demo/system_optimization_antennas_' + data_name + '_tau_' + str(
    #         tau) + '.npz'
    #     idx = [0, 2, 4, 6, 8, 10, 12, 14]
    #     plot_results(file_name, data_name, VERSUS_ANTENNAS, idx)

    k = 5
    for data_name in datasets:
        X, y = load_data(data_name)
        n, d = X.shape
        # s = int(numpy.ceil(0.8 * n / m))
        data_size_list = numpy.zeros(m, dtype=int)
        s = int(numpy.floor(0.8 * n / m))
        data_size_list[0: int(m / 10)] = numpy.random.randint(int(0.08 * s), int(0.1 * s + 1), size=int(m / 10))
        data_size_list[int(m / 10):] = numpy.random.randint(int(1 * s), int(1.1 * s + 1), size=9 * int(m / 10))
        optimization_scaling_factor = 0
        for i in range(m):
            optimization_scaling_factor = max(optimization_scaling_factor, 1e3 * data_size_list[i])

        demo = SystemOptimizationDemo(repeat, sigma, p, m, lam, delta, p_var_bound, gradient_bound, d, s,
                                      data_size_list, optimization_scaling_factor, distance_list)

        ini = 0
        tau_list = []
        snr_list = []
        for i in range(6):
            snr = ini + 5 * i
            base = numpy.power(10, snr / 10)
            tau = p / base
            tau_list.append(tau)
            snr_list.append(snr)
        demo.system_optimization_snr(k, tau_list, snr_list, data_name)

        file_name = home_dir + 'Outputs/system_optimization_demo/system_optimization_covtype_snr_k_' + str(
            k) + '.npz'
        plot_results(file_name, data_name, VERSUS_SNR, range(6))
