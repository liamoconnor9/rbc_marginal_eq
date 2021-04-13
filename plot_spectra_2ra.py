import pickle
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import FormatStrFormatter
plt.ioff()
import publication_settings

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [3.4, 3.4*2*golden_mean]})
# plt.gcf().subplots_adjust(left=0.15)

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\frac{\pi}{2}$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"$\frac{%s\pi}{2}$" % str(N)
    else:
        return r"${0}\pi$".format(N // 2)

def plot_EVs(evs_mat, labels):
    fig, axs = plt.subplots(2, sharex=False)
    # fig.suptitle(r'$Ra 2e9$')
    ylims = [(-2, 0.1), (-0.001, 0.0001)]
    xlims = [(0, 12*np.pi), (0, 30*np.pi)]
    maj_loc = [2*np.pi, 5*np.pi]
    for i, evs in enumerate(evs_mat):
        ax = axs[i]
        [kxs, EVs] = list(zip(*evs))
        kx_m, evs_m = [], []
        kx_s, evs_s = [], []
        for j in range(len(kxs)):
            if (EVs[j] > -1e-9):
                kx_m.append(kxs[j])
                evs_m.append(EVs[j])
            else:
                kx_s.append(kxs[j])
                evs_s.append(EVs[j])
        ax.plot(kx_s, evs_s, '.', label='Stable', linestyle='None', color='#08589e')
        ax.plot(kx_m, evs_m, 'x', label='Marginal', markersize=3, linestyle='None', color='black')
        ax.set_title(labels[i])
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])
        ax.legend(loc='lower left')
        ax.xaxis.set_major_locator(tck.MultipleLocator(maj_loc[i]))
        ax.xaxis.set_minor_locator(tck.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.grid(False)
        ax.axhline(y=0.0, color='black', alpha=0.2)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$\omega$')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    plt.tight_layout()
    # plt.title(r'Growth Rates: $Ra = 2e9$')
    # plt.legend(loc='lower right')
    # fig.text(0.00, 0.5, r'$\omega$', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.savefig(path + '/pubfigs/EV_spectra_2ra', bbox_inches='tight')
    plt.savefig(path + '/publication_materials/EV_spectra_2ra', bbox_inches='tight')

# fig = plt.figure()
path = os.path.dirname(os.path.abspath(__file__))
dir1 = path + '/RA2E5/results_conv/Iteration13597'
dir2 = path + '/RA1E9/results/Iteration12638'
iteration_strs_dir = [dir1, dir2]
labels = [r'$Ra \, = \, 2 \times 10^5$', r'$Ra \, = \, 10^9$']
ev_mat = []
for i, iteration_str in enumerate(iteration_strs_dir):
    evs = pickle.load(open(iteration_str + '/evs.pick', 'rb'))
    ev_mat.append(evs)

plot_EVs(ev_mat, labels)


