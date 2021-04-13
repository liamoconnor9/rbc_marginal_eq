import pickle
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
plt.ioff()
import publication_settings

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [3.4, 2.8]})
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
    fig, axs = plt.subplots(3, sharex=True)
    # fig.suptitle(r'Growth Rates: $Ra = 2e9$')
    markers = ['o', 'x', 'x']
    ylims = [(-0.001, 0.0001), (-0.00015, 0.00001), (-0.0003, 0.00002)]
    for i, evs in enumerate(evs_mat):
        ax = axs[i]
        [kxs, EVs] = list(zip(*evs))
        ax.plot(kxs, EVs, '.', linestyle='None')
        ax.set_title(labels[i])
        ax.set_xlim(0, 20*np.pi)
        ax.set_ylim(ylims[i])
        ax.xaxis.set_major_locator(tck.MultipleLocator(5*np.pi))
        ax.xaxis.set_minor_locator(tck.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.grid(False)
        ax.axhline(y=0.0, color='black', alpha=0.2)
        if (i == 1):
            ax.set_ylabel(r'$\omega$')

    plt.tight_layout()
    plt.xlabel(r'$k_x$')
    # plt.title(r'Growth Rates: $Ra = 2e9$')
    # plt.legend(loc='lower right')
    plt.savefig(path + '/pubfigs/EV_spectrum_lapse')

# fig = plt.figure()
iterations = [500, 1000, 2000]
path = os.path.dirname(os.path.abspath(__file__))
iteration_strs_dir = [path + '/RA2E9/results/Iteration' + str(iteration) for iteration in iterations]
labels = ['Iteration ' + str(iteration) for iteration in iterations]
ev_mat = []
for i, iteration_str in enumerate(iteration_strs_dir):
    evs = pickle.load(open(iteration_str + '/evs.pick', 'rb'))
    ev_mat.append(evs)

plot_EVs(ev_mat, labels)


