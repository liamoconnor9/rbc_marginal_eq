import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pickle
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
import dedalus.public as de
logger = logging.getLogger(__name__)
import sys
import os
import publication_settings2
matplotlib.rcParams.update(publication_settings2.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [2*3.4, 2*3.4*golden_mean]})
path = os.path.dirname(os.path.abspath(__file__))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

Prandtl = 1
Rayleigh = 4e8
P = (Rayleigh * Prandtl)**(-1/2)
iteration = 7471
iteration_path = path + '/RA4E8/results_conv/Iteration' + str(iteration)

z_basis = de.Chebyshev('z', 512, interval=(-1/2, 1/2))
d = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
z = d.grid(0)

T = d.new_field()
w = d.new_field()

kxs = []
labels = []
z_ar = dict()
for file in os.listdir(iteration_path + '/data'):
    if (not file.endswith('.pick')):
        continue
    f = open(iteration_path + '/data/' + file, 'rb')
    data = pickle.load(f)
    T['g'] = data['b']
    w['g'] = data['w']

    mod = T['g'].real * T['g'].real + T['g'].imag * T['g'].imag
    nb = np.argmax(mod)
    # print(nb)

    mid_re = T['g'].real[nb]
    mid_im = T['g'].imag[nb]
    r = np.sqrt(mid_re * mid_re + mid_im * mid_im)
    theta = np.arctan(mid_im / mid_re)
    if ((mid_im < 0 and mid_re < 0) or (mid_im < 0 and mid_re > 0)):
        theta += 2 * np.pi

    ZC = np.exp(-1j * theta) / r
    # ZC = 1.0
    kx = data['kx']
    z_ar[kx] = ZC



    p_vars = {
        w : '$w\'$',
        T : '$T\'$',
    }

    pi_mult_str = r'$k_x = $' + str(round(kx/np.pi, 1)) + r'$\pi$'
    kxs.append(kx)
    labels.append(pi_mult_str)

    nrow = 2
    ncol = 2

    nr = len(z) // 2
    plt.subplot(2, 2, 3)
    plt.plot(z[:nr], (T['g'] * ZC).imag[:nr])
    plt.xlim(-0.5, 0.0)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Im[\Theta]$', rotation=90)

    plt.subplot(2, 2, 1)
    plt.plot(z[:nr], (T['g'] * ZC).real[:nr], label = pi_mult_str)
    plt.xlim(-0.5, 0.0)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Re[\Theta]$', rotation=90)

plt.vlines(0.01053 - 0.5, -0.1, 1.1, linestyle = 'dashed', label=r'$z = \delta$', color = 'black')

handles, labels = plt.gca().get_legend_handles_labels()
# order = [4, 2, 3, 0, 1]
order = [2, 1, 0, 3]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper right', frameon = False)


for file in os.listdir(iteration_path + '/data'):
    if (not file.endswith('.pick')):
        continue
    f = open(iteration_path + '/data/' + file, 'rb')
    data = pickle.load(f)
    kx = data['kx']
    T['g'] = data['b']
    w['g'] = data['w']


    p_vars = {
        w : '$w\'$',
        T : '$T\'$',
    }

    pi_mult_str = r'$k_x = $' + str(round(kx/np.pi, 1)) + r'$\pi$'

    nrow = 2
    ncol = 2

    ZC = z_ar[kx]

    nr = len(z) // 2
    plt.subplot(nrow, ncol, 2)
    plt.plot(z[:nr], (w['g'] * ZC).real[:nr])
    plt.xlim(-0.5, 0.0)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Re[W]$', rotation=90)

    plt.subplot(nrow, ncol, 4)
    plt.plot(z[:nr], (w['g'] * ZC).imag[:nr])

    plt.xlim(-0.5, 0.0)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Im[W]$', rotation=90)




plt.suptitle('Eigenfunctions: ' + r'$\rm{Ra} = 4 \times 10^8$')
    
plt.savefig(path + '/publication_materials/grid_vars.pdf')
plt.legend()
    # plt.close()