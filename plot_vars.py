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
    kxs.append(kx)
    labels.append(pi_mult_str)

    nrow = 2
    ncol = 2
    sign = 1
    if (np.max(np.abs(T['g'].real)) != np.max(T['g'].real)):
        sign = -1

    # plt.legend()
    C = sign / max(np.abs(T['g'].real))

    plt.subplot(2, 2, 3)
    plt.plot(z, C * T['g'].imag)
    plt.xlim(-0.5, 0.5)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Im[\Theta]$', rotation=90)

    plt.subplot(2, 2, 1)
    plt.plot(z, C * T['g'].real, label = pi_mult_str)
    plt.xlim(-0.5, 0.5)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Re[\Theta]$', rotation=90)

handles, labels = plt.gca().get_legend_handles_labels()
# order = [4, 2, 3, 0, 1]
order = [2, 1, 0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'center', frameon = False)


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

    sign = 1
    if (max(np.abs(w['g'].real)) != np.max(w['g'].real)):
        sign = -1

    C = sign / max(np.abs(w['g'].real))
    plt.subplot(nrow, ncol, 2)
    plt.plot(z, C * w['g'].real)
    plt.xlim(-0.5, 0.5)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Re[W]$', rotation=90)

    plt.subplot(nrow, ncol, 4)
    plt.plot(z, C * w['g'].imag)

    plt.xlim(-0.5, 0.5)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\Im[W]$', rotation=90)

plt.suptitle('Eigenfunctions: ' + r'$\rm{Ra} = 4 \times 10^8$')
    
plt.savefig(path + '/publication_materials/grid_vars.pdf')
plt.legend()
    # plt.close()