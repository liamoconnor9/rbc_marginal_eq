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
import publication_settings
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [3.4, 3.4*golden_mean]})
path = os.path.dirname(os.path.abspath(__file__))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ra_vec = [1e5, 1e7, 1e9]
ra_dirs = ['/RA1E5/results_conv/', '/RA1E7/results_conv/', '/RA1E9/results/']
ra_strs = [r'$\rm{Ra} = 10^5$', r'$\rm{Ra} = 10^7$', r'$\rm{Ra} = 10^9$']
# final_iterations = [9664, 8098, 12639]

for i in range(len(ra_vec)):
    f = open(path + ra_dirs[i] + 'rbc_profiles_grid.pick', 'rb')
    b0z = pickle.load(f)[-1].real
    N = len(b0z)
    Nevp = int(1024 / N)
    z_basis0 = de.Chebyshev('z', N, interval=(-1/2, 1/2), dealias=3/2)
    z_basis = de.Chebyshev('z', 1024, interval=(-1/2, 1/2), dealias=3/2)
    domain0 = de.Domain([z_basis0], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
    domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
    z = domain.grid(0)
    z0 = domain0.grid(0)
    b0z_f = domain0.new_field()
    b0z_f['g'] = b0z
    b0_f = b0z_f.antidifferentiate(z_basis0, ('left', 0.5))
    b0_f.set_scales(Nevp)
    b0z_f.set_scales(Nevp)
    b0 = b0_f['g'].copy()
    b0z = b0z_f['g'].copy()

    ind = 0;
    xtra = 2.1
    endind = 0;
    for k in range(1023):
        if (b0z[k] < 0 and b0z[k + 1] > 0 and ind == 0):
            ind = k + 1
            # if (abs(b0z[k]) > abs(b0z[k + 1])):
            #     ind = k + 1
            # else:
            #     ind = k
            print(0.5 + z[ind])

        if (ind != 0 and z[k] + 0.5 > xtra * (z[ind] + 0.5)):
            endind = k;
            break;

            
            

    z_bl = z[:endind] + 0.5
    b0_bl = b0[:endind]
    b0_bl = (b0_bl - b0_bl[ind]) / (1 - 2 * b0_bl[ind])

    if (i == 2):
        plt.plot(z_bl / z_bl[ind], b0_bl, linestyle = 'dashed', linewidth = 0.8, label = ra_strs[i])
        # plt.plot(z_bl / z_bl[ind], b0_bl, linewidth = 2, label = ra_strs[i])
    elif (i == 1):
        # plt.plot(z_bl / z_bl[-1], b0_bl, linestyle = '-.', color = 'black', linewidth = 4, label = ra_strs[i])
        plt.plot(z_bl / z_bl[ind], b0_bl, color = 'black', linewidth = 2.0, label = ra_strs[i])
    else:
        plt.plot(z_bl / z_bl[ind], b0_bl, linewidth = 3.5, color = colors[3], label = ra_strs[i])

plt.xlabel(r'$\frac{z + 0.5}{\delta}$')
plt.ylabel(r'$\overline{T}$')
plt.legend()
plt.xlim(0, xtra - 0.1)
# plt.ylim(0, 0.5)
plt.savefig(path + '/publication_materials/b0_delta_solid.pdf')
