import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import dedalus.public as de
from collections import OrderedDict
from mpi4py import MPI
import os
import pickle
import h5py
import scipy.sparse.linalg
import scipy.signal
from multiprocessing import Process
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import publication_settings
matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
golden_mean = (np.sqrt(5)-1.0)/2.0
plt.rcParams.update({'figure.figsize': [3.4, 3.4*2*golden_mean]})

def plot_flux(Rayleigh, ax, path, iteration):

    f = open(path + '/convergence_data_Iteration' + str(iteration) + '.pick', 'rb')
    data = pickle.load(f)
    Nz = len(data['b0z'])
    z_basis = de.Chebyshev('z', Nz, interval=(-1/2, 1/2), dealias=3/2)
    domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
    z = domain.grid(0)
    Prandtl = 1
    P = (Rayleigh * Prandtl)**(-1/2)
    if ('kx5' in data.keys()):
        b0z = domain.new_field()
        wb1 = domain.new_field()
        wb2 = domain.new_field()
        b0z = data['b0z']
        wb1 = data['wb1']
        wb2 = data['wb2']
        wb3 = data['wb3']
        wb4 = data['wb4']
        wb5 = data['wb5']
        kx1 = data['kx1']
        kx2 = data['kx2']
        kx3 = data['kx3']
        kx4 = data['kx4']
        kx5 = data['kx5']
        amp1 = data['amp1']
        amp2 = data['amp2']
        amp3 = data['amp3']
        amp4 = data['amp4']
        amp5 = data['amp5']
        diffusion = - P*b0z
        advection1 = wb1*amp1
        advection2 = wb2*amp2
        advection3 = wb3*amp3
        advection4 = wb4*amp4
        advection5 = wb5*amp5
        flux = diffusion + advection1 + advection2 + advection3 + advection4 + advection5
        ax.plot(z, diffusion, label = 'Diffusion', color='black', zorder=10)
        pi_mult1 = str(round(kx1/np.pi, 1))
        pi_mult2 = str(round(kx2/np.pi, 1))
        pi_mult3 = str(round(kx3/np.pi, 1))
        pi_mult4 = str(round(kx4/np.pi, 1))
        pi_mult5 = str(round(kx5/np.pi, 1))
        ax.plot(z, advection1, label = 'Advection ($k_x  = $' + pi_mult1 + '$\pi$)')
        ax.plot(z, advection2, label = 'Advection ($k_x  = $' + pi_mult2 + '$\pi$)')
        ax.plot(z, advection3, label = 'Advection ($k_x  = $' + pi_mult3 + '$\pi$)')
        ax.plot(z, advection4, label = 'Advection ($k_x  = $' + pi_mult4 + '$\pi$)')
        ax.plot(z, advection5, label = 'Advection ($k_x  = $' + pi_mult5 + '$\pi$)')
        ax.plot(z, flux, label = 'Total', linestyle=(0, (5, 1)), color='black')
        ax.set_xlim(-0.5, 0.0)
        ax.set_yticks([0.000, 0.001, 0.002, 0.003])
        # ax.set_ylim(-0.0002, 0.0039)
        ax.legend(prop={'size': 7.5}, frameon=False, loc='center right', bbox_to_anchor=(0.52, 0.25, 0.5, 0.5))
        ax.set_title(r'$\rm{Ra} \, = \, 10^9$')
        # plt.title('Heat Flux')
        ax.set_xlabel(r'$z$')
        ax.set_ylabel('Flux')
        ax.annotate('(C)', xy=(-0.2, 1.1), xycoords='axes fraction', fontsize = plt.rcParams['font.size'] * 1.5)
    
    elif ('kx' in data.keys()):
        b0z = domain.new_field()
        wb1 = domain.new_field()
        wb2 = domain.new_field()
        b0z = data['b0z']
        wb = data['wb']
        kx = data['kx']
        amp = data['amp']
        diffusion = - P*b0z
        advection = wb*amp
        flux = diffusion + advection
        pi_mult = str(round(kx/np.pi, 1))
        ax.plot(z, diffusion, label = 'Diffusion', color='black', zorder=10)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.plot(z, advection, label = 'Advection ($k_x  = $' + pi_mult + '$\pi$)', color=colors[-1])
        ax.plot(z, flux, label = 'Total', linestyle=(0, (5, 1)), color='black')
        ax.set_xlim(-0.5, 0.0)
        ax.legend(frameon=False, loc='center right')
        ax.set_title(r'$\rm{Ra} \, = \, 2 \times  10^5$')
        ax.set_xlabel(r'$z$')
        ax.set_ylabel('Flux')
        # plt.title('Heat Flux (Iteration ' + str(self.iteration) + ')')
        # plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_flux.png')
        # plt.close()
        ax.annotate('(A)', xy=(-0.2, 1.1), xycoords='axes fraction', fontsize = plt.rcParams['font.size'] * 1.5)
    try:
        flux_avg = np.mean(flux)
        flux_const_dev = np.mean(np.abs(flux - flux_avg))
        logger.info('Constant flux deviation: ' + str(flux_const_dev))
        logger.info('Flux at boundary: ' + str(diffusion[0]))
        return flux_const_dev
    except Exception as e:
        logger.warning('Failed to calculate flux deviation with exception: ' + str(e))
        return np.inf

root_path = os.path.dirname(os.path.abspath(__file__))
fig, axs = plt.subplots(2, sharex=False, sharey=False)

ra1 = 2e5;
path1 = root_path + '/RA2E5/results_conv/Iteration13597'
iteration1 = 13597

ra2 = 1e9;
path2 = root_path + '/RA1E9/results/Iteration12638'
iteration2 = 12638

plot_flux(ra1, axs[0], path1, iteration1)
plot_flux(ra2, axs[1], path2, iteration2)
# plt.tight_layout()
plt.subplots_adjust(right=1.0)
# fig.text(0.00, 0.5, 'Flux', ha='center', va='center', rotation='vertical', fontsize=10)
# plt.ylabel('Flux')
# plt.savefig(root_path + '/pubfigs/flux_sup_n1', bbox_inches='tight')
plt.savefig(root_path + '/publication_materials/flux_sup_n.pdf', bbox_inches='tight')


