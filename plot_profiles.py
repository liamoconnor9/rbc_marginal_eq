import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.patches import Rectangle
import dedalus.public as de
import numpy as np
from collections import OrderedDict
from mpi4py import MPI
import pandas as pd
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
plt.rcParams.update({'figure.figsize': [3.4, 3.4*golden_mean]})

ra = 1e8
P = ra**(-1/2)
root_path = os.path.dirname(os.path.abspath(__file__))
e = h5py.File(root_path + '/profiles_nom/profiles_nom_s40.h5','r')

path = os.path.dirname(os.path.abspath(__file__)) + '/'
file_nm = path + 'ecs_profiles.xls'

xl = pd.ExcelFile(file_nm)
data = xl.parse("Sheet1")
z_ecs = np.array(list(data.iloc[:, 1])[:])
T_ecs = np.array(list(data.iloc[:, 0])[:])

Tz_nom = np.mean((e['tasks']['diffusive_flux'][()]), axis=0).squeeze() / P
z_nom = e['scales/z']['1.0'][:]
# z_nom = e['scales']['z'][()].squeeze()
f0 = pickle.load(open(root_path + '/RA1E8/results_new/rbc_profiles_grid.pick', 'rb'))
f1 = pickle.load(open(root_path + '/RA1E8/results_conv1/rbc_profiles_grid.pick', 'rb'))
# z2 = e['z'][()].squeeze()
# Tg = e['T'][()].squeeze()
z_basis2 = de.Chebyshev('z', len(Tz_nom), interval=(-1/2, 1/2))
domain2 = de.Domain([z_basis2], grid_dtype=np.float64)
z2 = domain2.grid(0)
Tzf = domain2.new_field()
Tzf['g'] = Tz_nom
Tf2 = Tzf.antidifferentiate(z_basis2, ('left', 0.5))
T2 = Tf2['g']

Tz0 = f0[1]
z_basis0 = de.Chebyshev('z', len(Tz0), interval=(-1/2, 1/2))
domain0 = de.Domain([z_basis0], grid_dtype=np.float64)
z0 = domain0.grid(0)
Tzf0 = domain0.new_field()
Tzf0['g'] = Tz0.real
Tf0 = Tzf0.antidifferentiate(z_basis0, ('left', 0.5))
T0 = Tf0['g']

Tz1 = f1[-5]
z_basis1 = de.Chebyshev('z', len(Tz1), interval=(-1/2, 1/2))
domain1 = de.Domain([z_basis1], grid_dtype=np.float64)
z1 = domain1.grid(0)
Tzf1 = domain1.new_field()
Tzf1['g'] = Tz1.real
Tf1 = Tzf1.antidifferentiate(z_basis1, ('left', 0.5))
T1 = Tf1['g']
# sys.exit()
# z_basis_sim = de.Chebyshev('z', 64*3, interval=(-1/2, 1/2), dealias=3/2)
# domain_sim = de.Domain([z_basis_sim], grid_dtype=np.float64)
fig = plt.figure()
ax = fig.add_subplot()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ax1 = fig.add_axes([0.32, 0.3, 0.35, 0.12])
# ax1 = fig.add_axes([0.29, 0.315, 0.3, 0.18])
ax1 = ax.inset_axes([0.1, 0.1, 0.35, 0.3])
ax.plot(z0, T0, label='Initial', color='black')
ax.plot(z1, T1, label='MSTE', color=colors[1])
ax.plot(z_ecs - 0.5, T_ecs - 0.5, label='ECS', color=colors[-1])
ax.plot(z2, T2, label='DNS', color=colors[0])
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel(r'$\rm{z}$')
ax.set_ylabel(r'$\bar{T}$')
# plt.xticks([-0.5, -0.495, -0.49])
ax.legend(frameon=False, ncol=2)
ax.add_patch(Rectangle((-0.5, -0.02), 0.05, 0.12, alpha=0.5, fill=None, linewidth=0.5))
ax.plot([-0.05, -0.45], [-0.1, 0.1], color='k', alpha=0.5, linewidth=0.5)
ax.plot([-0.4, -0.5], [-0.4, -0.02], color='k', alpha=0.5, linewidth=0.5)
# ax.indicate_inset_zoom(ax1)

ax1.plot(z0, T0, label='Initial', color='black')
ax1.plot(z1, T1, label='MSTE', color=colors[1])
ax1.plot(z_ecs - 0.5, T_ecs - 0.5, label='ECS', color=colors[-1])
ax1.plot(z2, T2, label='DNS', color=colors[0])
ax1.set_xlim(-0.5, -0.45)
ax1.set_ylim(-0.02, 0.1)
ax1.set_xticks([])
ax1.set_yticks([])
# ax1.set_xticks([-0.5, -0.47])
# ax1.set_yticks([-0.03, 0.03])
ax1.xaxis.tick_top()
ax1.yaxis.tick_right()
ax.set_title(r'$\rm{Ra} \, = \, 10^8$')
# plt.legend(loc=3, prop={'size': 8})
# plt.title('$b0_z$ Profiles')
plt.savefig(root_path + '/publication_materials/T_profs_na.pdf', bbox_inches='tight')
# plt.savefig(root_path + '/publication_materials/T_profs_na', bbox_inches='tight')