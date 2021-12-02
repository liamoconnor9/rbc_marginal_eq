import numpy as np
from mpi4py import MPI
import time
import pathlib
import pickle
import os
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import plot_bot_2d

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (4., 1.)
Prandtl = 1.
Rayleigh = 1e8

# Create bases and domain
Nx, Nz = 128, 128
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
problem.meta['p','b','u','w']['z']['dirichlet'] = True
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.parameters['Lx'] = Lx
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       = -(u*dx(b) + w*bz)")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(b) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

path = os.path.dirname(os.path.abspath(__file__)) + '/RA1E8/results_conv1'
iteration = 9228
iteration_path = path + '/Iteration'+str(iteration)
fields_dict = pickle.load(open(iteration_path + '/fields_dict.pick', 'rb'))
b0z_g = fields_dict[list(fields_dict.keys())[0]]['b0z']
plt.plot(b0z_g)
plt.savefig(path + '/test.png')
import sys
# sys.exit()
# Accomodate differing resolutions in EVP and IVP processes
N_evp = len(b0z_g)
evp_scale = int(N_evp / Nz)
x_evp, z_evp = domain.all_grids(scales=evp_scale)

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=evp_scale)
slices = domain.dist.grid_layout.slices(scales=evp_scale)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
b0z = b0z_g.real[slices[1]]
fields_lst = list(fields_dict.values())

eikx_lst = [np.exp(1j*fields['kx']*x_evp) for fields in fields_lst]
exp_fields = list(zip(eikx_lst, fields_lst))

# Initial conditions
x, z = domain.all_grids()
w = solver.state['w']
wz = solver.state['wz']
b = solver.state['b']
bz = solver.state['bz']
p = solver.state['p']
u = solver.state['u']
uz = solver.state['uz']

w.set_scales(evp_scale)
wz.set_scales(evp_scale)
b.set_scales(evp_scale)
bz.set_scales(evp_scale)
p.set_scales(evp_scale)
u.set_scales(evp_scale)
uz.set_scales(evp_scale)

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-1 * noise * (zt - z_evp) * (z_evp - zb)

b0z_f = domain.new_field()
b0z_f.set_scales(evp_scale)

b0z_f['g'] = b0z
b0z_f.set_scales(1)
b0 = b0z_f.antidifferentiate(z_basis, ('left', 0.5))
b0.set_scales(evp_scale)
# b0z_f.set_scales(evp_scale)
xf, zf = np.meshgrid(x, z)
zff = xf / 4.0 - 0.5
b['g'] = (b0['g'] + np.sqrt(2)*(sum([exp*fields['b'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real)
# b['g'] = ((b0['g']))
bz = b.differentiate('z')
# plot_bot_2d(b)
# plt.show()

p0 = b0.antidifferentiate(z_basis, ('left', 0))
p0.set_scales(evp_scale)
p['g'] = p0['g'] + np.sqrt(2)*(sum([exp*fields['p'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
w['g'] = np.sqrt(2)*(sum([exp*fields['w'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
u['g'] = np.sqrt(2)*(sum([exp*fields['u'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
wz = w.differentiate('z')
uz = u.differentiate('z')

print(np.shape(w['g']))
w.set_scales(1)
wz.set_scales(1)
b.set_scales(1)
bz.set_scales(1)
p.set_scales(1)
u.set_scales(1)
uz.set_scales(1)

# x, z = domain.all_grids()
xf, zf = np.meshgrid(x, z)
uf = np.copy(u['g'])
wf = np.copy(w['g'])
bf = (0.2+(xf * (xf - 4.0))**2) * np.copy(b['g'])
sf = np.sqrt(uf**2 + wf**2)
# sf = 1
from matplotlib.pyplot import figure

figure(figsize=(12, 4), dpi=80)
scale = 1.0
plt.pcolormesh(np.transpose(xf), xf / 4.0 - 0.5, bf, cmap = 'seismic',shading='gouraud')


# Create bases and domain
Nx, Nz = 30, 30
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
problem.meta['p','b','u','w']['z']['dirichlet'] = True
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
problem.parameters['Lx'] = Lx
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) - F*w       = -(u*dx(b) + w*bz)")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(b) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

path = os.path.dirname(os.path.abspath(__file__)) + '/RA1E8/results_conv1'
iteration = 9228
iteration_path = path + '/Iteration'+str(iteration)
fields_dict = pickle.load(open(iteration_path + '/fields_dict.pick', 'rb'))
b0z_g = fields_dict[list(fields_dict.keys())[0]]['b0z']

# Accomodate differing resolutions in EVP and IVP processes
N_evp = len(b0z_g)
evp_scale = int(N_evp / Nz)
x_evp, z_evp = domain.all_grids(scales=evp_scale)

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=evp_scale)
slices = domain.dist.grid_layout.slices(scales=evp_scale)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
b0z = b0z_g.real[slices[1]]
fields_lst = list(fields_dict.values())

eikx_lst = [np.exp(1j*fields['kx']*x_evp) for fields in fields_lst]
exp_fields = list(zip(eikx_lst, fields_lst))

# Initial conditions
x, z = domain.all_grids()
w = solver.state['w']
wz = solver.state['wz']
b = solver.state['b']
bz = solver.state['bz']
p = solver.state['p']
u = solver.state['u']
uz = solver.state['uz']

w.set_scales(evp_scale)
wz.set_scales(evp_scale)
b.set_scales(evp_scale)
bz.set_scales(evp_scale)
p.set_scales(evp_scale)
u.set_scales(evp_scale)
uz.set_scales(evp_scale)

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-1 * noise * (zt - z_evp) * (z_evp - zb)

b0z_f = domain.new_field()
b0z_f.set_scales(evp_scale)

b0z_f['g'] = b0z
b0z_f.set_scales(1)
b0 = b0z_f.antidifferentiate(z_basis, ('left', -0.5))
b0.set_scales(evp_scale)
b['g'] = (np.sqrt(2)*(sum([exp*fields['b'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real + F * pert)
bz = b.differentiate('z')
# plot_bot_2d(b)
# plt.show()

p0 = b0.antidifferentiate(z_basis, ('left', 0))
p0.set_scales(evp_scale)
p['g'] = p0['g'] + np.sqrt(2)*(sum([exp*fields['p'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
w['g'] = np.sqrt(2)*(sum([exp*fields['w'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
u['g'] = np.sqrt(2)*(sum([exp*fields['u'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
wz = w.differentiate('z')
uz = u.differentiate('z')

print(np.shape(w['g']))
w.set_scales(1)
wz.set_scales(1)
b.set_scales(1)
bz.set_scales(1)
p.set_scales(1)
u.set_scales(1)
uz.set_scales(1)

# x, z = domain.all_grids()
xf, zf = np.meshgrid(x, z)
uf = np.copy(u['g'])
wf = np.copy(w['g'])
bf = (xf * (xf - 4.0))**2 * np.copy(b['g'])
sf = np.sqrt(uf**2 + wf**2)
# sf = 1
# from matplotlib.pyplot import figure

# figure(figsize=(8, 4), dpi=80)
scale = 1.0
plt.quiver(np.transpose(xf), xf / 4.0 - 0.5, uf * scale, wf * scale, width=0.003, pivot='middle')
# plt.pcolormesh(uf)
# plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.title("Temperature and Velocity: " + r"$\rm{Ra} = 10^8$")
plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/ms_rbc_dfd/Img/flow1.png')