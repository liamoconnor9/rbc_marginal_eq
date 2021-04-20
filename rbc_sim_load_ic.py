"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib
import pickle
import os

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
Nx, Nz = 2048, 512
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

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():
    
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
    
    b0z_f['g'] = b0z + 1
    b0z_f.set_scales(1)
    b0 = b0z_f.antidifferentiate(z_basis, ('left', 0))
    b0.set_scales(evp_scale)
    b['g'] = b0['g'] + np.sqrt(2)*(sum([exp*fields['b'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real
    # b['g'] = b0['g'] + np.sqrt(2)*(sum([exp*fields['b'][:, None][slices[1]].transpose() for (exp, fields) in exp_fields])).real + F * pert
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
    
    w.set_scales(1)
    wz.set_scales(1)
    b.set_scales(1)
    bz.set_scales(1)
    p.set_scales(1)
    u.set_scales(1)
    uz.set_scales(1)

    # Timestepping and output
    dt = 5e-4
    stop_sim_time = 400
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 500
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time
# solver.stop_iteration = 20

# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots_eq', sim_dt=0.25, max_writes=50, mode=fh_mode)
# snapshots.add_task('b')
# snapshots.add_task('w')
# snapshots.add_task('u')

# profiles = solver.evaluator.add_file_handler('profiles_nonoise', sim_dt=0.1, max_writes=100, mode=fh_mode)
# profiles.add_task("P*integ(dz(b) - 1, 'x') / Lx", name='diffusive_flux')
# profiles.add_task("integ(w*b, 'x') / Lx", name='adv_flux')
# profiles.add_task("integ(integ(w**2 + u**2, 'x'), 'z') / Lx", name='ke')
# profiles.add_task("integ(integ((dx(w) - uz)**2, 'x'), 'z') / Lx", name='enst')

checkpoints = solver.evaluator.add_file_handler('checkpoints_nonoise', sim_dt=0.1, max_writes=50, mode=fh_mode)
checkpoints.add_system(solver.state)
# snapshots.add_task('b')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
