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
    iteration = 9226
    iteration_path = path + '/Iteration'+str(iteration)
    conv_data = pickle.load(open(iteration_path + '/convergence_data_Iteration' + str(iteration) + '.pick', 'rb'))
    b0z = conv_data['b0z'].real
    ef_1 = pickle.load(open(iteration_path + '/data/EigenFunctions_grid_iteration' + str(iteration) +'_1p5pi.pick', 'rb'))
    ef_2 = pickle.load(open(iteration_path + '/data/EigenFunctions_grid_iteration' + str(iteration) +'_4p5pi.pick', 'rb'))
    ef_3 = pickle.load(open(iteration_path + '/data/EigenFunctions_grid_iteration' + str(iteration) +'_10p5pi.pick', 'rb'))
    ef_4 = pickle.load(open(iteration_path + '/data/EigenFunctions_grid_iteration' + str(iteration) +'_11p0pi.pick', 'rb'))

    N_evp = len(b0z.real)
    evp_scale = int(N_evp / Nz)
    x_evp, z_evp = domain.all_grids(scales=evp_scale)
    
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=evp_scale)
    slices = domain.dist.grid_layout.slices(scales=evp_scale)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # phi1 = 2*np.pi*np.random.rand()
    # phi2 = 2*np.pi*np.random.rand()
    # phi3 = 2*np.pi*np.random.rand()
    # phi4 = 2*np.pi*np.random.rand()

    phi1 = 0
    phi2 = 0
    phi3 = 0
    phi4 = 0

    logger.info('Phi1: ' + str(phi1))
    logger.info('Phi2: ' + str(phi2))
    logger.info('Phi3: ' + str(phi3))
    logger.info('Phi4: ' + str(phi4))

    # phases = dict()
    # phases['phi1'] = phi1
    # phases['phi2'] = phi2
    # phases['phi3'] = phi3
    # f_phases = open(os.path.dirname(os.path.abspath(__file__)) + '/phases.pick', 'wb')
    # pickle.dump(phases, f_phases)

    eikx1 = np.exp(1.5j*np.pi*x_evp)
    eikx2 = np.exp(4.5j*np.pi*x_evp)
    eikx3 = np.exp(10.5j*np.pi*x_evp)
    eikx4 = np.exp(11.0j*np.pi*x_evp)

    amp1 = np.sqrt(conv_data['amp1']) * np.exp(1j*phi1)
    amp2 = np.sqrt(conv_data['amp2']) * np.exp(1j*phi2)
    amp3 = np.sqrt(conv_data['amp3']) * np.exp(1j*phi3)
    amp4 = np.sqrt(conv_data['amp4']) * np.exp(1j*phi4)

    w1 = ef_1['w'][:, None][slices[1]]
    wz1 = ef_1['wz'][:, None][slices[1]]
    b1 = ef_1['b'][:, None][slices[1]]
    bz1 = ef_1['bz'][:, None][slices[1]]
    p1 = ef_1['p'][:, None][slices[1]]
    u1 = ef_1['u'][:, None][slices[1]]
    uz1 = ef_1['uz'][:, None][slices[1]]
    
    w2 = ef_2['w'][:, None][slices[1]]
    wz2 = ef_2['wz'][:, None][slices[1]]
    b2 = ef_2['b'][:, None][slices[1]]
    bz2 = ef_2['bz'][:, None][slices[1]]
    p2 = ef_2['p'][:, None][slices[1]]
    u2 = ef_2['u'][:, None][slices[1]]
    uz2 = ef_2['uz'][:, None][slices[1]]
    
    w3 = ef_3['w'][:, None][slices[1]]
    wz3 = ef_3['wz'][:, None][slices[1]]
    b3 = ef_3['b'][:, None][slices[1]]
    bz3 = ef_3['bz'][:, None][slices[1]]
    p3 = ef_3['p'][:, None][slices[1]]
    u3 = ef_3['u'][:, None][slices[1]]
    uz3 = ef_3['uz'][:, None][slices[1]]

    w4 = ef_4['w'][:, None][slices[1]]
    wz4 = ef_4['wz'][:, None][slices[1]]
    b4 = ef_4['b'][:, None][slices[1]]
    bz4 = ef_4['bz'][:, None][slices[1]]
    p4 = ef_4['p'][:, None][slices[1]]
    u4 = ef_4['u'][:, None][slices[1]]
    uz4 = ef_4['uz'][:, None][slices[1]]

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
    # logger.warning('len: ' + str(len(b0z)))
    b0z_f.set_scales(evp_scale)
    
    b0z = b0z[slices[1]]
    b0z_f['g'] = b0z + 1
    b0z_f.set_scales(1)
    b0 = b0z_f.antidifferentiate(z_basis, ('left', 0))
    b0.set_scales(evp_scale)
    b['g'] = b0['g'] + np.sqrt(2)*(eikx1*amp1*(b1.transpose()) + eikx2*amp2*(b2.transpose()) + eikx3*amp3*(b3.transpose()).real + eikx4*amp4*(b4.transpose())).real + F * pert
    bz = b.differentiate('z')
    # plot_bot_2d(b)
    # plt.show()

    w['g'] = np.sqrt(2)*(eikx1*amp1*(w1.transpose()) + eikx2*amp2*(w2.transpose()) + eikx3*amp3*(w3.transpose()).real + eikx4*amp4*(w4.transpose())).real
    wz['g'] = np.sqrt(2)*(eikx1*amp1*(wz1.transpose()) + eikx2*amp2*(wz2.transpose()) + eikx3*amp3*(wz3.transpose()).real + eikx4*amp4*(wz4.transpose())).real
    p['g'] = np.sqrt(2)*(eikx1*amp1*(p1.transpose()) + eikx2*amp2*(p2.transpose()) + eikx3*amp3*(p3.transpose()).real + eikx4*amp4*(p4.transpose())).real
    u['g'] = np.sqrt(2)*(eikx1*amp1*(u1.transpose()) + eikx2*amp2*(u2.transpose()) + eikx3*amp3*(u3.transpose()).real + eikx4*amp4*(u4.transpose())).real
    uz['g'] = np.sqrt(2)*(eikx1*amp1*(uz1.transpose()) + eikx2*amp2*(uz2.transpose()) + eikx3*amp3*(uz3.transpose()).real + eikx4*amp4*(uz4.transpose())).real
    
    w.set_scales(1)
    wz.set_scales(1)
    b.set_scales(1)
    bz.set_scales(1)
    p.set_scales(1)
    u.set_scales(1)
    uz.set_scales(1)


    # Timestepping and output
    dt = 5e-4
    stop_sim_time = 200
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

profiles = solver.evaluator.add_file_handler('profiles_eq', sim_dt=0.1, max_writes=100, mode=fh_mode)
profiles.add_task("P*integ(dz(b) - 1, 'x') / Lx", name='diffusive_flux')
profiles.add_task("integ(w*b, 'x') / Lx", name='adv_flux')
profiles.add_task("integ(integ(w**2 + u**2, 'x'), 'z') / Lx", name='ke')
profiles.add_task("integ(integ((dx(w) - uz)**2, 'x'), 'z') / Lx", name='enst')

checkpoints = solver.evaluator.add_file_handler('checkpoints_eq', sim_dt=0.25, max_writes=50, mode=fh_mode)
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
