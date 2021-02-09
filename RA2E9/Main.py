import cProfile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import dedalus.public as de
import eigentools as eig
import numpy as np
from collections import OrderedDict 
from mpi4py import MPI
import os
import pickle
import scipy.sparse.linalg
from multiprocessing import Process
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
from Simulation import *     # pylint: disable=unused-wildcard-import

# Settings/options
restart = False
load_profile = False
subDir = '/results'
profile_file_name = os.path.dirname(os.path.abspath(__file__)) + '/avg_profs/averaged_avg_profs_ra1e8.h5'
path = os.path.dirname(os.path.abspath(__file__)) + subDir
if (not os.path.exists(path)):
    logger.warning('No results directory to resume from. Restarting...')
    restart = True
    
# Global parameters
Nz = 1024
pi_range = 20
Prandtl = 1
Rayleigh = 1e9
growth_tol = 1e-9
end_sim_time = 1e12
timestep_red_factor = 5
amp_gain_coeff = 1.0

# Temporal params
del_t_newton_nom = 0.000125
del_t_broyden_nom = 0.000125
T_mem_coeff = 1.0
flex_regime = False
rapid_adapt = False

# Macro timestep params
dT_regime = False
suppress_dT = True
del_T = 0.001
avgd_iter_count = 50
show_avg_flux = False
write_avgd_efs = False

# Ra growth params
ra_growth_mode = False
Ra_growth_coeff_nom = 1.000
Ra_growth_coeff = 1.00

##################################################
# Initialize simulation object with params
##################################################
logger.warning('Rank ' + str(CW.rank) + ' reporting for duty!')
sim = Simulation(subDir, Nz, pi_range, del_t_newton_nom, del_t_broyden_nom, T_mem_coeff, Rayleigh, Ra_growth_coeff, Prandtl, growth_tol)
sim.suppress_memory = True


##################################################
# 1 MARGINAL MODE
# Newtons's method finds the amplitude associated with the new marginal state
##################################################
def step_single_mode(sim):
    if (sim.attempt > 0):
        logger.warning('Repeating iteration ' + str(sim.iteration))
        logger.warning('Attempt: ' + str(sim.attempt + 1))
    logger.info('Step size: ' + str(sim.del_t))
    kx_m = sim.kx_marginals[0]
    # kx_interp = kx_interps[0]

    # Evolve marginal profile according to diffusion
    logger.info("Diffusing marginal profile")
    bz_d = sim.diffusion_IVP()
    logger.info("Solving EVP for diffused profile")

    # Solve EVP for diffused profile
    # diffused_EVs = sim.solve_EVP_spectrum(bz_d, caption_string='Diffused', show_spectrum=True, save_spectrum=True)
    growth_d = sim.solve_EVP_kx(bz_d, kx_m)
    logger.info('Diffusive growth rate = ' + str(growth_d) + ' at kx = ' + str(kx_m))
    if (growth_d < 0):
        raise Exception('Negative diffusive growth rate calculated. Terminating loop.')
    
    # Evolve marginal profile according to advection
    wb_zz = sim.solve_EVP_wb()[0]
    sim.b0z_a.set_scales(sim.domain.dealias)
    sim.b0z.set_scales(sim.domain.dealias)
    logger.info('Advecting marginal background state')
    sim.b0z_a['g'] = sim.b0z['g'] - sim.del_t * wb_zz
    sim.b0z.set_scales(1)

    logger.info("Solving EVP for advected profile")
    growth_a = sim.solve_EVP_kx(sim.b0z_a, kx_m)
    if (growth_a > 0):
        raise Exception('Positive advective growth rate: ' + str(growth_a) + ' at kx: ' + str(kx_m))
    logger.info("Advective growth rate = " + str(growth_a) + " for kx = " + str(kx_m))

    amp = -growth_d/growth_a
    logger.info("First order corrected advective flux amplitude = " + str(amp))
    logger.info('Entering amplitude iterative solver now...')

    # Evolve marginal profile wrt diffusion and advection
    evs, b0z, amp, kx_m = sim.newton_amplitude_solver(amp)
    sim.next_iteration(evs, b0z['g'], [amp])


##################################################
# 2 OR MORE MARGINAL MODES 
# Broyden's method finds amplitudes associated with the new marginal state
##################################################
def step_multiple_modes(sim):
    if (sim.attempt > 0):
        logger.warning('Repeating iteration ' + str(sim.iteration))
        logger.warning('Attempt: ' + str(sim.attempt + 1))
    logger.info('Step size: ' + str(sim.del_t))
    logger.info('Memory suppression: ' + str(sim.suppress_memory))
    logger.info('Ra growth mode: ' + str(ra_growth_mode))
    
    # Evolve marginal profile according to diffusion
    logger.info("Diffusing marginal profile")
    bz_d = sim.diffusion_IVP()
    logger.info("Solving EVP for diffused profile")

    # Solve EVP for diffused profile
    growth_d_vec = np.zeros((sim.kx_regime, 1))
    kx_neg = -1
    for i in range(CW.rank, len(sim.kx_marginals), CW.size):
        kx_m = sim.kx_marginals[i]
        growth_d_vec[i, 0] = sim.solve_EVP_kx(bz_d, kx_m) - sim.ev_dict[kx_m]
        if (growth_d_vec[i, 0] < 0):
            kx_neg = i
            logger.warning('Diffused EV remains stable (might yield negative amplitude) at index: ' + str(kx_neg))
    if CW.rank == 0:
        CW.Reduce(MPI.IN_PLACE, growth_d_vec, op=MPI.SUM, root=0)
    else:
        CW.Reduce(growth_d_vec, growth_d_vec, op=MPI.SUM, root=0)
    CW.Bcast(growth_d_vec, root=0)
    kx_neg = CW.reduce(kx_neg, op=MPI.MAX)
    sim.kx_neg = CW.bcast(kx_neg, root=0)

    for i in range(sim.kx_regime):
        logger.info('Diffusive growth rate = ' + str(growth_d_vec[i][0]) + ' at kx = ' + str(sim.kx_marginals[i]))

    wb_zz_ar = sim.solve_EVP_wb()
    wb_zz1 = wb_zz_ar[0]
    wb_zz1_mid = wb_zz1[len(wb_zz1) // 2]
    logger.info('Current wbzz1 midpoint: ' + str(wb_zz1_mid))
    logger.info('Threshold wbzz1 midpoint: ' + str(1e-6))
    # for i, iterate in enumerate(sim.iters_new_ts):
    #     logger.info('examining iteration of ts reduction: ' + str(iterate))
    #     if (iterate > sim.iteration):
    #         logger.info('removing iteration: ' + str(sim.iters_new_ts[i]))
    #         del sim.iters_new_ts[i]
    # sim.iters_new_ts = [0]
    if (abs(wb_zz1_mid) > 1e-6):
        logger.info('iterations new timestep: ' + str(sim.iters_new_ts))
        if (len(sim.iters_new_ts) == 0 or (sim.iteration - sim.iters_new_ts[-1]) > 5):
            sim.del_t_broyden /= 1.2
            sim.iters_new_ts.append(sim.iteration)
            logger.info('!!!!! Instability in advective flux. Reducing time step to ' + str(sim.del_t_broyden))
    
    if (abs(wb_zz1_mid) < 1e-8 and sim.iteration > 2000 and (sim.iteration - sim.iters_new_ts[-1]) > 30):
        sim.del_t_broyden *= 1.2
        sim.iters_new_ts.append(sim.iteration)
        logger.info('Instability is not apparent. Increasing time step to ' + str(sim.del_t_broyden))

    sim.Rayleigh *= sim.ra_growth_coeff
    logger.info('Ra exponential growth factor = ' + str(sim.ra_growth_coeff))
    logger.info('Ra = ' + str(sim.Rayleigh))
    sim.Rayleigh *= sim.ra_growth_coeff
    sim.b0z.set_scales(sim.domain.dealias)
    growth_a_mat = np.zeros((sim.kx_regime, sim.kx_regime))
    for ind in range(CW.rank, sim.kx_regime**2, CW.size):
        i = int(ind / sim.kx_regime)
        j = ind - i*sim.kx_regime
        kx_m = sim.kx_marginals[j]  
        logger.info('Advecting marginal profile (kx = ' + str(sim.kx_marginals[i]) + ')')
        sim.b0z_a.set_scales(sim.domain.dealias)
        sim.b0z.set_scales(sim.domain.dealias)
        sim.b0z_a['g'] = sim.b0z['g'] - sim.del_t * wb_zz_ar[i]
        sim.b0z_a.set_scales(1)
        logger.info('Solving EVP for advected profile (kx = ' + str(sim.kx_marginals[i]) + ')')
        # for j, kx_m in enumerate(sim.kx_marginals):
        growth_a_mat[j, i] = sim.solve_EVP_kx(sim.b0z_a, kx_m) - sim.ev_dict[kx_m]
    if (CW.rank == 0):
        CW.Reduce(MPI.IN_PLACE, growth_a_mat, op=MPI.SUM, root=0)
    else:
        CW.Reduce(growth_a_mat, growth_a_mat, op=MPI.SUM, root=0)
    CW.Bcast(growth_a_mat, root=0)
    # print('Advective growth matrix: ' + str(growth_a_mat))
    sim.b0z.set_scales(1)
    # for i in range(sim.kx_regime):
    #     logger.info('Advecting marginal profile (kx = ' + str(sim.kx_marginals[i]) + ')')
    #     sim.b0z_a.set_scales(sim.domain.dealias)
    #     sim.b0z_a['g'] = sim.b0z['g'] - sim.del_t * wb_zz_ar[i]
    #     sim.b0z_a.set_scales(1)
    #     logger.info('Solving EVP for advected profile (kx = ' + str(sim.kx_marginals[i]) + ')')
    #     for j, kx_m in enumerate(sim.kx_marginals):
    #         growth_a_mat[j, i] = sim.solve_EVP_kx(sim.b0z_a, kx_m) - sim.ev_dict[kx_m]
    # sim.b0z.set_scales(1)

    amp_gain_vec = amp_gain_coeff * np.ones((sim.kx_regime, 1))
    # if (min(amp_vec1)[0] < 0 or (sim.kx_regime == 3 and max(amp_vec1)[0] == amp_vec1[1][0])):
    amp_vec1 = -np.multiply(np.matmul(np.linalg.inv(growth_a_mat), growth_d_vec), amp_gain_vec)
    if (CW.rank == 0 and min(amp_vec1)[0] < 0):
        logger.info('Nominal First Guess: ' + str(amp_vec1))
        if (len(sim.amplitudes[-1]) == sim.kx_regime):
            amp_vec1 = np.array([sim.amplitudes[-1]]).T
            logger.warning('Bad initial guess, recycling amplitudes from previous iteration...')
        else:
            # amp_vec1 = 100*np.ones_like(amp_vec1)
            amp_vec1 = np.abs(amp_vec1)
            for i, amps in enumerate(sim.amplitudes[::-1]):
                if (len(amps) == sim.kx_regime):
                    amp_vec1 = np.array([amps]).T
                    logger.info('Recycling amplitudes from iteration ' + str(sim.iteration - i))
                    break
            logger.warning('Bad initial guess, amplitudes from previous iteration are not compatible...')
    amp_vec1 = CW.bcast(amp_vec1, root=0)

    for i, amp in enumerate(amp_vec1):
        logger.info('[FIRST GUESS] Amp' + str(i + 1) + ': ' + str(amp[0]))
    sim.wb.set_scales(1)
    sim.wb['g'] = np.array([amp_vec1[i, 0]*sim.wb_ar[i] for i in range(sim.kx_regime)]).sum(axis=0)
    sim.wb_mem_avg()
    sim.wb.set_scales(1)
    b0z_e = sim.evolution_IVP(sim.wb)
    if (CW.rank == 0):
        plt.plot(sim.wb['g'])
        plt.title('Amplitudes: ' + str(amp_vec1))
        plt.savefig(sim.iteration_path + '/advection_approx1')
        plt.close()
    b0z_e['g'] = CW.bcast(b0z_e['g'], root=0)
    evs = sim.solve_EVP_spectrum(b0z_e, show_spectrum=False, save_spectrum=True, caption_string='Approx1')
    if (CW.rank == 0):
        kx_local_max = sim.get_kx_local_max(evs)
    else:
        kx_local_max = None
    kx_local_max = CW.bcast(kx_local_max, root=0)
    ev_dict = dict(evs)
    evs_local_max = []
    for i, kx in enumerate(kx_local_max):
        evs_local_max.append(ev_dict[kx])
        logger.info('[FIRST GUESS] Evolved EV local maximum ' + str(i+1) + ': ' + str(ev_dict[kx]) + ' at kx: ' + str(kx))
    evs_local_max = CW.bcast(evs_local_max, root=0)
    if (rapid_adapt and max([abs(ev) for ev in evs_local_max]) > 1e-3):
        raise ExpectedException('Bad initial guess. Try again buddy')
    ev_lm_vec1 = np.array([evs_local_max]).T

    logger.info('Approximating Jacobian matrix with finite difference method...')
    amp_del_vec = 1.1 * amp_vec1
    d_amp_vec = amp_del_vec - amp_vec1
    ev_del_mat = np.zeros((sim.kx_regime, sim.kx_regime))
    amp_del_partial = np.zeros_like(amp_vec1)
    for i in range(CW.rank, sim.kx_regime, CW.size):
        amp_del_partial[:] = amp_vec1
        amp_del_partial[i] = amp_del_vec[i]
        ev_del_vec, b0z_evolved = sim.solve_EVs_local_max(amp_del_partial, kx_local_max, find_maxima=False)
        ev_del_mat[:, i] = ev_del_vec
    if (CW.rank == 0):
        CW.Reduce(MPI.IN_PLACE, ev_del_mat, op=MPI.SUM, root=0)
    else:
        CW.Reduce(ev_del_mat, ev_del_mat, op=MPI.SUM, root=0)
    if (CW.rank == 0):
        J1 = np.zeros((sim.kx_regime, sim.kx_regime))
        for i in range(sim.kx_regime): #for each row
            for j in range(sim.kx_regime): #for each column
                J1[i, j] = (ev_del_mat[i, j] - evs_local_max[i]) / d_amp_vec[j][0]
        amp_vec2 = amp_vec1 - np.matmul(np.linalg.inv(J1), ev_lm_vec1)
    else:
        J1 = None
        amp_vec2 = None
    J1 = CW.bcast(J1, root=0)
    amp_vec2 = CW.bcast(amp_vec2, root=0)

    kx_neg = -1
    amp_neg = 0.0
    for i, amp in enumerate(amp_vec2):
        logger.info('[SECOND GUESS] Amp' + str(i+1) + ': ' + str(amp[0]))                
        if amp[0] < amp_neg:
            kx_neg = i
            amp_neg = amp[0]

    if (min(amp_vec2)[0] < 0):
        CW.barrier()
        stable_mode = False
        for i, kx in enumerate(sim.kx_marginals):
            if (not kx in sim.kx_ev_zero):
                kx_neg = i
                stable_mode = True
                logger.info('rejecting stable mode...')
        if (not stable_mode):
            for i, kx in enumerate(sim.kx_marginals):
                if (not kx in sim.kx_ev_max):
                    kx_neg = i
                    logger.info('rejecting mode which is not a local maximum...')
        kx_neg = CW.bcast(kx_neg, root=0)
        # if (amp_neg < -1e2):
        #     EV_arg = sim.evs
        #     kx_local_max = []
        #     EV_arg.sort(key=lambda x:x[0])
        #     for index in range(1, len(EV_arg) - 1):
        #         if (EV_arg[index - 1][1] < EV_arg[index][1] and EV_arg[index][1] > EV_arg[index + 1][1] and EV_arg[index][1] > -sim.growth_tol):
        #             kx_local_max.append(EV_arg[index][0])

            # for amps in sim.amplitudes[::-1]:
            #     if len(amps) == sim.kx_regime:
            #         kx_neg = amps.index(min(amps))
            #         logger.warning('ditching small amplitude mode index: ' + str(kx_neg))
            #         break
        # kx_neg = -1
        # ev_marg_min = 0.0
        # for ind, kx in enumerate(sim.kx_marginals):
        #     if (kx in sim.ev_dict.keys() and sim.ev_dict[kx] < ev_marg_min):
        #         kx_neg = ind
        # logger.warning('Negative amplitude calculated on second guess.')
        if (sim.kx_neg != -1):
            kx_neg = sim.kx_neg
            sim.kx_neg = -1
        raise ExpectedException('Negative amplitude calculated on second guess. Ignoring stable mode: kx' + str(kx_neg))

    evs_local_max, b0z_evolved, kx_local_max, evs = sim.solve_EVs_local_max(amp_vec2, kx_local_max, find_maxima=True)
    for i in range(sim.kx_regime):
        logger.info('[SECOND GUESS] Evolved EV local maximum ' + str(i+1) + ': ' + str(evs_local_max[i]) + ' at kx: ' + str(kx_local_max[i]))
    ev_lm_vec2 = np.array([evs_local_max]).T

    d_ev_vec = ev_lm_vec2 - ev_lm_vec1
    d_amp_vec = amp_vec2 - amp_vec1

    logger.info('Jacobian matrix constructed. Entering Broyden solver...')
    evs, b0z_evolved, amp_vec_conv = sim.broyden_amplitude_solver(J1, kx_local_max, b0z_evolved, evs, ev_lm_vec2, d_ev_vec, amp_vec2, d_amp_vec)
    for i in range(sim.kx_regime):
        logger.info('Successful amplitude ratio ' + str(i+1) + ': ' + str(amp_vec_conv[i][0] / amp_vec1[i][0]))
    const_flux_res = sim.next_iteration(evs, b0z_evolved['g'], list(amp_vec_conv[:, 0]))
    sim.attempt += 1
    if (ra_growth_mode and sim.ra_growth_coeff == 1.0 and const_flux_res < 2e-4):
        sim.ra_growth_coeff = Ra_growth_coeff_nom
        sim.suppress_memory = True
        logger.info('Flux is stable and ra growth implemented. Initiating exponential flux growth.')


##################################################
# MACRO TIME STEP
# THIS AVERAGES THE ADVECTIVE FLUX FROM LAST SEVERAL ITERATIONS 
# AND TAKES A SINGLE LARGE TIMESTEP
# 1 AMPLITUDE FOR THE AVERAGED FLUX MEANS WE CAN ONLY MARGINALIZE 1 MODE
##################################################
def step_macro(sim):
    sim.dT_regime = True
    logger.info("Beginning macro timestep evolution process.")
    sim.del_t = del_T
    sim.wb['g'] = sim.get_avg_wb(avgd_iter_count, show_flux=show_avg_flux)
    sim.kx_regime = 1
    evs, b0z, amp, kx_m = sim.newton_amplitude_solver(1.0)
    logger.info('Macro timestep evolution process succeeded. Returning to nominal process.')
    sim.next_iteration(evs, b0z['g'], [amp])
    sim.dT_regime = dT_regime = False

def print_nu_del():
    phi = sim.b0z['g'][0]
    Prandtl = sim.Prandtl
    Rayleigh = sim.Rayleigh
    P = (Rayleigh * Prandtl)**(-1/2)
    nu = -phi
    logger.info('Nusselt number: ' + str(nu))
    for i in range(Nz):
        if (sim.b0z['g'][i] > 0):
            logger.info('b0z < 0 found at index: ' + str(i))
            z_bl = sim.z[i]
            del_bl = z_bl + 0.5
            logger.info('boundary layer thickness: ' + str(del_bl))
            break

##################################################
# Restart process or pick up where we left off
##################################################
if (restart):
    if (load_profile):
        sim.load_profile(profile_file_name)
        sim.evs = sim.marginalize_stable_profile()
        sim.plot_EVs(sim.evs, save_spectrum=True, show_spectrum=True)
    else:
        try:
            sim.construct_profile()
        except Exception as e:
            if (not isinstance(e, ExpectedException)):
                logger.error('Unexpected exception: ' + str(e))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logger.error(exc_type, fname, exc_tb.tb_lineno)
                sys.exit()            
        logger.info('Background profile constructed, calculating EV spectrum now...')
        logger.info('Calculating EV spectrum now...')
        # sim.evs = pickle.load(open(sim.iteration_path + '/evs.pick', 'rb'))
        # sim.ev_dict = dict(sim.evs)
        # logger.info('EV spectrum loaded. Proceeding with main loop.')
        sim.evs = sim.solve_EVP_spectrum(sim.b0z, save_spectrum=True, show_spectrum=False)
    sim.ev_dict = dict(sim.evs)
    pickle.dump(sim.evs, open(sim.iteration_path + '/evs.pick', 'wb'))
else:
    sim.retrieve_state()
    logger.info('Background profile retrieved')
    try:
        sim.evs = pickle.load(open(sim.iteration_path + '/evs.pick', 'rb'))
        sim.ev_dict = dict(sim.evs)
        logger.info('EV spectrum loaded. Proceeding with main loop.')
    except:
        logger.info('Calculating EV spectrum now...')
        sim.evs = sim.solve_EVP_spectrum(sim.b0z, save_spectrum=True, show_spectrum=False)
        sim.ev_dict = dict(sim.evs)
        pickle.dump(sim.evs, open(sim.iteration_path + '/evs.pick', 'wb'))

print_nu_del()

##################################################
# Survey eigenvalue spectrum for maginal modes
##################################################
kx_max, growth_max = sim.evs[0]
# print(CW.rank, sim.evs)
if (CW.rank == 0):
    sim.kx_marginals = sim.get_kx_local_max(sim.evs, new_iteration=True)
    sim.kx_marginals_ar.append(sim.kx_marginals)
    growth_m_ar = []
    for i, kx in enumerate(sim.kx_marginals):
        growth_m = sim.ev_dict[kx]
        logger.info('Local max growth rate ' + str(i+1) + ': ' + str(growth_m) + ' for kx: ' + str(kx))
        growth_m_ar.append(growth_m)
else:
    growth_m_ar = None
    sim.kx_marginals = None
    sim.kx_marginals_ar = None
sim.update_dt_params()
growth_m_ar = CW.bcast(growth_m_ar, root=0)
sim.kx_marginals = CW.bcast(sim.kx_marginals, root=0)
sim.kx_marginals_ar = CW.bcast(sim.kx_marginals_ar, root=0)
sim.kx_regime = CW.bcast(sim.kx_regime, root=0)
logger.info('Absolute max growth rate: ' + str(growth_max) + ' for kx: ' + str(kx_max))

##################################################
# Writes averaged eigenfunctions for 2D simulation
##################################################
if (write_avgd_efs):
    sim.store_avg_eigenfunctions(avgd_iter_count)

def step(sim):
    global dT_regime

    logger.info('############################################################################')
    logger.info("Iteration " + str(sim.iteration) + " beginning...")
    logger.info('Ra = ' + str(sim.Rayleigh))
    logger.info("Marginal wavenumbers: " + str(sim.kx_marginals))
    
    ##################################################
    # Toggles macro timestep if not supressed 
    ##################################################
    if (sim.iteration > 1000 and sim.iteration % 100 ==0):
        dT_regime = not suppress_dT

    ##################################################
    # One marginal mode
    ##################################################
    if (sim.kx_regime == 1):
        sim.attempt = 0
        sim.del_t = sim.del_t_newton
        while (sim.attempt == 0 or repeat_iteration):
            sim.del_t = sim.del_t_newton / (timestep_red_factor**sim.attempt)
            try:
                repeat_iteration = False
                step_single_mode(sim)
            except Exception as e:
                CW.Barrier()
                logger.warning('Iteration failled with exception: ' + str(e))
                repeat_iteration = True
            finally:
                sim.attempt += 1
                if (sim.attempt > 10):
                    logger.error('Process encountered terminal error')
                    sys.exit()


    ##################################################
    # Two or more marginal modes
    ##################################################
    elif (sim.kx_regime > 1 and not dT_regime):
        # sim.suppress_memory = False
        sim.attempt = 0
        sim.del_t = sim.del_t_broyden
        while (sim.attempt == 0 or repeat_iteration):
            sim.del_t = sim.del_t_broyden / (timestep_red_factor**sim.attempt)
            try:
                repeat_iteration = False
                if (sim.kx_regime > 1):
                    step_multiple_modes(sim)
                else:
                    step_single_mode(sim)
                # cProfile.runctx('step_multiple_modes(sim)', {'sim' : sim, 'step_multiple_modes' : step_multiple_modes}, {}, filename='profs/prof.{:d}'.format(CW.rank))

            # Exception handling used for adaptive timestepping. Certainly not ideal but it works.
            except Exception as e:
                CW.Barrier()
                if (not isinstance(e, ExpectedException)):
                    logger.error('Unexpected exception: ' + str(e))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    logger.error(exc_type, fname, exc_tb.tb_lineno)
                    sys.exit()
                if (sim.ra_growth_coeff > 1.0):
                    sim.ra_growth_coeff = 1.0
                    logger.info('Ra growth iteration failed. Fixing Ra to stabilize state')
                    continue
                elif ('Negative amplitude' in str(e)):
                    logger.warning('Negative amplitude in triple wavenumber regime. Returning to double wavenumber regime..')
                    sim.kx_regime -= 1
                    kx_neg_ind = int(str(e)[-1])
                    logger.warning('stable kx at index: ' + str(kx_neg_ind))
                    del sim.kx_marginals[kx_neg_ind]
                    continue
                else:
                    logger.warning('Solver failed with exception: ' + str(e))
                    if (sim.ra_growth_coeff == 1.0):
                        sim.suppress_memory = True
                    repeat_iteration = True
                    sim.attempt += 1
            finally:
                if (not sim.suppress_memory and sim.attempt > 5):
                    logger.info('Iteration failed (very badly). Suppressing memory term and retrying.')
                    sim.attempt = 0
                    sim.suppress_memory = True
                elif (sim.attempt > 10):
                    sys.exit()
    ##################################################
    # Macro timestep
    ##################################################
    else:
        step_macro(sim)

##################################################
##################################################
# MAIN LOOP
##################################################
##################################################
while(sim.sim_time < end_sim_time):
    step(sim)

# cProfile.runctx('profile_loop(sim)', {'sim' : sim, 'profile_loop' : profile_loop}, {}, filename='profs/prof.{:d}'.format(CW.rank))