import time
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
import h5py
import scipy.sparse.linalg
import scipy.signal
from multiprocessing import Process
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys

class Simulation():
    def __init__(self, subDir, Nz, pi_range, del_t_newton, del_t_broyden, T_mem_coeff, rayleigh, ra_growth_coeff, prandtl, growth_tol):
        self.path = os.path.dirname(os.path.abspath(__file__)) + subDir
        self.Nz = Nz
        self.z_basis = de.Chebyshev('z', Nz, interval=(-1/2, 1/2), dealias=3/2)
        self.domain = de.Domain([self.z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
        self.Nz_hires_coeff = 1.0
        self.z_basis_hires = eig.tools.basis_from_basis(self.z_basis, self.Nz_hires_coeff)
        self.domain_hires = de.Domain([self.z_basis_hires], grid_dtype=np.complex128, comm=MPI.COMM_SELF)
        self.z = self.domain.grid(0)
        self.b0z = self.domain.new_field()
        self.b0 = self.domain.new_field()
        self.wb = self.domain.new_field()
        self.b0z_a = self.domain.new_field()
        self.b0z_hires = self.domain_hires.new_field()
        self.init_b0z_func = 'log-atan'
        self.evs = None
        self.ev_dict = None
        self.ev_maxima_count = 0
        self.wb_ar = None
        self.b0z_d = None
        self.wb_mem = None
        self.Rayleigh = rayleigh
        self.ra_growth_coeff = ra_growth_coeff
        self.Prandtl = prandtl
        self.pi_range = pi_range
        self.del_t_newton = del_t_newton
        self.del_t_broyden = del_t_broyden
        self.T_mem = del_t_broyden*T_mem_coeff
        self.suppress_memory = True
        self.del_t = 0.0
        self.dT_regime = False
        self.attempt = 0
        self.iters_new_ts = []
        self.v0_eig = np.random.RandomState(seed=42).rand(Nz*7)
        self.v0_eig_hires = np.random.RandomState(seed=42).rand(int(Nz*7*self.Nz_hires_coeff))
        self.kx_domain = list(map(lambda n: n*np.pi/2, range(1, pi_range*2)))
        self.growth_tol = growth_tol
        self.ivp_partitions = 5
        self.kx_regime = 1
        self.iteration = 0
        self.sim_time = 0.0
        self.sim_times = [0.0]
        self.RBC_EVP = self.build_EVP()
        self.RBC_EVP_HI = self.build_EVP(high_res=True)
        self.RBC_EP = eig.Eigenproblem(self.RBC_EVP, reject=False, sparse=True)
        self.solve_EP_manual = False
        if (CW.rank == 0 and not os.path.exists(self.path)):
            os.mkdir(self.path)   
    # Builds Eigentools EigenProblem object from specified background profile and wave number
    def build_EVP(self, kx=1, b0z=None, override_cutoff=False, high_res=False):
        if (high_res):
            args = [self.domain_hires, ['p', 'b', 'u', 'w', 'bz', 'uz', 'wz']]
        else:
            args = [self.domain, ['p', 'b', 'u', 'w', 'bz', 'uz', 'wz']]
        # args.append('omega')
        if (override_cutoff):
            rayleigh_benard = de.EVP(*args, ncc_cutoff=1e-18, eigenvalue='omega')   # pylint: disable=no-value-for-parameter
        else:
            rayleigh_benard = de.EVP(*args, eigenvalue='omega')   # pylint: disable=no-value-for-parameter
            # rayleigh_benard = de.EVP(*args, max_ncc_terms=100, ncc_cutoff=1e-18, eigenvalue='omega')   # pylint: disable=no-value-for-parameter
        if (b0z == None):
            if (high_res):
                self.b0z.set_scales(self.Nz_hires_coeff)
                self.b0z_hires['g'] = self.b0z['g']
                rayleigh_benard.parameters['b0z'] = self.b0z_hires
                self.b0z.set_scales(1)
            else:
                rayleigh_benard.parameters['b0z'] = self.b0z
        else:
            if (high_res):
                b0z.set_scales(self.Nz_hires_coeff)
                self.b0z_hires['g'] = b0z['g']
                rayleigh_benard.parameters['b0z'] = self.b0z_hires
                self.b0z.set_scales(1)
            else:
                rayleigh_benard.parameters['b0z'] = b0z
        rayleigh_benard.parameters['kx'] = kx   #horizontal wavenumber
        rayleigh_benard.parameters['P'] = (self.Rayleigh * self.Prandtl)**(-1/2)
        rayleigh_benard.parameters['R'] = (self.Rayleigh / self.Prandtl)**(-1/2)
        rayleigh_benard.substitutions['dx(A)'] = '1j*kx*A'
        rayleigh_benard.substitutions['dt(A)'] = 'omega*A'
        rayleigh_benard.add_equation("dx(u) + wz = 0")
        rayleigh_benard.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz)) + b0z*w     = -(u*dx(b) + w*bz)")
        rayleigh_benard.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
        rayleigh_benard.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
        rayleigh_benard.add_equation("bz - dz(b) = 0")
        rayleigh_benard.add_equation("uz - dz(u) = 0")
        rayleigh_benard.add_equation("wz - dz(w) = 0")
        rayleigh_benard.add_bc("left(b) = 0")
        rayleigh_benard.add_bc("left(u) = 0")
        rayleigh_benard.add_bc("left(w) = 0")
        rayleigh_benard.add_bc("right(b) = 0")
        rayleigh_benard.add_bc("right(u) = 0")
        rayleigh_benard.add_bc("right(w) = 0")
        self.b0z.set_scales(1)
        return rayleigh_benard

    # Creates analytic profile that should achieve marginal stability
    def construct_profile(self):
        def construct_tanh_b0z(B):
            a = B / (2*B - 2*np.log(np.cosh(B)))
            return a*(np.tanh(B*(self.z+0.5)) - np.tanh(B*(self.z-0.5)) - 2)
        
        def construct_log_atan_b0z(d):
            # self.b0['g'] = 3/8 - (np.sqrt(3)/(8*np.pi)*np.log((1 + 2*np.pi/(3*np.sqrt(3))*(self.z+0.5)/d)**3 / (1 + (2*np.pi/(3*np.sqrt(3))*(self.z+0.5)/d)**3)) + 3/(4*np.pi)*np.arctan(4*(self.z+0.5)/(9*d) - 1/np.sqrt(3)))
            # midpt = self.b0['g'][self.Nz // 2 - 1]
            # self.b0['g'] = (self.b0['g'] - midpt) / (1 - 2*midpt)
            # self.b0['g'][self.Nz // 2:] = -self.b0['g'][self.Nz // 2 - 1::-1]
            # b0z = self.b0.differentiate(0)
            # b0z.set_scales(1)
            # self.b0.set_scales(1)
            # plt.plot(self.b0['c'])
            # plt.show()
            # d = 0.01
            num1fac1 = np.sqrt(3)*(8*np.pi**3*(self.z + 0.5)**3 / (81*np.sqrt(3)*d**3) + 1)
            num1fac2_term1 = 2*np.pi * (2*np.pi*(self.z + 0.5)/(3*np.sqrt(3)*d) + 1)**2 / (np.sqrt(3)*d*((8*np.pi**3*(self.z+0.5)**3)/(81*np.sqrt(3)*d**3)+1))
            num1fac2_term2 = 8*np.pi**3*(self.z+0.5)**2*(np.pi*2*(self.z+0.5)/(3*np.sqrt(3)*d) + 1)**3 / (27*np.sqrt(3)*d**3*((8*np.pi**3*(self.z+0.5)**3) / (81*np.sqrt(3)*d**3) + 1)**2)
            den1 = 4*np.pi*(np.pi*2*(self.z+0.5)/(3*np.sqrt(3)*d)+1)**3
            term2 = 2 / (3*d*((1/np.sqrt(3) - np.pi*4*(self.z+0.5)/(9*d))**2 + 1))
            b0z_q4 = 0.5 * (-num1fac1 * (num1fac2_term1 - num1fac2_term2) / den1 - term2)
            b0z = b0z_q4 + b0z_q4[::-1]
            # plt.plot(b0z)
            # plt.show()
            return b0z

        def construct_b0z_array(B):
            if (self.init_b0z_func == 'log-atan'):
                return construct_log_atan_b0z(B)
            else:
                self.init_b0z_func = 'tanh'
                return construct_tanh_b0z(B)
        
        if (self.init_b0z_func == 'log-atan'):
            B_vals = {5e8 : 0.0004341752080607313}
            B1 = 0.0005
        else:
            B_vals = {1e8 : 194.64515, 2e8 : 246.01737783198408, 5e8 : 334.9652429555079}
            B1 = 10
            logger.warning('Invalid initial buoyancy profile selected. Proceeding with tanh...')

        if self.Rayleigh in B_vals.keys():    
            B = B_vals[self.Rayleigh] 
            self.b0z['g'] = construct_b0z_array(B)
        else:
            logger.info('Developing new marginal profile for Ra = ' + str(self.Rayleigh))
            pi_range_default = self.pi_range
            self.update_pi_range(pi_range=5)
            self.b0z['g'] = construct_b0z_array(B1)
            CW.barrier()
            ev1 = self.solve_EVP_spectrum(self.b0z)[0][1]
            logger.info('B parameter: ' + str(B1))
            logger.info('Ev: ' + str(ev1))
            if (CW.rank == 0):
                if (ev1 > 0):
                    B2 = 1.1*B1
                else:
                    B2 = 0.9*B1
            else:
                B2 = 0
            B2 = CW.bcast(B2, root=0)
            self.b0z['g'] = construct_b0z_array(B2)
            CW.barrier()
            ev2 = self.solve_EVP_spectrum(self.b0z)[0][1]
            logger.info('B parameter: ' + str(B2))
            logger.info('Ev: ' + str(ev2))
            if (CW.rank == 0):
                B_m = B1 - ev1 * (B2 - B1) / (ev2 - ev1)
            else:
                B_m = 0
            B_m = CW.bcast(B_m, root=0)
            self.b0z['g'] = construct_b0z_array(B_m)
            CW.barrier()
            ev_m = self.solve_EVP_spectrum(self.b0z)[0][1]
            ev_m = CW.bcast(ev_m, root=0)
            logger.info('B parameter: ' + str(B_m))
            logger.info('Ev: ' + str(ev_m))
            while (abs(ev_m) > self.growth_tol):
                CW.barrier()
                B2 = B_m
                if (CW.rank == 0):
                    B_m = B1 - ev1 * (B_m - B1) / (ev_m - ev1)
                B_m = CW.bcast(B_m, root=0)
                logger.info('B parameter: ' + str(B_m))
                ev1 = ev_m
                self.b0z['g'] = construct_b0z_array(B_m)
                CW.barrier()
                ev_m = self.solve_EVP_spectrum(self.b0z)[0][1]
                ev_m = CW.bcast(ev_m, root=0)
                logger.info('Ev: ' + str(ev_m))
                B1 = B2
            CW.barrier()
            logger.info('obtained marginal profile')
            self.update_pi_range(pi_range=pi_range_default)
        self.profiles = [self.b0z['g'].copy()]
        self.sim_times = [0.0]
        self.amplitudes = []
        self.kx_marginals_ar = []
        self.iteration = 0
        self.rbc_data = {'profiles' : self.profiles, 'sim_times' : self.sim_times, 'amplitudes' : self.amplitudes, 'kx_marginals_ar' : self.kx_marginals_ar, 'iteration' : self.iteration}
        self.iteration_path = self.path + '/Iteration'+str(self.iteration)
        if (CW.rank == 0 and not os.path.isdir(self.iteration_path)):
            os.mkdir(self.iteration_path)
        CW.barrier()
        self.sim_time = 0.0

    def load_profile(self, profile_file_name):
        f = h5py.File(profile_file_name, 'r')
        b0z_g = -f['kappa_flux'][()].squeeze() / ((self.Rayleigh / self.Prandtl)**(-1/2))
        N_sim = len(b0z_g) / self.Nz
        if (N_sim % 1 != 0.0):
            raise ValueError('Loaded profile resolution is not a whole number multiple of the selected EVP resolution')
        self.b0z.set_scales(int(N_sim))
        self.b0z['g'] = b0z_g
        self.b0z.set_scales(1)
        self.profiles = [self.b0z['g'].copy()]
        self.sim_times = [0.0]
        self.amplitudes = []
        self.kx_marginals_ar = []
        self.iteration = 0
        self.rbc_data = {'profiles' : self.profiles, 'sim_times' : self.sim_times, 'amplitudes' : self.amplitudes, 'kx_marginals_ar' : self.kx_marginals_ar, 'iteration' : self.iteration}
        self.iteration_path = self.path + '/Iteration'+str(self.iteration)
        if (not os.path.isdir(self.iteration_path)):
            os.mkdir(self.iteration_path)
        self.sim_time = 0.0

    def marginalize_stable_profile(self):
        evs = self.solve_EVP_spectrum(self.b0z, show_spectrum=True)
        ev_max_init = evs[0][1]
        delta_t_init = 0.0
        if (ev_max_init > self.growth_tol):
            raise Exception('Loaded profile is unstable')
        elif (abs(ev_max_init) < self.growth_tol):
            logger.info('Loaded profile is marginally stable')
            return evs
        else:
            logger.info('Loaded profile is stable. Diffusing to marginal stability...')
            delta_t_m = 0.07243517016976714
            b0z_diffused = self.diffusion_IVP(delta_t=delta_t_m)
            evs = self.solve_EVP_spectrum(b0z_diffused)
            ev_max = evs[0][1]
            while (abs(ev_max) > self.growth_tol):
                delta_t = delta_t_m - ev_max*(delta_t_m - delta_t_init) / (ev_max - ev_max_init)
                b0z_diffused = self.diffusion_IVP(delta_t=delta_t)
                evs = self.solve_EVP_spectrum(b0z_diffused)
                ev_max_init = ev_max
                ev_max = evs[0][1]
                delta_t_init = delta_t_m
                delta_t_m = delta_t
                logger.info('Diffusion time: ' + str(delta_t))
                logger.info('EV max: ' + str(ev_max))
            self.b0z = b0z_diffused
            return evs

    # Loads most recent profile from previous analysis
    def retrieve_state(self):
        self.profiles = pickle.load(open(self.path + '/rbc_profiles_grid.pick', "rb"))
        self.sim_times = pickle.load(open(self.path + '/sim_times.pick', 'rb'))
        try:
            self.amplitudes = pickle.load(open(self.path + '/amplitudes.pick', 'rb'))
        except EOFError:
            self.amplitudes = []
        self.kx_marginals_ar = pickle.load(open(self.path + '/kx_marginals.pick', 'rb'))
        self.rbc_data = pickle.load(open(self.path + '/rbc_data.pick', 'rb'))
        if ('pi_range' in self.rbc_data.keys()):
            self.update_pi_range(pi_range = self.rbc_data['pi_range'])
        if ('Ra' in self.rbc_data.keys()):
            try:
                self.Rayleigh = self.rbc_data['Ra'][-1]
            except:
                pass
        self.iteration = len(self.profiles) - 1
        if ('del_t_broyden' in self.rbc_data.keys() and self.iteration > 100):
            self.del_t_broyden = self.rbc_data['del_t_broyden']
        # if ('del_t_newton' in self.rbc_data.keys() and self.iteration > 100):
        #     self.del_t_newton = self.rbc_data['del_t_newton']
        if ('iters_new_ts' in self.rbc_data.keys()):
            self.iters_new_ts = self.rbc_data['iters_new_ts']
        if (self.iteration > 10):
            conv_data_old = pickle.load(open(self.path + '/Iteration'+str(self.iteration - 1) + '/convergence_data_Iteration' + str(self.iteration - 1) + '.pick', 'rb'))
            try:
                self.wb_mem = conv_data_old['wb_mem']
            except:
                self.wb_mem = np.zeros_like(self.profiles[-1])
        self.iteration_path = self.path + '/Iteration'+str(self.iteration)
        if (not os.path.isdir(self.iteration_path)):
            os.mkdir(self.iteration_path)
        self.sim_time = self.sim_times[-1]
        try:
            self.b0z['g'] = self.profiles[-1]
        except:
            prof = self.profiles[-1]
            N_p = len(prof) / self.Nz
            self.b0z.set_scales(N_p)
            self.b0z['g'] = prof
            self.b0z.set_scales(1)

    def solve_EVP_kx(self, b0z_arg, kx):
        # self.RBC_EVP.ncc_kw['cutoff'] = 1e-6
        # self.RBC_EVP.namespace['b0z'].value = b0z_arg   # pylint: disable=unsubscriptable-object
        # self.RBC_EVP.namespace['kx'].value = kx   # pylint: disable=unsubscriptable-object
        self.RBC_EVP = self.build_EVP(b0z=b0z_arg, kx=kx)
        solver = self.RBC_EVP.build_solver()

        # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
        solver.solve_sparse(solver.pencils[0], N=15, target=0, rebuild_coeffs=True, v0=self.v0_eig)
        # solver.solve_dense(solver.pencils[0])

        # Return largest imaginary part
        return np.max(solver.eigenvalues.real)

    def solve_EVP_local_max(self, b0z_arg):
        self.RBC_EVP = self.build_EVP(b0z=b0z_arg)
        solver = self.RBC_EVP.build_solver()

        # Create function to compute max growth rate for given kx
        def max_growth_rate(kx):
            # Change kx parameter
            self.RBC_EVP.namespace['kx'].value = kx   # pylint: disable=unsubscriptable-object
            # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
            solver.solve_sparse(solver.pencils[0], N=10, target=0, rebuild_coeffs=True, v0=self.v0_eig)
            # solver.solve_dense(solver.pencils[0])
            # Return largest imaginary part
            return np.max(solver.eigenvalues.real)

        kx_support = [] + self.kx_marginals
        kx_marginal_indices = [index for index, kx in enumerate(self.kx_domain) if kx in self.kx_marginals]
        for marginal_index in kx_marginal_indices:
            if (marginal_index == 0 or marginal_index == len(self.kx_domain) - 1):
                continue
            if (not self.kx_domain[marginal_index - 1] in kx_support):
                kx_support.append(self.kx_domain[marginal_index - 1])
            if (not self.kx_domain[marginal_index + 1] in kx_support):
                kx_support.append(self.kx_domain[marginal_index + 1])
        kx_support = sorted(kx_support)
        # Compute growth rate over local wavenumbers
        kx_local = kx_support[CW.rank::CW.size]
        growth_local = np.array([max_growth_rate(kx) for kx in kx_local])

        # Reduce growth rates to root process
        growth_global = np.zeros_like(kx_support)
        growth_global[CW.rank::CW.size] = growth_local
        if CW.rank == 0:
            CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
        else:
            CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

        max_indices = list(scipy.signal.find_peaks(growth_global)[0])
        kx_ev_lm = [(kx_support[index], growth_global[index]) for index in max_indices]
        if (len(max_indices) == self.kx_regime):
            return kx_ev_lm, list(zip(kx_support, growth_global))
        if (len(max_indices) > self.kx_regime):
            lm_min_index = kx_ev_lm.index(min(kx_ev_lm, key = lambda e: e[1]))
            del kx_ev_lm[lm_min_index]
            return kx_ev_lm, list(zip(kx_support, growth_global))
        else:
            support_indices = [self.kx_domain.index(kx) for kx in kx_support]
            prev_index = support_indices[0]
            grouped_indices = [[prev_index]]
            missing_indices = []
            for index in support_indices[1:]:
                if index - prev_index == 1:
                    grouped_indices[-1].append(index)
                else:
                    grouped_indices.append([index])
                prev_index = index
            for index_group in grouped_indices:
                kx_group = [self.kx_domain[index] for index in index_group]
                ev_group = [ev for (kx, ev) in list(zip(kx_support, growth_global)) if kx in kx_group]
                if (ev_group[0] > ev_group[1] and index_group[0] - 1 >= 0):
                    missing_indices.append(index_group[0] - 1)
                if (ev_group[-1] > ev_group[-2] and index_group[-1] + 1 <= self.pi_range*2 - 1):
                    missing_indices.append(index_group[-1] + 1)
            missing_kx = [self.kx_domain[index] for index in missing_indices]
            missing_evs = [self.solve_EVP_kx(b0z_arg, kx) for kx in missing_kx]
            kx_evs = list(zip(kx_support, growth_global)) + list(zip(missing_kx, missing_evs))
            kx_evs.sort(key=lambda x:x[0])
            [kx_ar, ev_ar] = list(zip(*kx_evs))
            max_indices = list(scipy.signal.find_peaks(ev_ar)[0])
            kx_ev_lm = [(kx_ar[index], ev_ar[index]) for index in max_indices]
            if (len(max_indices) == self.kx_regime):
                return kx_ev_lm, kx_evs
            elif (len(max_indices) > self.kx_regime):
                lm_min_index = kx_ev_lm.index(min(kx_ev_lm, key = lambda e: e[1]))
                del kx_ev_lm[lm_min_index]
                return kx_ev_lm, kx_evs
            else:
                kx_ev_spectrum = self.solve_EVP_spectrum(b0z_arg, solved_evs=kx_evs)
                kx_local_max = self.get_kx_local_max(kx_ev_spectrum)
                return [(kx, ev) for kx, ev in kx_ev_spectrum if kx in kx_local_max], kx_ev_spectrum

    def solve_EVP_spectrum(self, b0z_arg, solved_evs=None, caption_string=str(), show_spectrum=False, save_spectrum=False):

        # Create function to compute max growth rate for given kx
        def max_growth_rate(kx):
            # Change kx parameter
            self.RBC_EVP.namespace['kx'].value = kx   # pylint: disable=unsubscriptable-object
            # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
            solver.solve_sparse(solver.pencils[0], N=10, target=0, rebuild_coeffs=True, v0=self.v0_eig)
            ev = np.max(solver.eigenvalues.real)

            if (ev > 0.1 and self.iteration > 0):
                logger.warning('Spurious eigenvalue on rank ' + str(CW.rank) + ', recalculating with dense solver...')
                logger.warning('Original ev: ' + str(ev) + ' with kx: ' + str(kx))
                self.RBC_EVP_HI = self.build_EVP(kx=kx, b0z=b0z_arg, override_cutoff=True, high_res=True)
                logger.warning('auxiliary evp built')
                solver_hires = self.RBC_EVP_HI.build_solver()
                logger.warning('auxiliary solver built')
                solver_hires.solve_dense(solver_hires.pencils[0], rebuild_coeffs=True)
                ev_fin = np.isfinite(solver_hires.eigenvalues)
                ev_small = [ev_hr < 10 for ev_hr in solver_hires.eigenvalues]
                ev_good = [bool_fin and bool_small for bool_fin, bool_small in zip(ev_fin, ev_small)]
                solver_hires.eigenvalues = solver_hires.eigenvalues[ev_good]
                logger.warning('dense solve complete')
                ev = np.max(solver_hires.eigenvalues.real)
                logger.warning('recalculated ev: ' + str(ev))
                # if (ev > 0.001):
                #     plt.plot(self.RBC_EVP_HI.namespace['b0z']['g'].real)
                #     plt.savefig(self.iteration_path + 'b0z_badEV')
                #     sys.exit()
                # print('low res ev: ' + str(ev))
                # # b0z = self.RBC_EVP.namespace['b0z']['g']
                # # plt.plot(b0z)
                # # plt.savefig(self.iteration_path + '/spurious_b0z')
                # # solver.solve_dense(solver.pencils[0])
                # print('high res ev: ' + str(ev))
                # self.b0z.set_scales(1)
                # b0z_arg.set_scales(1)
            # Return largest imaginary part
            # print('solving for kx: ' + str(kx) + ' with rank: ' + str(CW.rank))
            return ev

        if (CW.rank == 0):
            b0z_g = b0z_arg['g'].copy()
        else:
            b0z_g = None
        b0z_g = CW.bcast(b0z_g, root=0)
        b0z_arg['g'] = b0z_g
        self.RBC_EVP = self.build_EVP(b0z=b0z_arg, override_cutoff=True)
        solver = self.RBC_EVP.build_solver()
        CW.Barrier()
        # if (solved_evs == None or len(solved_evs) == 0):
        # Compute growth rate over local wavenumbers
        kx_local = self.kx_domain[CW.rank::CW.size]
        growth_local = np.array([max_growth_rate(kx) for kx in kx_local])

        # Reduce growth rates to root process
        growth_global = np.zeros_like(self.kx_domain)
        growth_global[CW.rank::CW.size] = growth_local
        if CW.rank == 0:
            CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
        else:
            CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)
        # elif (len(solved_evs) != len(self.kx_domain)):
        #     [kx_solved, ev_solved] = list(zip(*solved_evs))
        #     kx_residuals = [kx for kx in self.kx_domain if kx not in kx_solved]
        #     kx_local = kx_residuals[CW.rank::CW.size]
        #     growth_local = np.array([max_growth_rate(kx) for kx in kx_local])
        #     ev_residuals = np.zeros_like(kx_residuals)
        #     ev_residuals[CW.rank::CW.size] = growth_local
        #     if CW.rank == 0:
        #         CW.Reduce(MPI.IN_PLACE, ev_residuals, op=MPI.SUM, root=0)
        #     else:
        #         CW.Reduce(ev_residuals, ev_residuals, op=MPI.SUM, root=0)
        #     kx_evs = solved_evs + list(zip(kx_residuals, ev_residuals))
        #     kx_evs.sort(key=lambda x:x[0])
        #     growth_global = list(zip(*kx_evs))[1]
        # else:
        #     growth_global = list(zip(*solved_evs))[1]

        # Plot growth rates from root process
        if (CW.rank == 0 and (save_spectrum or show_spectrum)):
            f, ax = plt.subplots(figsize=(5, 5))
            ax.plot(self.kx_domain, growth_global, '.')
            ax.set_xlim(0, self.pi_range*np.pi)
            ax.xaxis.set_major_locator(tck.MultipleLocator(5*np.pi))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(np.pi))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            ax.grid(True)
            plt.xlabel(r'$k_x$')
            plt.ylabel(r'$\mathrm{Im}(\omega)$')
            if (len(caption_string) > 0):
                plt.title(r'RBC growth rates (' + caption_string + '), Iteration ' + str(self.iteration) + ', Time ' + str(self.sim_time))
            else:
                plt.title(r'RBC growth rates, Iteration ' + str(self.iteration) + ', Time ' + str(self.sim_time))
            if (save_spectrum):
                if (len(caption_string) > 0):
                    plt.savefig(self.iteration_path + '/EV_spectrum_iteration' + str(self.iteration) + '_' + caption_string)
                else:
                    plt.savefig(self.iteration_path + '/EV_spectrum_iteration' + str(self.iteration))
                    plt.ylim(-0.005, 0.0005)
                    plt.xlim(0, self.pi_range*np.pi)
                    plt.savefig(self.iteration_path + '/EV_spectrum_zoomed_iteration' + str(self.iteration))
            if (show_spectrum):
                plt.show()
            plt.close()

        EVs = list(zip(list(self.kx_domain), list(growth_global)))
        EVs.sort(key=lambda x:x[1], reverse=True)
        CW.Barrier()
        return EVs
        
    def get_kx_local_max(self, EV_arg, new_iteration=False, halt_on_work_lm=False):
        # ev_sorted = EV_arg
        # ev_sorted.sort(key=lambda x:x[1])
        kx_local_max = []
        EV_arg.sort(key=lambda x:x[0])
        if new_iteration:
            for index in range(1, len(EV_arg) - 1):
                if (EV_arg[index - 1][1] < EV_arg[index][1] and EV_arg[index][1] > EV_arg[index + 1][1] and EV_arg[index][1] > -1e-4):
                    kx_local_max.append(EV_arg[index][0])
                elif(EV_arg[index][1] > -50*self.growth_tol):
                    kx_local_max.append(EV_arg[index][0])
        else:
            for index in range(1, len(EV_arg) - 1):
                if (EV_arg[index - 1][1] < EV_arg[index][1] and EV_arg[index][1] > EV_arg[index + 1][1] and EV_arg[index][1] > -1e4*self.growth_tol):
                    kx_local_max.append(EV_arg[index][0])
            if (len(kx_local_max) < self.kx_regime):
                EV_arg.sort(key=lambda x:x[1])
                for i in range(self.kx_regime):
                    if (not EV_arg[-i-1][0] in kx_local_max):
                        logger.info('included mode: ' + str(EV_arg[-i-1][0]))
                        kx_local_max.append(EV_arg[-i-1][0])
                    if (len(kx_local_max) == self.kx_regime):
                        break
            elif (len(kx_local_max) > self.kx_regime):
                EV_arg.sort(key=lambda x:x[1])
                logger.info('EV arg: ' + str(EV_arg))
                for i in range(len(EV_arg))[::-1]:
                    if (EV_arg[-i-1][0] in kx_local_max):
                        kx_local_max.remove(EV_arg[-i-1][0])
                        logger.info('excluded mode: ' + str(EV_arg[-i-1][0]))
                    if (len(kx_local_max) == self.kx_regime):
                        break
            kx_local_max = sorted(kx_local_max)
        # kx_local_max.append(EV_arg[2][0])
        # kx_local_max.append(EV_arg[3][0])
        evs_local_max = [kx_ev for kx_ev in EV_arg if kx_ev[0] in kx_local_max]
        evs_local_max.sort(key=lambda x:x[1], reverse=True)
        maxima_count = len(kx_local_max)
        if (new_iteration):
            self.ev_maxima_count = maxima_count
        logger.info('Quantity of local maxima EVs: ' + str(maxima_count))
        if (new_iteration):
            self.kx_regime = maxima_count
            return kx_local_max
        else:
            return kx_local_max
        # if (maxima_count == 0):
        #     raise Exception('Quantity of local maxima EVs is zero')
        # elif (maxima_count == 1):
        #     if (new_iteration):
        #         logger.info('Proceeding with single wavenumber regime.')
        #         self.kx_regime = 1
        #         return kx_local_max
        #     else:
        #         if (self.kx_regime == 1):
        #             return kx_local_max
        #         else:
        #             raise ExpectedException('One local maximum in Broyden regime')
        # elif (maxima_count == 2):  
        #     if (new_iteration):
        #         # if (evs_local_max[1][1] < -2e-4):
        #         #     logger.info('Two local maxima, one is stable. Proceeding with single wavenumber regime.')
        #         #     self.kx_regime = 1
        #         #     # self.del_t = self.del_t_nominal / 10.0
        #         #     kx_local_max.remove(evs_local_max[1][0])
        #         #     return kx_local_max
        #         # else:
        #         logger.info('Two marginalish local maxima. Proceeding with double wavenumber regime.')
        #         self.kx_regime = 2
        #         return kx_local_max
        #     else:
        #         if (self.kx_regime == 1):
        #             logger.warning('Ignoring the following kx: ' + str(evs_local_max[1][0]))
        #             kx_local_max.remove(evs_local_max[1][0])
        #             return kx_local_max
        #         elif (self.kx_regime > 1):
        #             if (self.kx_regime > 2):
        #                 raise ExpectedException('Two local maxima in triple wavenumber regime')
        #             return kx_local_max               
        # elif (maxima_count == 3):
        #     if (new_iteration):
        #         if (evs_local_max[2][1] < -5e-5):
        #             logger.info('Three local maxima, one is stable. Proceeding with double wavenumber regime.')
        #             self.kx_regime = 2
        #             kx_local_max.remove(evs_local_max[2][0])
        #             return kx_local_max
        #         else:
        #             logger.info('Three marginalish local maxima. Proceeding with triple wavenumber regime.')
        #             self.kx_regime = 3
        #             return kx_local_max
        #     else:
        #         if (self.kx_regime == 1):
        #             logger.warning('Ignoring the following kx: ' + str([evs_local_max[1][0], evs_local_max[2][0]]))
        #             return [evs_local_max[0][0]]
        #         elif (self.kx_regime == 2):
        #             logger.warning('Ignoring the following kx: ' + str(evs_local_max[2][0]))
        #             kx_local_max.remove(evs_local_max[2][0])
        #             return kx_local_max
        #         else:
        #             return kx_local_max
        # elif (maxima_count == 4):
        #     if (new_iteration):
        #         logger.info('Four marginal wavenumbers. Proceeding with quadrupel kx regime')
        #         self.kx_regime = 4
        #         return kx_local_max
        #     else:
        #         return kx_local_max
        # else:
        #     if (not new_iteration):
        #         if (self.kx_regime == 1):
        #             logger.warning('Ignoring the following kx: ' + str([evs_local_max[1][0], evs_local_max[2][0]]))
        #             return [evs_local_max[0][0]]
        #         else:
        #             # logger.warning('Ignoring the following kx: ' + str(evs_local_max[2][0]))
        #             for i in range(self.kx_regime, maxima_count):
        #             # for kx in evs_local_max[self.kx_regime:][0]:
        #                 print('remove kx: ' + str(evs_local_max[i][0]))
        #                 kx_local_max.remove(evs_local_max[i][0])
        #             if (len(kx_local_max) != self.kx_regime):
        #                 logger.error('Incorrect quantity of local max kx')
        #             else:
        #                 logger.info('It worked')
        #                 return kx_local_max
        #     else:
        #         raise Exception('Local max count is greater than 3. Unexpected case...')

    def solve_EVP_wb(self):
        # self.RBC_EVP.ncc_kw['cutoff'] = 1e-18'
        wb_zz_ar = np.zeros((self.kx_regime, int(self.Nz*self.domain.dealias[0])))
        self.wb_ar = np.zeros((self.kx_regime, self.Nz))
        data_path = self.iteration_path + '/data'
        if (CW.rank == 0 and not os.path.isdir(data_path)):
            os.mkdir(data_path)
        CW.Barrier()
        for i in range(CW.rank, len(self.kx_marginals), CW.size):
            kx = self.kx_marginals[i]
            logger.info('Solving EVP to obtain advective flux at kx: ' + str(kx))
            self.RBC_EP = eig.Eigenproblem(self.build_EVP(kx=kx, override_cutoff=True), sparse=True)
            if (self.solve_EP_manual):
                self.RBC_EP.pencil = 0
                self.RBC_EP.N = 0
                self.RBC_EP.target = 0
                self.RBC_EP.solver.solve_sparse(self.RBC_EP.solver.pencils[0], N=15, target=0, rebuild_coeffs=True, v0=self.v0_eig)
                ev_index = np.arange(len(self.RBC_EP.solver.eigenvalues), dtype=int)
                growth_m = np.max(self.RBC_EP.solver.eigenvalues.real)
                growth_index = np.where(self.RBC_EP.solver.eigenvalues.real == growth_m)[0]
                index_m = ev_index[growth_index[0]]
                freq_m = self.RBC_EP.solver.eigenvalues[growth_index[0]].imag
            else:
                try:
                    growth_m, index_m, freq_m = self.RBC_EP.growth_rate()
                except:
                    logger.warning('Eigentools solver failed to obtain growth rate. Retrying with manual solve...')
                    self.RBC_EP.pencil = 0
                    self.RBC_EP.N = 0
                    self.RBC_EP.target = 0
                    self.RBC_EP.solver.solve_sparse(self.RBC_EP.solver.pencils[0], N=15, target=0, rebuild_coeffs=True, v0=self.v0_eig)
                    ev_index = np.arange(len(self.RBC_EP.solver.eigenvalues), dtype=int)
                    growth_m = np.max(self.RBC_EP.solver.eigenvalues.real)
                    growth_index = np.where(self.RBC_EP.solver.eigenvalues.real == growth_m)[0]
                    index_m = ev_index[growth_index[0]]
                    freq_m = self.RBC_EP.solver.eigenvalues[growth_index[0]].imag

            # logger.warning('growth rate obtained, kx: ' + str(kx))
            wb_g = self.get_wb_save_data(self.RBC_EP, growth_m, index_m, freq_m)
            # logger.warning('eigenfunctions obtained, kx: ' + str(kx))
            self.wb_ar[i] = wb_g
            self.wb.set_scales(1)
            self.wb['g'] = wb_g
            self.b0z.set_scales(1)
            wb_zz = de.operators.differentiate(self.wb, z=2)['g']
            # if (False):
            #     if (not os.path.isdir(self.iteration_path + '/figures')):
            #         os.mkdir(self.iteration_path + '/figures')
            #     pi_mult = round(kx/np.pi, 1)
            #     plt.plot(wb_zz)
            #     plt.title(r'$<wb>_{zz}$' + ', ' + r'$k_x = $' + str(pi_mult) + r'$\pi$')
            #     plt.savefig(self.iteration_path + '/figures/' + str(int(pi_mult)) + 'p' + str(int(10*(pi_mult - int(pi_mult)))) + 'pi_wbzz')
            #     plt.close()
            wb_zz_ar[i] = wb_zz.real
            logger.warning('Approximate unstable growth: ' + str(growth_m) + ' at kx: ' + str(kx))
        if CW.rank == 0:
            CW.Reduce(MPI.IN_PLACE, wb_zz_ar, op=MPI.SUM, root=0)
            CW.Reduce(MPI.IN_PLACE, self.wb_ar, op=MPI.SUM, root=0)
        else:
            CW.Reduce(wb_zz_ar, wb_zz_ar, op=MPI.SUM, root=0)
            CW.Reduce(self.wb_ar, self.wb_ar, op=MPI.SUM, root=0)
        CW.Bcast(wb_zz_ar, root=0)
        CW.Bcast(self.wb_ar, root=0)
        return wb_zz_ar

    # Obtains eigenfunctions from EigenProblem object and saves them at each iteration
    def get_wb_save_data(self, EP, growth, index, freq):
        EP.solver.set_state(index)
        # logger.warning('rank: ' + str(CW.rank) + ', index')
        w_f = EP.solver.state['w']
        b_f = EP.solver.state['b']
        # if (self.iteration % 10 == 0):
        # w_f['c'][180:] = 0
        # b_f['c'][180:] = 0
        wb_exact = (w_f['g']*np.conj(b_f['g'])).real
        wb = 0.5*(wb_exact[:] + wb_exact[::-1])
        # logger.warning('rank: ' + str(CW.rank) + ', wb')
        # wb_temp = self.domain.new_field()
        # wbz = self.domain.new_field()
        # wb_temp['g'] = wb
        # wbz_temp = de.operators.differentiate(wb_temp, z=1)
        # wbz['g'] = wbz_temp['g']
        # # wbz_temp['c'][200:] = np.mean(wbz_temp[200:])
        # wbz['c'][200:] = wbz['c'][200]
        # wb_temp = wbz.antidifferentiate(self.z_basis, ('left', 0))
        # wb = wb_temp['g']
        kx = EP.EVP.namespace['kx'].value
        # logger.warning('rank: ' + str(CW.rank) + ', kx')
        data_g = {
            'time' : self.sim_time,
            'kx' : kx,
            'b0z' : EP.EVP.namespace['b0z']['g'],
            'growth' : growth,
            'index' : index,
            'freq' : freq,
            'w' : EP.solver.state['w']['g'],
            'b' : EP.solver.state['b']['g'],
            'p' : EP.solver.state['p']['g'],
            'u' : EP.solver.state['u']['g'],
            'bz' : EP.solver.state['bz']['g'],
            'uz' : EP.solver.state['uz']['g'],
            'wz' : EP.solver.state['wz']['g'],
            'wb_exact' : wb_exact,
            'wb' : wb,
        }
        # logger.warning('rank: ' + str(CW.rank) + ', dictionary')
        data_path = self.iteration_path + '/data'
        # if (CW.rank == 0 and not os.path.isdir(data_path)):
        #     os.mkdir(data_path)
        # logger.warning('rank: ' + str(CW.rank) + ', data path')
        pi_mult = round(kx/np.pi, 1)
        f_data_g = open(data_path + '/EigenFunctions_grid_iteration'+str(self.iteration)+ '_' + str(int(pi_mult)) + 'p' + str(int(10*(pi_mult - int(pi_mult)))) + 'pi.pick', 'wb')
        pickle.dump(data_g, f_data_g)
        # logger.warning('rank: ' + str(CW.rank) + ', dump')
        return wb

    # Evolves a given profile wrt diffusion and returns the diffused profile
    def diffusion_IVP(self, delta_t=None):
        ra_coeff=self.ra_growth_coeff
        problem = de.IVP(self.domain, variables=['b','bz'])
        problem.meta['b']['z']['dirichlet'] = True
        problem.parameters['P'] = (self.Rayleigh * ra_coeff * self.Prandtl)**(-1/2)
        problem.parameters['R'] = ((self.Rayleigh * ra_coeff) / self.Prandtl)**(-1/2)
        problem.add_equation("dt(b) - P*(dz(bz)) = 0")
        problem.add_equation("bz - dz(b) = 0")
        problem.add_bc("left(b) = 0")
        problem.add_bc("right(b) = 0")

        # Build solver
        solver = problem.build_solver(de.timesteppers.RK222)
        
        # Initial conditions
        b0z = self.domain.new_field()
        b0z['g'] = self.b0z['g'] + 1
        b0 = b0z.antidifferentiate(self.z_basis, ('left', 0))
        b = solver.state['b']
        b.set_scales(self.domain.dealias)
        b['g'] = b0['g']
        bz = solver.state['bz']
        bz.set_scales(self.domain.dealias)
        bz['g'] = b0z['g']

        # Timestepping and output
        if (delta_t == None):
            del_t = self.del_t
        else:
            del_t = delta_t
        dt = del_t / self.ivp_partitions
        solver.stop_sim_time = del_t

        # Main loop
        while solver.proceed:
            dt = solver.step(dt)
        
        # Return diffused profile
        # b.set_scales(1)
        b0z_diffused = b.differentiate(0)
        b0z_diffused.set_scales(1)
        b0z_diffused['g'] = b0z_diffused['g'] - 1
        return b0z_diffused

    # Evolves a given profile wrt diffusion, advection, and a given amplitude. Returns the evolved profile
    def evolution_IVP(self, wb, amp=1):
        problem = de.IVP(self.domain, variables=['b','bz'])
        problem.meta['b']['z']['dirichlet'] = True
        problem.parameters['P'] = (self.Rayleigh * self.Prandtl)**(-1/2)
        problem.parameters['R'] = (self.Rayleigh / self.Prandtl)**(-1/2)
        problem.parameters['A'] = amp
        problem.parameters['wb_z'] = de.operators.differentiate(wb, z=1)
        problem.add_equation("dt(b) - P*(dz(bz)) = -A*wb_z")
        problem.add_equation("bz - dz(b) = 0")
        problem.add_bc("left(b) = 0")
        problem.add_bc("right(b) = 0")

        # Build solver
        solver = problem.build_solver(de.timesteppers.RK222)
        
        # Initial conditions
        b0z = self.domain.new_field()
        b0z['g'] = self.b0z['g'] + 1
        b0 = b0z.antidifferentiate(self.z_basis, ('left', 0))
        b = solver.state['b']
        b.set_scales(self.domain.dealias)
        b['g'] = b0['g']
        bz = solver.state['bz']
        bz.set_scales(self.domain.dealias)
        bz['g'] = b0z['g']

        # Timestepping and output
        dt = self.del_t / self.ivp_partitions
        solver.stop_sim_time = self.del_t

        # Main loop
        while solver.proceed:
            dt = solver.step(dt)
        
        # Return diffused profile
        b.set_scales(1)
        b0z_evolved = b.differentiate(0)
        b0z_evolved.set_scales(1)
        b0z_evolved['g'] = b0z_evolved['g'] - 1
        return b0z_evolved

    # Given the a marginally stable background profile (obtained from the previous iteration), the advective flux, and a guess for the amplitude,
    # this obtains a new, evolved, marginally stable profile by finding the amplitude that yields a negligible eigenvalue
    def newton_amplitude_solver(self, amp_0):
        exp_coef = 1.1
        b0z = self.evolution_IVP(self.wb, amp=amp_0)
        evs = self.solve_EVP_spectrum(b0z)
        kx_0, growth_0 = evs[0]
        logger.info('First marginal growth approximation: ' + str(growth_0) + ' at kx: ' + str(kx_0))
        if (growth_0 > 0):
            amp_1 = amp_0*exp_coef
        else:
            amp_1 = amp_0/exp_coef
        logger.info('Second amplitude guess ' + str(amp_1))
        b0z = self.evolution_IVP(self.wb, amp=amp_1)
        b0z['g'] = CW.bcast(b0z['g'], root=0)
        evs = self.solve_EVP_spectrum(b0z)
        # plt.plot(b0z['g'], label='b0z 0')
        # plt.plot(b0z1['g'], label='b0z 1')
        # plt.legend()
        # plt.show()
        # print('rank: ' + str(CW.rank) + 'max diff: ' + str(max(np.abs(b0z['g'] - b0z1['g']))))
        kx_1, growth_1 = evs[0]
        logger.info('Second marginal growth approximation: ' + str(growth_1) + ' at kx: ' + str(kx_1))
        b0z.set_scales(1)
        attempt = 1
        while (attempt == 1 or abs(growth_lin) > self.growth_tol):
            logger.info('Beginning Newton iteration ' + str(attempt))
            if (CW.rank == 0):
                amp_lin = amp_0 - growth_0 * (amp_1 - amp_0) / (growth_1 - growth_0)
            else:
                amp_lin = None
            amp_lin = CW.bcast(amp_lin, root=0)
            logger.info("Refined advective flux amplitude = " + str(amp_lin))
            if (amp_lin < 0):
                # logger.warning('Newton\'s method gives negative amplitude. Calculating intercept (amp = 0) instead.')
                # amp_lin = 0
                raise ExpectedException('Newton\'s method gives negative amplitude. Terminating process :(((')
            # print('amplin: ' + str(amp_lin) + ', rank: ' + str(CW.rank))
            if (CW.rank == 0):
                b0z = self.evolution_IVP(self.wb, amp=amp_lin)
            b0z['g'] = CW.bcast(b0z['g'], root=0)
            evs = self.solve_EVP_spectrum(b0z)
            evs = CW.bcast(evs)
            kx, growth_lin = evs[0]
            b0z.set_scales(1)
            logger.info('Refined marginal growth approximation: ' + str(growth_lin) + ' at kx: ' + str(kx))
            amp_0 = amp_1
            growth_0 = growth_1
            amp_1 = amp_lin
            growth_1 = growth_lin
            attempt = attempt + 1
            if (attempt > 50):
                raise ExpectedException('Newton\'s method exceeded iteration limit. Terminating process :(((')        
            CW.Barrier()
        if (amp_lin < 0):
            raise Exception('Negative amplitude calculated')
        evs = CW.bcast(evs, root=0)
        b0z.set_scales(1)
        # plt.plot(b0z['g'], label=str(CW.rank))
        # if (CW.rank == 0):
        #     plt.legend()
        #     plt.show()
        #     plt.plot(evs)
        #     plt.show()
        self.wb.set_scales(1)
        if (CW.rank == 0):
            conv_data = {'Broyden' : False, 'time' : self.sim_time, 'del_t' : self.del_t, 'kx_marginals' : self.kx_marginals, 'kx' : kx, 'amp' : amp_lin, 'wb' : self.wb['g'], 'b0z' : self.b0z['g'], 'wb_mem' : amp_lin*self.wb['g']}
            f_wb = open(self.iteration_path + '/convergence_data_Iteration'+str(self.iteration) + '.pick', 'wb')
            pickle.dump(conv_data, f_wb)
        logger.info('Success! Solver converged on an amplitude that yields a marginally stable eigenvalue!')
        return evs, b0z, amp_lin, kx

    def broyden_amplitude_solver(self, J, kx_lm, b0z_e, evs, ev_vec1, d_ev, amp_vec1, d_amps):
        attempt = 0
        while (max([abs(ev[0]) for ev in ev_vec1]) > self.growth_tol):
            logger.info('Beginning Broyden iteration ' + str(attempt))
            J = J + np.matmul((d_ev - np.matmul(J, d_amps)) / np.linalg.norm(d_amps)**2, d_amps.T)
            amp_vec2 = amp_vec1 - np.matmul(np.linalg.inv(J), ev_vec1)
            # amp_neg = 0
            for i, amp in enumerate(amp_vec2):
                logger.info('Amp' + str(i+1) + ': ' + str(amp))
                # if amp < amp_neg:
                #     amp_neg = amp
                #     kx_neg = i
            kx_neg = -1
            ev_marg_min = 0.0
            if (min(amp_vec2)[0] < 0):
                EV_arg = self.evs
                kx_local_max = []
                EV_arg.sort(key=lambda x:x[0])
                for index in range(1, len(EV_arg) - 1):
                    if (EV_arg[index - 1][1] < EV_arg[index][1] and EV_arg[index][1] > EV_arg[index + 1][1] and EV_arg[index][1] > -self.growth_tol):
                        kx_local_max.append(EV_arg[index][0])
                for i, kx in enumerate(self.kx_marginals):
                    if (not kx in kx_local_max):
                        kx_neg = i 
                # for ind, kx in enumerate(self.kx_marginals):
                #     if (kx in self.ev_dict.keys() and self.ev_dict[kx] < ev_marg_min):
                #         kx_neg = ind
                raise ExpectedException('Negative amplitude calculated in Broyden\'s method, kx: ' + str(kx_neg))
            evs_lm, b0z_e, kx_lm, evs = self.solve_EVs_local_max(amp_vec2, kx_lm, find_maxima=True)
            for i in range(self.kx_regime):
                logger.info('Evolved EV local maximum ' + str(i + 1) + ': ' + str(evs_lm[i]) + ' at kx: ' + str(kx_lm[i]))
            ev_vec2 = np.array([evs_lm]).T

            d_ev = ev_vec2 - ev_vec1
            d_amps = amp_vec2 - amp_vec1
            ev_vec1 = ev_vec2
            amp_vec1 = amp_vec2
            attempt = attempt + 1
            if (attempt > 14 and max([abs(ev[0]) for ev in ev_vec1]) > self.growth_tol):
                raise ExpectedException('Broyden\'s method stalled.')
        if (min(amp_vec1)[0] < 0):
            raise Exception('Negative amplitude calculated in Broyden\'s method.')
        logger.info('Success! Broyden\'s method has converged.')
        # wb_mem = np.array([amp_vec1[i, 0]*self.wb_ar[i] for i in range(self.kx_regime)]).sum(axis=0)
        conv_data = {'Broyden' : True, 'time' : self.sim_time, 'del_t' : self.del_t, 'b0z' : self.b0z['g'], 'kx_marginals' : self.kx_marginals, 'wb_mem' : self.wb['g'], 'Ra' : self.Rayleigh}
        for i in range(self.kx_regime):
            if (amp_vec1[i, 0] < 0):
                logger.error('Solver converged on a negative amplitude (not supposed to happen). Terminating process.')
                sys.exit()
            conv_data['amp' + str(i+1)] = amp_vec1[i, 0]
            conv_data['wb' + str(i+1)] = self.wb_ar[i]
            conv_data['kx' + str(i+1)] = self.kx_marginals[i]
        if (CW.rank == 0):
            f_wb = open(self.iteration_path + '/convergence_data_Iteration'+str(self.iteration) + '.pick', 'wb')
            pickle.dump(conv_data, f_wb)
        CW.Barrier()  
        return evs, b0z_e, amp_vec1

    def solve_EVs_local_max(self, amp_vec, kx_lm, find_maxima=False, show_spectrum=False):
        if (self.kx_regime != len(amp_vec) or self.kx_regime != len(self.wb_ar) or self.kx_regime != len(kx_lm)):
            raise Exception('kx_regime does not agree with amps, wb, or kx argument. This is very bad.')
        self.wb.set_scales(1)
        self.wb['g'] = np.array([amp_vec[i, 0]*self.wb_ar[i] for i in range(self.kx_regime)]).sum(axis=0)
        self.wb_mem_avg()
        b0z_e = self.evolution_IVP(self.wb)
        ev_lm = []
        if (find_maxima):
            evs = self.solve_EVP_spectrum(b0z_e, show_spectrum=show_spectrum)
            if (CW.rank == 0):
                try:
                    kx_ar = self.get_kx_local_max(evs)
                except:
                    logger.info('wrong local max count.')
                    self.plot_EVs(evs, debug=True)
                    sys.exit()
            else:
                kx_ar = None
            kx_ar = CW.bcast(kx_ar, root=0)
            ev_dict = dict(evs)
            for kx in kx_ar:
                ev_lm.append(ev_dict[kx])
            ev_lm = CW.bcast(ev_lm, root=0)
            return ev_lm, b0z_e, kx_ar, evs
        else:
            for kx in kx_lm:
                ev_lm.append(self.solve_EVP_kx(b0z_e, kx))
            return ev_lm, b0z_e

    def get_avg_wb(self, avgd_iter_count, show_flux=False):
        wb_sum = np.zeros_like(self.wb['g'])
        delta_T = 0
        for i in range(self.iteration - avgd_iter_count - 1, self.iteration):
            data = pickle.load(open(self.path + '/Iteration'+str(i) + '/convergence_data_Iteration' + str(i) + '.pick', 'rb'))
            delta_T = delta_T + data['del_t']
            if ('amp3' in data):
                wb_sum = wb_sum + data['del_t'] * (data['amp1']*np.array(data['wb1']) + data['amp2']*np.array(data['wb2']) + data['amp3']*np.array(data['wb3']))
            elif ('amp2' in data):
                wb_sum = wb_sum + data['del_t'] * (data['amp1']*np.array(data['wb1']) + data['amp2']*np.array(data['wb2']))
            else:
                wb_sum = wb_sum + data['del_t'] * (data['amp']*np.array(data['wb']))
        wb_avg_ar = wb_sum / delta_T
        P = (self.Rayleigh / self.Prandtl)**(-1/2)
        dif = -P*self.b0z['g']
        plt.plot(self.z, dif, label='diffusion')
        plt.plot(self.z, wb_avg_ar, label='advection')
        plt.plot(self.z, dif+wb_avg_ar, label='total')
        plt.legend()
        plt.title('Heat Flux')
        plt.savefig(self.iteration_path + '/Iteration' + str(self.iteration) + '_HeatFlux')
        if (show_flux):
            plt.show()
        return wb_avg_ar

    def store_avg_eigenfunctions(self, avgd_iter_count):
        def empty_field_dict():
            f_empty = np.zeros_like(self.b0z['g'])
            return {
                # 'amp' : 0.0,
                'w' : f_empty.copy(),
                'b' : f_empty.copy(),
                'p' : f_empty.copy(),
                'u' : f_empty.copy(),
                # 'bz' : f_empty.copy(),
                # 'uz' : f_empty.copy(),
                # 'wz' : f_empty.copy(),
            }
        fields_dict = dict()
        delta_T = 0
        for i in range(self.iteration - avgd_iter_count - 1, self.iteration):
            iteration_i_path = self.path + '/Iteration'+ str(i) + '/'
            conv_data = pickle.load(open(iteration_i_path + '/convergence_data_Iteration' + str(i) + '.pick', 'rb'))
            del_t = conv_data['del_t']
            kx1 = conv_data['kx1']
            kx2 = conv_data['kx2']
            kx3 = conv_data['kx3']
            amp1 = conv_data['amp1']            
            amp2 = conv_data['amp2']
            amp3 = conv_data['amp3']
            delta_T += del_t
            for file in os.listdir(iteration_i_path + 'data'):
                if (not file.endswith('.pick')):
                    continue
                f = open(iteration_i_path + 'data/' + file, 'rb')
                data = pickle.load(f)
                kx = data['kx']
                if (abs(kx - kx1) < 1e-8):
                    amp = np.sqrt(amp1)
                elif (abs(kx - kx2) < 1e-8):
                    amp = np.sqrt(amp2)
                elif (abs(kx - kx3) < 1e-8):
                    amp = np.sqrt(amp3)
                else:
                    raise ValueError('Wavenumber in data file matches none of the converged wavenumbers!')
                if (not kx in fields_dict.keys()):
                    fields_dict[kx] = empty_field_dict()
                    fields_dict[kx]['kx'] = kx
                # Removes sign ambiguity to avoid cancellation
                # Each iteration follows the same sign convection across all eigenfunctions
                # Testing 'w' @ 100 is just an arbitrary choice (cannot test on boundary)
                # if (fields_dict[kx]['w'].real[100] < 0.0):
                #     amp *= -1
                phi = np.angle(data['w'][10])
                fields_dict[kx]['p'] += data['p'] * np.exp(-1j*phi) * amp * del_t
                fields_dict[kx]['b'] += data['b'] * np.exp(-1j*phi) * amp * del_t
                fields_dict[kx]['u'] += data['u'] * np.exp(-1j*phi) * amp * del_t
                fields_dict[kx]['w'] += data['w'] * np.exp(-1j*phi) * amp * del_t
                # fields_dict[kx]['bz'] += data['bz'] * amp * del_t
                # fields_dict[kx]['uz'] += data['uz'] * amp * del_t
                # fields_dict[kx]['wz'] += data['wz'] * amp * del_t
                f.close()
        for kx in fields_dict.keys():                
            fields_dict[kx]['p'] /= delta_T
            fields_dict[kx]['b'] /= delta_T
            fields_dict[kx]['u'] /= delta_T
            fields_dict[kx]['w'] /= delta_T
            fields_dict[kx]['b0z'] = self.b0z['g']
            # fields_dict[kx]['bz'] /= delta_T
            # fields_dict[kx]['uz'] /= delta_T
            # fields_dict[kx]['wz'] /= delta_T
        pickle.dump(fields_dict, open(self.iteration_path + '/fields_dict.pick', 'wb'))

    def wb_mem_avg(self):
        # return
        # if (not self.suppress_memory):
        if (False):
            self.wb.set_scales(1)
            wb_n = self.wb['g']
            self.wb['g'] = self.wb_mem * (1 - self.del_t / self.T_mem) + wb_n * self.del_t / self.T_mem
            CW.barrier()
        # self.wb['c'][200:] = 0

    def next_iteration(self, evs, b0z_eg, amps):
        if (CW.rank == 0 and not os.path.exists(self.iteration_path + '/figures')):
            os.mkdir(self.iteration_path + '/figures')
        CW.Barrier()
        for i in range(CW.rank, 3, CW.size):
            if (i == 0):
                # logger.warning('plotting grid')
                self.plot_vars_grid()
            if (i == 1):
                # logger.warning('plotting coeff')
                self.plot_vars_coeff()
            if (i == 2):
                # logger.warning('plotting flux')
                self.plot_flux()

        # if (CW.rank == 0):
        #     logger.info('plotting eigenfunctions grid')
        #     logger.info('plotting eigenfunctions coeff')
        #     self.plot_vars_coeff()
        #     logger.info('plotting flux')
        #     flux_const_dev = self.plot_flux()
        # else:
        #     flux_const_dev = None
        # flux_const_dev = CW.bcast(flux_const_dev, root=0)
        logger.info('Iteration ' + str(self.iteration) + ' completed successfully. Saving results...')
        b0z_g = 0.5 * (b0z_eg[:] + b0z_eg[::-1])
        self.iteration += 1
        self.iteration_path = self.path + '/Iteration'+str(self.iteration)
        if (CW.rank == 0 and not os.path.isdir(self.iteration_path)):
            os.mkdir(self.iteration_path)
        self.Rayleigh *= self.ra_growth_coeff
        self.evs = evs
        self.ev_dict = dict(evs)
        self.amplitudes.append(amps)
        self.wb_mem = np.array([amps[i]*self.wb_ar[i] for i in range(self.kx_regime)]).sum(axis=0)
        self.b0z['g'] = b0z_g
        # self.b0z['c'][250:] = 0
        self.profiles.append(b0z_g)
        self.sim_time = round(self.sim_time + self.del_t, 8)
        logger.info('Current simulation time: ' + str(self.sim_time))
        self.sim_times.append(self.sim_time)
        if (CW.rank == 0):
            self.plot_EVs(evs)
            self.kx_marginals = self.get_kx_local_max(evs, new_iteration=True)
        self.kx_marginals = CW.bcast(self.kx_marginals, root=0)
        self.kx_regime = CW.bcast(self.kx_regime, root=0)
        self.kx_marginals_ar.append(self.kx_marginals)
        # self.update_dt_params()
        self.update_pi_range()
        self.rbc_data['profiles'] = self.profiles
        self.rbc_data['sim_times'] = self.sim_times
        self.rbc_data['kx_marginals_ar'] = self.kx_marginals_ar
        self.rbc_data['amplitudes'] = self.amplitudes
        self.rbc_data['iteration'] = self.iteration
        self.rbc_data['pi_range'] = self.pi_range
        self.rbc_data['del_t_newton'] = self.del_t_newton
        self.rbc_data['del_t_broyden'] = self.del_t_broyden
        self.rbc_data['iters_new_ts'] = self.iters_new_ts
        # self.rbc_data['flux_const_dev'] = flux_const_dev
        try:
            self.rbc_data['Ra'].append(self.Rayleigh)
        except:
            pass
        if (CW.rank == 0):
            pickle.dump(self.evs, open(self.iteration_path + '/evs.pick', 'wb'))
            pickle.dump(self.profiles, open(self.path + '/rbc_profiles_grid.pick', 'wb'))
            pickle.dump(self.sim_times, open(self.path + '/sim_times.pick', 'wb'))
            pickle.dump(self.kx_marginals_ar, open(self.path + '/kx_marginals.pick', 'wb'))
            pickle.dump(self.amplitudes, open(self.path + '/amplitudes.pick', 'wb'))
            pickle.dump(self.rbc_data, open(self.path + '/rbc_data.pick', 'wb'))
        # return 
        return 0
        # if (self.kx_regime > 1):
        #     self.pi_range = 15

    def update_dt_params(self):
        if (self.kx_regime == 2):
            self.del_t_broyden = self.del_t_broyden
            # if (self.ev_maxima_count <= 2):
            #     self.T_mem = self.del_t_broyden
            # else:
            #     self.T_mem = 5 * self.del_t_broyden
        elif (self.ev_maxima_count == 3 and self.kx_regime == 3):
            self.del_t_broyden = self.del_t_broyden
            # self.T_mem = self.del_t_broyden

    def update_pi_range(self, pi_range = None):
        if (self.pi_range <= 5):
            self.pi_range = 10
        if (pi_range == None and self.pi_range * np.pi - self.kx_marginals[-1] < 2*np.pi):
            self.pi_range = self.pi_range + 5
            self.kx_domain = list(map(lambda n: n*np.pi/2, range(1, self.pi_range*2)))
        if (pi_range != None):
            self.pi_range = pi_range
            self.kx_domain = list(map(lambda n: n*np.pi/2, range(1, pi_range*2)))

    def plot_EVs(self, EV_spectrum, show_spectrum=False, save_spectrum=True, debug=False):
        [kx_ar, EV_ar] = list(zip(*EV_spectrum))
        f, ax = plt.subplots(figsize=(5, 5))
        ax.plot(kx_ar, EV_ar, '.')
        ax.set_xlim(0, self.pi_range*np.pi)
        ax.xaxis.set_major_locator(tck.MultipleLocator(5*np.pi))
        ax.xaxis.set_minor_locator(tck.MultipleLocator(np.pi))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.grid(True)
        plt.xlabel(r'$k_x$')
        plt.ylabel(r'$\mathrm{Im}(\omega)$')
        plt.title(r'RBC growth rates, Iteration ' + str(self.iteration) + ', Time ' + str(self.sim_time))
        if (save_spectrum):
            if (not debug):
                plt.savefig(self.iteration_path + '/EV_spectrum_iteration' + str(self.iteration))
                plt.ylim(-0.001, 0.0001)
                plt.xlim(0, self.pi_range*np.pi)
                plt.savefig(self.iteration_path + '/EV_spectrum_zoomed_iteration' + str(self.iteration))
            else:
                plt.savefig(self.iteration_path + '/EV_spectrum_debug_iteration' + str(self.iteration))
                plt.ylim(-0.01, 0.0005)
                plt.xlim(0, self.pi_range*np.pi)
                plt.savefig(self.iteration_path + '/EV_spectrum_debug_zoomed_iteration' + str(self.iteration))
        if (show_spectrum):
            plt.show()
        plt.close()

    def plot_vars_grid(self):
        P = (self.Rayleigh * self.Prandtl)**(-1/2)

        b0z = self.domain.new_field()
        p = self.domain.new_field()
        b = self.domain.new_field()
        u = self.domain.new_field()
        w = self.domain.new_field()
        bz = self.domain.new_field()
        uz = self.domain.new_field()
        wz = self.domain.new_field()
        wb = self.domain.new_field()
        wb_mem = self.domain.new_field()
        Pbzz = self.domain.new_field()
        bt = self.domain.new_field()

        # if (not os.path.exists(self.iteration_path + '/figures')):
        #     os.mkdir(self.iteration_path + '/figures')
        for file in os.listdir(self.iteration_path + '/data'):
            if (not file.endswith('.pick')):
                continue
            f = open(self.iteration_path + '/data/' + file, 'rb')
            data = pickle.load(f)
            # b0z['c'] = data['b0z']
            kx = data['kx']
            p['g'] = data['p']
            b['g'] = data['b']
            u['g'] = data['u']
            w['g'] = data['w']
            bz['g'] = data['bz']
            uz['g'] = data['uz']
            wz['g'] = data['wz']
            wb['g'] = data['wb']
            # wb_mem['g'] = data['wb_mem']
            wbz = wb.differentiate(0)
            Pbzz.set_scales(self.domain.dealias)
            Pbzz['g'] = P * bz.differentiate(0)['g'] 
            bt.set_scales(self.domain.dealias)
            bt['g'] = Pbzz['g'] - wbz['g']
            bt.set_scales(1)

            p_vars = {
                # b : 'b',
                # w : 'w',
                u : 'u',
                # wb : '<w\'b\'>',
                # Pbzz : '$\kappa\Delta b$',
                # b0z : '$b0_z$'
                # # bt : '$\partial_t b$'
                w : '$w\'$',
                b : '$b\'$',
                wb : '$<w\'b\'>$',
                uz : '$u\'_z$',
                wz : '$w\'_z$',
                bz : '$b\'_z$',
                # wb_mem : '$<w\'b\'>_{avg}$'
                wbz : '$<w\'b\'>_z$'
            }

            # p_vars = [b, w, u, wb, Pbzz, bt]
            # var_names = ['b', 'w', 'u', '<w\'b\'>', '$\kappa\Delta b$', '$\partial_t b$']

            fig_height = 8
            nrow = 2
            ncol = 4

            plt.figure(figsize=[fig_height*2.0,fig_height])
            for i, v in enumerate(p_vars):
                plt.subplot(nrow,ncol,i+1)
                v.set_scales(1)
                plt.plot(self.z, v['g'].real)
                plt.xlim(-0.5, 0.5)
                plt.xlabel('z', fontsize=14)
                plt.ylabel(p_vars[v], fontsize=14, rotation=90)
                # if i == 0:
                #     plt.legend()

            pi_mult = round(kx/np.pi, 1)
            pi_mult_str = str(pi_mult).replace('.', 'p')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle('Iteration ' + str(self.iteration) + ', $k_x = $'  + str(pi_mult) + '$\pi$ eigenfunctions')
            plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_kx' + pi_mult_str + 'pi_eigenfunctions_grid.png')
            plt.close()

    def plot_vars_coeff(self):
        P = (self.Rayleigh * self.Prandtl)**(-1/2)

        b0z = self.domain.new_field()
        p = self.domain.new_field()
        b = self.domain.new_field()
        u = self.domain.new_field()
        w = self.domain.new_field()
        bz = self.domain.new_field()
        uz = self.domain.new_field()
        wz = self.domain.new_field()
        wb = self.domain.new_field()
        wb_mem = self.domain.new_field()
        Pbzz = self.domain.new_field()
        bt = self.domain.new_field()

        # if (not os.path.exists(self.iteration_path + '/figures')):
        #     os.mkdir(self.iteration_path + '/figures')
        for file in os.listdir(self.iteration_path + '/data'):
            if (not file.endswith('.pick')):
                continue
            f = open(self.iteration_path + '/data/' + file, 'rb')
            data = pickle.load(f)
            kx = data['kx']
            p['g'] = data['p']
            b['g'] = data['b']
            # b0z['c'] = data['b0z']
            u['g'] = data['u']
            w['g'] = data['w']
            bz['g'] = data['bz']
            uz['g'] = data['uz']
            wz['g'] = data['wz']
            wb['g'] = data['wb']
            # wb_mem['g'] = data['wb_mem']
            wbz = wb.differentiate(0)
            wb.set_scales(1)
            Nz = len(wz['c'])

            p_vars = {
                # b0z : '$\partial_z b_0$',
                u : '$u\'$',
                w : '$w\'$',
                b : '$b\'$',
                wb : '$<w\'b\'>$',
                uz : '$u\'_z$',
                wz : '$w\'_z$',
                bz : '$b\'_z$',
                wbz : '$<w\'b\'>_z$'
                # wb_mem : '$<w\'b\'>_{avg}$'
            }
            fig_height = 8
            nrow = 2
            ncol = 4

            plt.figure(figsize=[fig_height*2.0,fig_height])
            for i, v in enumerate(p_vars):
                v.set_scales(1)
                ax = plt.subplot(nrow,ncol,i+1)
                if (0 < i < 4):
                    plt.plot(list(range(1, int(Nz + 1)))[::2], [abs(c) for c in v['c'].real[::2]], linestyle='None', marker='.')
                else:
                    plt.plot(list(range(1, int(Nz + 1)))[::2], [abs(c) for c in v['c'].real[1::2]], linestyle='None', marker='.')
                # plt.plot(z, v['g'].imag, label='imag')
                # plt.xlim(0, 50)
                # plt.xlabel('z', fontsize=14)
                plt.ylabel(p_vars[v], fontsize=14, rotation=90)
                plt.yscale('log')
                # ax.set_yscale('log')

            pi_mult = round(kx/np.pi, 1)
            pi_mult_str = str(pi_mult).replace('.', 'p')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle('Iteration ' + str(self.iteration) + ', $k_x = $'  + str(pi_mult) + '$\pi$ eigenfunctions (coefficient magnitudes)')
            plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_kx' + pi_mult_str + 'pi_eigenfunctions_coeff.png')
            # plt.show()
            plt.close()

    def plot_flux(self):
        # if (not os.path.exists(self.iteration_path + '/figures')):
        #     os.mkdir(self.iteration_path + '/figures')
        try:
            f = open(self.iteration_path + '/convergence_data_Iteration' + str(self.iteration) + '.pick', 'rb')

        except:
            return
        data = pickle.load(f)
        Prandtl = self.Prandtl
        Rayleigh = self.Rayleigh
        P = (Rayleigh * Prandtl)**(-1/2)
        if ('kx4' in data.keys()):
            b0z = self.domain.new_field()
            wb1 = self.domain.new_field()
            wb2 = self.domain.new_field()
            b0z = data['b0z']
            wb1 = data['wb1']
            wb2 = data['wb2']
            wb3 = data['wb3']
            wb4 = data['wb4']
            kx1 = data['kx1']
            kx2 = data['kx2']
            kx3 = data['kx3']
            kx4 = data['kx4']
            amp1 = data['amp1']
            amp2 = data['amp2']
            amp3 = data['amp3']
            amp4 = data['amp4']
            diffusion = - P*b0z
            advection1 = wb1*amp1
            advection2 = wb2*amp2
            advection3 = wb3*amp3
            advection4 = wb4*amp4
            flux = diffusion + advection1 + advection2 + advection3 + advection4
            plt.plot(self.z, diffusion, label = 'Diffusion')
            pi_mult1 = str(round(kx1/np.pi, 1))
            pi_mult2 = str(round(kx2/np.pi, 1))
            pi_mult3 = str(round(kx3/np.pi, 1))
            pi_mult4 = str(round(kx4/np.pi, 1))
            plt.plot(self.z, advection1, label = 'Advection ($k_x  = $' + pi_mult1 + '$\pi$)')
            plt.plot(self.z, advection2, label = 'Advection ($k_x  = $' + pi_mult2 + '$\pi$)')
            plt.plot(self.z, advection3, label = 'Advection ($k_x  = $' + pi_mult3 + '$\pi$)')
            plt.plot(self.z, advection4, label = 'Advection ($k_x  = $' + pi_mult4 + '$\pi$)')
            plt.plot(self.z, flux, label = 'Total')
            plt.xlim(-0.5, 0.5)
            plt.legend()
            plt.title('Heat Flux (Iteration ' + str(self.iteration) + ')')
            plt.xlabel('z')
            plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_flux.png')
            plt.close()

        elif ('kx3' in data.keys()):
            b0z = self.domain.new_field()
            wb1 = self.domain.new_field()
            wb2 = self.domain.new_field()
            b0z = data['b0z']
            wb1 = data['wb1']
            wb2 = data['wb2']
            wb3 = data['wb3']
            kx1 = data['kx1']
            kx2 = data['kx2']
            kx3 = data['kx3']
            amp1 = data['amp1']
            amp2 = data['amp2']
            amp3 = data['amp3']
            diffusion = - P*b0z
            advection1 = wb1*amp1
            advection2 = wb2*amp2
            advection3 = wb3*amp3
            flux = diffusion + advection1 + advection2 + advection3
            plt.plot(self.z, diffusion, label = 'Diffusion')
            pi_mult1 = str(round(kx1/np.pi, 1))
            pi_mult2 = str(round(kx2/np.pi, 1))
            pi_mult3 = str(round(kx3/np.pi, 1))
            plt.plot(self.z, advection1, label = 'Advection ($k_x  = $' + pi_mult1 + '$\pi$)')
            plt.plot(self.z, advection2, label = 'Advection ($k_x  = $' + pi_mult2 + '$\pi$)')
            plt.plot(self.z, advection3, label = 'Advection ($k_x  = $' + pi_mult3 + '$\pi$)')
            plt.plot(self.z, flux, label = 'Total')
            plt.xlim(-0.5, 0.5)
            plt.legend()
            plt.title('Heat Flux (Iteration ' + str(self.iteration) + ')')
            plt.xlabel('z')
            plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_flux.png')
            plt.close()
        
        elif ('kx1' in data.keys()):
            b0z = self.domain.new_field()
            wb1 = self.domain.new_field()
            wb2 = self.domain.new_field()
            b0z = data['b0z']
            wb1 = data['wb1']
            wb2 = data['wb2']
            kx1 = data['kx1']
            kx2 = data['kx2']
            amp1 = data['amp1']
            amp2 = data['amp2']
            diffusion = - P*b0z
            advection1 = wb1*amp1
            advection2 = wb2*amp2
            flux = diffusion + advection1 + advection2
            plt.plot(self.z, diffusion, label = 'Diffusion')
            pi_mult1 = str(round(kx1/np.pi, 1))
            pi_mult2 = str(round(kx2/np.pi, 1))
            plt.plot(self.z, advection1, label = 'Advection ($k_x  = $' + pi_mult1 + '$\pi$)')
            plt.plot(self.z, advection2, label = 'Advection ($k_x  = $' + pi_mult2 + '$\pi$)')
            plt.plot(self.z, flux, label = 'Total')
            plt.xlim(-0.5, 0.5)
            plt.legend()
            plt.title('Heat Flux (Iteration ' + str(self.iteration) + ')')
            plt.xlabel('z')
            plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_flux.png')
            plt.close()
        
        elif ('kx' in data.keys()):
            b0z = self.domain.new_field()
            wb1 = self.domain.new_field()
            wb2 = self.domain.new_field()
            b0z = data['b0z']
            wb = data['wb']
            kx = data['kx']
            amp = data['amp']
            diffusion = - P*b0z
            advection = wb*amp
            flux = diffusion + advection
            plt.plot(self.z, diffusion, label = 'Diffusion')
            pi_mult = str(round(kx/np.pi, 1))
            plt.plot(self.z, advection, label = 'Advection ($k_x  = $' + pi_mult + '$\pi$)')
            plt.plot(self.z, flux, label = 'Total')
            plt.xlim(-0.5, 0.5)
            plt.legend()
            plt.title('Heat Flux (Iteration ' + str(self.iteration) + ')')
            plt.xlabel('z')
            plt.savefig(self.iteration_path + '/figures' + '/Iteration' + str(self.iteration) + '_flux.png')
            plt.close()
        try:
            flux_avg = np.mean(flux)
            flux_const_dev = np.mean(np.abs(flux - flux_avg))
            logger.info('Constant flux deviation: ' + str(flux_const_dev))
            logger.info('Flux at boundary: ' + str(diffusion[0]))
            return flux_const_dev
        except Exception as e:
            logger.warning('Failed to calculate flux deviation with exception: ' + str(e))
            return np.inf

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

class ExpectedException(Exception):
    pass