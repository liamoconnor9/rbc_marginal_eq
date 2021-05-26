"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import pickle
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import dedalus.public as de
from dedalus.extras import plot_tools
import publication_settings

matplotlib.rcParams.update(publication_settings.params)
plt.rcParams.update({'figure.autolayout': True})
plt.gcf().subplots_adjust(left=0.15)

fig = plt.figure()

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        nu_data = pickle.load(open(path + '/nu_data_eq_noflow.pick', 'rb'))
        wb_ar = nu_data['wb_ar']
        diff_ar = nu_data['diff_ar']
        # nu_max_ar = nu_data['nu_max_ar']
        z = file['/scales/z/1.0'][:]
        sim_times_ar = nu_data['sim_times_ar']
        nu_data['z'] = z
        for index in range(start, start+count):
            wb = file['tasks']['adv_flux'][index][0]
            wb_ar.append(wb)
            diff = file['tasks']['diffusive_flux'][index][0]
            diff_ar.append(diff)
            sim_times_ar.append(file['scales/sim_time'][index])
        nu_data['wb_ar'] = wb_ar
        nu_data['diff_ar'] = diff_ar
        nu_data['sim_times_ar'] = sim_times_ar
        pickle.dump(nu_data, open(path + '/nu_data_eq_noflow.pick', 'wb'))
        


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    global path 
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = False
    if (write_data):
        args = docopt(__doc__)

        output_path = pathlib.Path(args['--output']).absolute()
        nu_data = {'nu_ar' : [], 'sim_times_ar' : [], 'diff_ar' : [], 'wb_ar' : []}
        pickle.dump(nu_data, open(path + '/nu_data_eq_noflow.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)
        nu_data = pickle.load(open(path + '/nu_data_eq_noflow.pick', 'rb'))
        z = nu_data['z']
        wb_ar = nu_data['wb_ar']
        diff_ar = nu_data['diff_ar']
        sim_times_ar = nu_data['sim_times_ar']
        
        z_basis = de.Chebyshev('z', len(z), interval=(-1/2, 1/2))
        domain = de.Domain([z_basis], grid_dtype=np.float64)
        nu_ar = []
        diff = domain.new_field()
        wb = domain.new_field()
        # print(diff_ar[0].shape)
        for i in range(len(sim_times_ar)):
            print(i)
            diff['g'] = -diff_ar[i].squeeze()[:].real
            wb['g'] = wb_ar[i].squeeze()[:].real
            diff_int = de.operators.integrate(diff, 'z')
            adv_int = de.operators.integrate(wb, 'z')
            diff_scal = diff_int.evaluate()['g'][0]
            adv_scal = adv_int.evaluate()['g'][0]
            print('diff_scal: ' + str(diff_scal))
            nu_ar.append((adv_scal + diff_scal) / diff_scal)
        nu_data['nu_ar'] = nu_ar
        pickle.dump(nu_data, open(path + '/nu_data_eq_noflow.pick', 'wb'))
    else:
        nu_data_eq_noflow = pickle.load(open(path + '/nu_data_eq_noflow.pick', 'rb'))
        nu_data_eq = pickle.load(open(path + '/nu_data_eq.pick', 'rb'))
        nu_data = pickle.load(open(path + '/nu_data_nom.pick', 'rb'))

        nu_ar_eq_noflow = nu_data_eq_noflow['nu_ar']
        nu_ar_eq = nu_data_eq['nu_ar']
        nu_ar = nu_data['nu_ar']
        sim_times_ar_eq_noflow = nu_data_eq_noflow['sim_times_ar']
        sim_times_ar_eq = nu_data_eq['sim_times_ar']
        sim_times_ar = nu_data['sim_times_ar']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.plot(sim_times_ar, nu_ar, color='black', label = 'Linear IC')
        plt.plot(sim_times_ar_eq, nu_ar_eq, color=colors[-1], label = 'MSTE IC')
        plt.plot(sim_times_ar_eq_noflow, nu_ar_eq_noflow, color=colors[0], label = 'MSTE No Flow IC')
        plt.xlim(0, 100)
        plt.legend(frameon=False)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathrm{Nu}$')
        plt.title(r'$\rm{Ra} \, = \, 10^8$')
        # plt.savefig(path + '/pubfigs/sim_eq_nu')
        plt.savefig(path + '/publication_materials/sim_eq_nu_noflow.pdf')
