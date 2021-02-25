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
        ke_data = pickle.load(open(path + '/ke_data_nom.pick', 'rb'))
        ke_ar = ke_data['ke_ar']
        # ke_max_ar = ke_data['ke_max_ar']
        sim_times_ar = ke_data['sim_times_ar']
        for index in range(start, start+count):
            ke_avg = file['tasks']['ke'][index][0, 0]
            ke_ar.append(ke_avg)
            sim_times_ar.append(file['scales/sim_time'][index])
        ke_data['ke_ar'] = ke_ar
        ke_data['sim_times_ar'] = sim_times_ar
        pickle.dump(ke_data, open(path + '/ke_data_nom.pick', 'wb'))
        


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    global path 
    path = os.path.dirname(os.path.abspath(__file__))
    write_data = False
    if (write_data):
        ke_data = {'ke_ar' : [], 'sim_times_ar' : []}
        pickle.dump(ke_data, open(path + '/ke_data_nom.pick', 'wb'))

        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        post.visit_writes(args['<files>'], main, output=output_path)
    else:
        ke_data_eq = pickle.load(open(path + '/ke_data_eq.pick', 'rb'))
        ke_data = pickle.load(open(path + '/ke_data_nom.pick', 'rb'))

        ke_ar_eq = ke_data_eq['ke_ar']
        ke_ar = ke_data['ke_ar']
        sim_times_ar_eq = ke_data_eq['sim_times_ar']
        sim_times_ar = ke_data['sim_times_ar']
        plt.plot(sim_times_ar, ke_ar, color='darkred', label = 'Conductive I.C.')
        plt.plot(sim_times_ar_eq, ke_ar_eq, color='darkblue', label = 'Equilibrated I.C.')
        plt.xlim(0, 400)
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\overline{|\mathbf{u}|^2}$')
        plt.title(r'$Ra \, = \, 10^8$')
        plt.savefig(path + '/pubfigs/sim_eq_ke')
