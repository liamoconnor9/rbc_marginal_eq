import os
import sys
import pickle

rewind_iterations = 50
path = os.path.dirname(os.path.abspath(__file__)) + '/RA1E9/results'

profiles = pickle.load(open(path + '/rbc_profiles_grid.pick', "rb"))[:-rewind_iterations]
sim_times = pickle.load(open(path + '/sim_times.pick', 'rb'))[:-rewind_iterations]
amplitudes = pickle.load(open(path + '/amplitudes.pick', 'rb'))[:-rewind_iterations]
kx_marginals = pickle.load(open(path + '/kx_marginals.pick', 'rb'))[:-rewind_iterations]
rbc_data = pickle.load(open(path + '/rbc_data.pick', 'rb'))
try:
    iteration = max(rbc_data['iteration'] - rewind_iterations, 0)
    rbc_data['iteration'] = iteration
except Exception as e:
    print('Failed to rewind iteration count')
    print(e)
try:
    ra = rbc_data['Ra'][:-rewind_iterations]
    rbc_data['Ra'] = ra
except Exception as e:
    print('Failed to rewind rayleigh numbers')
    print(e)
rbc_data['profiles'] = profiles
rbc_data['sim_times'] = sim_times
rbc_data['kx_marginals'] = kx_marginals
rbc_data['amplitudes'] = amplitudes

pickle.dump(profiles, open(path + '/rbc_profiles_grid.pick', 'wb'))
pickle.dump(sim_times, open(path + '/sim_times.pick', 'wb'))        
pickle.dump(kx_marginals, open(path + '/kx_marginals.pick', 'wb'))
pickle.dump(amplitudes, open(path + '/amplitudes.pick', 'wb'))
pickle.dump(rbc_data, open(path + '/rbc_data.pick', 'wb'))