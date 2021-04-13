conda activate dedalus

mpiexec_mpt -np 1 python3 plot_flux.py

python3 plot_spectra_2ra.py

mpiexec_mpt -np 1 python3 plot_ke.py

mpiexec_mpt -np 1 python3 plot_nu.py

mpiexec_mpt -np 1 python3 plot_profiles.py
