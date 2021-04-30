#PBS -S /bin/bash
#PBS -N sim_figs0
#PBS -l select=32:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=24:00:00
#PBS -j oe
module load mpi-sgi/mpt
module load comp-intel
export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true
cd ~/scratch/dedalus/EVP_evolution/
mpiexec_mpt -np 896 python3 rbc_sim_load_ic.py
mpiexec_mpt -np 896 python3 -m dedalus merge_procs checkpoints_nonoise
mpiexec_mpt -np 896 python3 plot_slices.py checkpoints_nonoise/*.h5 --output=frames_nonoise
mpiexec_mpt -np 896 python3 -m dedalus merge_procs profiles_nonoise