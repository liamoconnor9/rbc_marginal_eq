#PBS -S /bin/bash
#PBS -N rbc_sim_eq
#PBS -l select=4:ncpus=28:mpiprocs=28:model=bro
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
mpiexec_mpt -np 112 python3 rbc_sim_load_ic.py