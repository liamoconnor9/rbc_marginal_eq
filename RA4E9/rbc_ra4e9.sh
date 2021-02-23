#PBS -S /bin/bash
#PBS -N rbc_ra4e9
#PBS -l select=1:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=72:00:00
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
cd ~/scratch/dedalus/EVP_evolution/RA4E9/
mpiexec_mpt -np 28 python3 Main.py