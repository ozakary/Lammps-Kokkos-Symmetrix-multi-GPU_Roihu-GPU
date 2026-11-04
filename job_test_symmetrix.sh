#!/bin/bash
#SBATCH --job-name=lammps-symmetrix
#SBATCH --account=<project>
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72   # 1 MPI rank per GPU, 72 Grace cores per GPU
#SBATCH --gres=gpu:gh200:1                       # 1 GH200
#SBATCH --output=lammps_symmetrix_%j.out
#SBATCH --error=lammps_symmetrix_%j.err

module purge
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30

# Drop any build-time CUDA stub paths inherited from the submitting shell
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -e cuda-stub -e '/stubs' | paste -sd:)
unset LIBRARY_PATH

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

PROJAPPL=/projappl/<project>/<username>/packages/lammps-symmetrix

# Add LAMMPS and symmetrix shared libraries to the runtime library path
export LD_LIBRARY_PATH=$PROJAPPL/lammps-symmetrix-build/lib64:$PROJAPPL/lammps-symmetrix-build/lib:$LD_LIBRARY_PATH

srun $PROJAPPL/lammps-symmetrix-build/bin/lmp \
  -in your_input.in \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton on neigh half
