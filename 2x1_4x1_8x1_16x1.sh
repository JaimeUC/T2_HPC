#!/bin/bash
#SBATCH --job-name=fix_1t
#SBATCH --partition=hpc-iic3533
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=res_1thread_%j.out

source ~/miniconda3/bin/activate kmeans_env
export NUMBA_NUM_THREADS=1

# IMPORTANTE: --bind-to none arregla el error que te daba antes

echo "=== Corriendo 16 MPI x 1 Thread (Config 4) ==="
mpirun --bind-to none -n 16 python kmeans_distributed.py

echo "=== Corriendo 8 MPI x 1 Thread (Config 3) ==="
mpirun --bind-to none -n 8 python kmeans_distributed.py

echo "=== Corriendo 4 MPI x 1 Thread (Config 2) ==="
mpirun --bind-to none -n 4 python kmeans_distributed.py

echo "=== Corriendo 2 MPI x 1 Thread (Config 1) ==="
mpirun --bind-to none -n 2 python kmeans_distributed.py