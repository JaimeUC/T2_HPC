#!/bin/bash
#SBATCH --job-name=fix_2x8
#SBATCH --partition=hpc-iic3533
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1    # 1 proceso por nodo (ocupa sus 8 cores con threads)
#SBATCH --cpus-per-task=8      # 8 threads por proceso
#SBATCH --time=00:15:00
#SBATCH --output=res_2x8_%j.out

source ~/miniconda3/bin/activate kmeans_env
export NUMBA_NUM_THREADS=8

echo "=== Corriendo 2 MPI x 8 Threads (Config 1 Faltante) ==="
mpirun --bind-to none -n 2 python kmeans_distributed.py