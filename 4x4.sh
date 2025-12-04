#!/bin/bash
#SBATCH --job-name=fix_4x4
#SBATCH --partition=hpc-iic3533
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2    # 2 procesos por nodo
#SBATCH --cpus-per-task=4      # 4 threads por proceso
#SBATCH --time=00:15:00
#SBATCH --output=res_4x4_%j.out

source ~/miniconda3/bin/activate kmeans_env
export NUMBA_NUM_THREADS=4

echo "=== Corriendo 4 MPI x 4 Threads (Config 2 Faltante) ==="
mpirun --bind-to none -n 4 python kmeans_distributed.py