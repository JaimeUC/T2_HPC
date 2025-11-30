import numpy as np
from mpi4py import MPI
from numba import njit, prange
import time


# 1. Generación de Datos

def generate_distributed_data(n_total, d, k, seed, rank, n_procs):

    # Cada proceso genera su porción de datos independientemente.

    # Calcular cuántos puntos le tocan a este proceso
    n_local = n_total // n_procs
    
    # Ajustar para el último proceso si la división no es exacta
    if rank == n_procs - 1:
        n_local += n_total % n_procs
        
    # Asegurar reproducibilidad
    np.random.seed(seed + rank)
    
    # Generar datos aleatorios
    local_data = np.random.rand(n_local, d).astype(np.float64)
    
    return local_data

# 2. Funciones con Numba (Parallel)

@njit(parallel=True)
def compute_distances(data, centroids):

    # Calcula la distancia (cuadrada) euclidiana de cada punto a cada centroide.

    n_samples, n_features = data.shape
    n_centroids = centroids.shape[0]
    
    distances = np.zeros((n_samples, n_centroids), dtype=np.float64)
    
    for i in prange(n_samples):
        for c in range(n_centroids):
            dist = 0.0
            for f in range(n_features):
                diff = data[i, f] - centroids[c, f]
                dist += diff * diff
            distances[i, c] = dist
            
    return distances

@njit(parallel=True)
def assign_labels(distances):

    # Asigna cada punto al índice del cluster más cercano.

    n_samples = distances.shape[0]
    labels = np.zeros(n_samples, dtype=np.int32)
    
    for i in prange(n_samples):
        min_dist = distances[i, 0]
        min_idx = 0
        for c in range(1, distances.shape[1]):
            if distances[i, c] < min_dist:
                min_dist = distances[i, c]
                min_idx = c
        labels[i] = min_idx
        
    return labels

@njit(parallel=True)
def compute_local_sums(data, labels, k):
    
    # Acumula las sumas de coordenadas y conteos por cluster.

    n_samples, n_features = data.shape
    
    cluster_sums = np.zeros((k, n_features), dtype=np.float64)
    cluster_counts = np.zeros(k, dtype=np.int32)
    
    for i in range(n_samples):
        l = labels[i]
        cluster_counts[l] += 1
        for f in range(n_features):
            cluster_sums[l, f] += data[i, f]
            
    return cluster_sums, cluster_counts


# 3. Main y Lógica MPI

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N_TOTAL = 4_000_000  # 4 millones de muestras
    D = 20               # 20 features
    K = 10               # Clusters
    SEED = 42
    MAX_ITER = 20        # Máximo iteraciones
    TOL = 1e-4           # Tolerancia de convergencia
    
    # 1. Generar datos distribuidos
    local_data = generate_distributed_data(N_TOTAL, D, K, SEED, rank, size)
    
    # 2. Inicializar centroides (solo en root)
    centroids = np.zeros((K, D), dtype=np.float64)
    if rank == 0:
        np.random.seed(SEED)
        # Elegir K puntos aleatorios como centroides iniciales
        centroids = np.random.rand(K, D).astype(np.float64)
        
    # 3. Distribuir centroides iniciales a todos los procesos
    comm.Bcast(centroids, root=0)
    
    # Sincronización antes de medir tiempo
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Bucle Principal K-Means
    for iteration in range(MAX_ITER):
        
        # A. Calcular distancias y asignar etiquetas (Numba)
        distances = compute_distances(local_data, centroids)
        labels = assign_labels(distances)
        
        # B. Sumas locales parciales (Numba)
        local_sums, local_counts = compute_local_sums(local_data, labels, K)
        
        # C. Reducción Global (MPI)
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        
        # D. Actualizar Centroides
        new_centroids = np.zeros_like(centroids)
        
        # Evitar división por cero
        for k in range(K):
            if global_counts[k] > 0:
                new_centroids[k] = global_sums[k] / global_counts[k]
            else:
                # Si un cluster se queda vacío, mantener el viejo o re-inicializar
                new_centroids[k] = centroids[k] 
        
        # E. Chequear convergencia (Norma del desplazamiento de centroides)
        diff = np.linalg.norm(new_centroids - centroids)
        
        # Actualizar centroides para la siguiente iteración
        centroids = new_centroids
        
        if rank == 0:
            print(f"Iteración {iteration+1}: movimiento centroides = {diff:.6f}")
            
        if diff < TOL:
            if rank == 0:
                print("Converged!")
            break

    # Fin del cronómetro
    comm.Barrier()
    end_time = MPI.Wtime()
    
    if rank == 0:
        print("-" * 30)
        print(f"Resultados finales con {size} procesos MPI")
        print(f"Tiempo total de ejecución: {end_time - start_time:.4f} segundos")
        print("-" * 30)