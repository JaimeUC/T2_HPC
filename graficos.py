import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. DATOS

# Datos de Tiempos de Ejecución
# MPI: Cantidad de procesos
# Threads: Cantidad de hilos por proceso
# Time: Tiempo en segundos
data = [
    # Config 0: 1 MPI (Saturación de memoria visible)
    {"MPI": 1, "Threads": 1, "Time": 20.0373},
    {"MPI": 1, "Threads": 2, "Time": 18.4311},
    {"MPI": 1, "Threads": 4, "Time": 18.4561},
    {"MPI": 1, "Threads": 8, "Time": 17.0160},
    
    # Config 1: 2 MPI
    {"MPI": 2, "Threads": 1, "Time": 11.0506},
    {"MPI": 2, "Threads": 2, "Time": 9.7073},
    {"MPI": 2, "Threads": 4, "Time": 9.6228},
    {"MPI": 2, "Threads": 8, "Time": 6.0641},
    
    # Config 2: 4 MPI
    {"MPI": 4, "Threads": 1, "Time": 9.7521},
    {"MPI": 4, "Threads": 2, "Time": 6.4358},
    {"MPI": 4, "Threads": 4, "Time": 5.2305},
    
    # Config 3: 8 MPI (El ganador está aquí)
    {"MPI": 8, "Threads": 1, "Time": 30.8541},
    {"MPI": 8, "Threads": 2, "Time": 4.3006},
    
    # Config 4: 16 MPI (Sobrecarga de comunicación)
    {"MPI": 16, "Threads": 1, "Time": 9.3272}
]

# Datos de Convergencia (Iteración vs Movimiento)
convergence_values = [
    2.775545, 0.167440, 0.091300, 0.059599, 0.043317, 
    0.034831, 0.030537, 0.028728, 0.027918, 0.027764,
    0.028038, 0.028408, 0.028461, 0.028827, 0.029085,
    0.029447, 0.029867, 0.029995, 0.030271, 0.030295
]
iterations = list(range(1, len(convergence_values) + 1))

# 2. PROCESAMIENTO DE DATOS

df = pd.DataFrame(data)

# Calcular Total CPUs (Procesos x Threads)
df["Total_CPUs"] = df["MPI"] * df["Threads"]

# Crear etiquetas para los gráficos
df["Label"] = df["MPI"].astype(str) + " MPI x " + df["Threads"].astype(str) + " Thr"

# Calcular Speedup (Base: 1 MPI x 1 Thread)
t_base = df[(df["MPI"] == 1) & (df["Threads"] == 1)]["Time"].values[0]
df["Speedup"] = t_base / df["Time"]

# Calcular Eficiencia (Speedup / N_CPUs)
df["Efficiency"] = df["Speedup"] / df["Total_CPUs"]


# 3. GENERACIÓN DE GRÁFICOS


# Estilo general
plt.style.use('seaborn-v0_8-whitegrid')

# GRÁFICO 1: SPEEDUP
plt.figure(figsize=(10, 6))

# Línea ideal
plt.plot([1, 16], [1, 16], 'k--', label="Ideal (Lineal)", alpha=0.5)

# Series por Configuración MPI
colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red', 16: 'purple'}
for mpi_procs in sorted(df["MPI"].unique()):
    subset = df[df["MPI"] == mpi_procs].sort_values("Total_CPUs")
    plt.plot(subset["Total_CPUs"], subset["Speedup"], 
             marker='o', linestyle='-', linewidth=2, 
             label=f"{mpi_procs} Procesos MPI", color=colors.get(mpi_procs))

plt.title("Análisis de Speedup: Escalabilidad Híbrida", fontsize=14)
plt.xlabel("Total de CPUs (Procesos x Threads)", fontsize=12)
plt.ylabel("Speedup ($T_{serial} / T_{paralelo}$)", fontsize=12)
plt.xticks([1, 2, 4, 8, 16])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("grafico_speedup.png", dpi=300)
print("Generado: grafico_speedup.png")

# GRÁFICO 2: EFICIENCIA
plt.figure(figsize=(10, 6))

# Línea ideal (Eficiencia = 1.0)
plt.axhline(y=1.0, color='k', linestyle='--', label="Eficiencia Ideal", alpha=0.5)

for mpi_procs in sorted(df["MPI"].unique()):
    subset = df[df["MPI"] == mpi_procs].sort_values("Total_CPUs")
    plt.plot(subset["Total_CPUs"], subset["Efficiency"], 
             marker='s', linestyle='-', linewidth=2,
             label=f"{mpi_procs} Procesos MPI", color=colors.get(mpi_procs))

plt.title("Análisis de Eficiencia Paralela", fontsize=14)
plt.xlabel("Total de CPUs", fontsize=12)
plt.ylabel("Eficiencia", fontsize=12)
plt.xticks([1, 2, 4, 8, 16])
plt.ylim(0, 1.2)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("grafico_eficiencia.png", dpi=300)
print("Generado: grafico_eficiencia.png")

# GRÁFICO 3: COMPARATIVA 16 CPUs (EL "MAX RESOURCE")
# Filtramos solo los casos donde se usan 16 CPUs en total
df_16 = df[df["Total_CPUs"] == 16].sort_values("Time", ascending=True)

plt.figure(figsize=(8, 6))
bars = plt.bar(df_16["Label"], df_16["Time"], 
               color=['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728'], 
               alpha=0.8, edgecolor='black')

plt.title("Tiempo de Ejecución con 16 CPUs (Distintas Estrategias)", fontsize=14)
plt.ylabel("Tiempo (segundos) - Menor es mejor", fontsize=12)
plt.xlabel("Configuración (Procesos x Threads)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Añadir valor encima de las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{height:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.savefig("grafico_comparativa_16cpus.png", dpi=300)
print("Generado: grafico_comparativa_16cpus.png")

# GRÁFICO 4: CONVERGENCIA
plt.figure(figsize=(10, 5))
plt.plot(iterations, convergence_values, marker='o', color='purple', linestyle='-', linewidth=2, markersize=5)

plt.title("Convergencia del Algoritmo (Movimiento de Centroides)", fontsize=14)
plt.xlabel("Iteración", fontsize=12)
plt.ylabel("Desplazamiento (Norma L2)", fontsize=12)
plt.xticks(iterations)
plt.yscale('log') # Escala logarítmica para apreciar mejor el ajuste fino
plt.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("grafico_convergencia.png", dpi=300)
print("Generado: grafico_convergencia.png")