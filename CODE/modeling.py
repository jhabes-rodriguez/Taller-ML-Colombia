import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def run_modeling():
    input_path = 'DATA/normalized_data.csv'
    original_path = 'DATA/earthquakes_filtered.csv'
    
    if not os.path.exists(input_path) or not os.path.exists(original_path):
        print("Error: Los archivos de datos no existen.")
        return

    X_scaled = pd.read_csv(input_path)
    df_original = pd.read_csv(original_path)

    # 1. Análisis de K (2 a 10)
    wcss = []
    silhouettes = []
    K_range = range(2, 11)
    
    print("Calculando métricas para K de 2 a 10...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        wcss.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, clusters))
        print(f"K={k} procesado.")

    os.makedirs("IMAGENES", exist_ok=True)

    # 2. Generar Gráfico del Método del Codo
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, wcss, 'bx-')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Método del Codo')
    plt.grid(True)
    plt.savefig('IMAGENES/elbow_method.png')
    plt.close()

    # 3. Generar Gráfico de Silhouette Score
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, silhouettes, 'ro-')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Análisis de Silueta por K')
    plt.grid(True)
    plt.savefig('IMAGENES/silhouette_score.png')
    plt.close()

    print("\nVisualizaciones guardadas en IMAGENES/")

    # 4. Entrenamiento Final con K seleccionado (K=5)
    k_final = 5
    print(f"\nEjecutando clustering final con K={k_final}...")
    kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    df_original['cluster'] = kmeans_final.fit_predict(X_scaled)

    # Guardar resultados
    output_path = 'DATA/clustered_data.csv'
    df_original.to_csv(output_path, index=False)
    
    # 5. Mapa de Clústeres Final
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(df_original['longitude'], df_original['latitude'], 
                        c=df_original['cluster'], cmap='Set1', s=25, alpha=0.6)
    plt.colorbar(scatter, label='Clúster')
    plt.title(f'Segmentación Geofísica de Colombia (K={k_final})')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.savefig('IMAGENES/clusters_map.png', dpi=300)
    plt.close()

    # Imprimir métricas para el reporte
    print("\n--- Resultados de Métricas ---")
    for k, w, s in zip(K_range, wcss, silhouettes):
        print(f"K={k} | WCSS: {w:.2f} | Silueta: {s:.4f}")

if __name__ == "__main__":
    run_modeling()
