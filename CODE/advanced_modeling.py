import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
import os

def run_advanced_modeling():
    input_path = 'DATA/earthquakes_advanced.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} no existe.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Selección de Features para el "Súper Modelo"
    # Usamos las coordenadas cartesianas 3D, la energía (en log para escalamiento) y la densidad temporal
    df['log_energy'] = np.log10(df['energy_joules'])
    features = ['pos_x', 'pos_y', 'pos_z', 'log_energy', 'rolling_30d_count']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Iniciando Modelamiento Avanzado...")

    # 2. DBSCAN para detección de outliers (Noise Reduction)
    # eps y min_samples ajustados para datos sísmicos
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_label'] = dbscan.fit_predict(X_scaled)
    
    n_outliers = len(df[df['dbscan_label'] == -1])
    print(f"- DBSCAN identificó {n_outliers} sismos como ruido (outliers).")

    # 3. PCA para visualización y simplificación
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    print(f"- PCA completado. Varianza explicada: {np.sum(pca.explained_variance_ratio_):.2f}")

    # 4. K-Means sobre datos limpios (sin outliers de DBSCAN)
    df_clean = df[df['dbscan_label'] != -1].copy()
    X_clean_scaled = scaler.transform(df_clean[features])
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_clean['advanced_cluster'] = kmeans.fit_predict(X_clean_scaled)
    
    print("- K-Means avanzado ejecutado sobre datos limpios.")

    # 5. Sub-Clustering Jerárquico del Nido de Bucaramanga
    # Buscamos el clúster que corresponde a Bucaramanga (mayor profundidad media)
    buc_cluster_id = df_clean.groupby('advanced_cluster')['depth'].mean().idxmax()
    df_buc = df_clean[df_clean['advanced_cluster'] == buc_cluster_id].copy()
    
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(scaler.transform(df_buc[features]), method='ward'))
    plt.title('Dendrograma: Sub-estructuras del Nido de Bucaramanga')
    plt.xlabel('Sismos')
    plt.ylabel('Distancia Euclidiana')
    os.makedirs("IMAGENES", exist_ok=True)
    plt.savefig('IMAGENES/bucaramanga_dendrogram.png')
    plt.close()
    print("- Sub-clustering jerárquico del Nido de Bucaramanga completado.")

    # 6. Guardar Resultados
    output_path = 'DATA/earthquakes_advanced_results.csv'
    df_clean.to_csv(output_path, index=False)
    
    # Visualización PCA Clusters
    plt.figure(figsize=(10, 7))
    plt.scatter(df_clean['pca_1'], df_clean['pca_2'], c=df_clean['advanced_cluster'], cmap='Set1', alpha=0.5)
    plt.title('Clusters en Espacio PCA (2 Componentes Principales)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('IMAGENES/pca_clusters.png')
    plt.close()

    print(f"\nModelamiento avanzado finalizado. Resultados en: {output_path}")

if __name__ == "__main__":
    run_advanced_modeling()
