import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def run_experiment():
    input_path = 'DATA/earthquakes_filtered.csv'
    if not os.path.exists(input_path):
        print("Error: DATA/earthquakes_filtered.csv no existe.")
        return

    df = pd.read_csv(input_path)
    features = ['latitude', 'longitude', 'depth', 'mag']
    X = df[features]

    # --- 1. K-Means SIN ESCALAR ---
    print("Corriendo K-Means sin escalar...")
    kmeans_raw = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_raw'] = kmeans_raw.fit_predict(X)

    # --- 2. K-Means CON ESCALAR ---
    print("Corriendo K-Means con StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster_scaled'] = kmeans_scaled.fit_predict(X_scaled)

    # --- 3. Visualización y Comparación ---
    os.makedirs("IMAGENES", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Sin Escalar
    scatter1 = ax1.scatter(df['longitude'], df['latitude'], c=df['cluster_raw'], cmap='viridis', s=20, alpha=0.6)
    ax1.set_title('K-Means SIN Escalar (Dominancia de Profundidad)')
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    plt.colorbar(scatter1, ax=ax1)

    # Plot Con Escalar
    scatter2 = ax2.scatter(df['longitude'], df['latitude'], c=df['cluster_scaled'], cmap='viridis', s=20, alpha=0.6)
    ax2.set_title('K-Means CON StandardScaler (Variables Equilibradas)')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    plt.colorbar(scatter2, ax=ax2)

    plt.tight_layout()
    plt.savefig('IMAGENES/scaling_comparison.png')
    plt.close()

    # Análisis de escalas
    print("\n--- Análisis de Escalas (Rangos) ---")
    print(X.describe().loc[['min', 'max']])
    
    # Guardar resultados para análisis
    df.to_csv('DATA/scaling_experiment_results.csv', index=False)
    print("\nResultados guardados en DATA/scaling_experiment_results.csv e IMAGENES/scaling_comparison.png")

if __name__ == "__main__":
    run_experiment()
