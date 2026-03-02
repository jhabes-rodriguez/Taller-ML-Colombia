import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

def perform_eda():
    path = 'DATA/earthquakes_colombia.csv'
    if not os.path.exists(path):
        print(f"Error: {path} no existe.")
        return

    df = pd.read_csv(path)
    
    # 1. Conteo de registros
    total_records = len(df)
    colombia_records = df[df['place'].str.contains('Colombia', na=False, case=False)].shape[0]
    
    print("--- 1. CONTEO DE REGISTROS ---")
    print(f"Total de registros: {total_records}")
    print(f"Registros en Colombia: {colombia_records}")
    print()

    # 2. Valores nulos
    print("--- 2. VALORES NULOS ---")
    null_counts = df.isnull().sum()
    null_percent = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame({'Nulos': null_counts, 'Porcentaje': null_percent})
    print(null_df[null_df['Nulos'] > 0])
    print()

    # 3. Distribuciones (Mag y Depth)
    print("--- 3. ESTADÍSTICAS DESCRIPTIVAS ---")
    print(df[['mag', 'depth']].describe())
    print()

    # 4. Correlaciones
    print("--- 4. CORRELACIÓN ---")
    corr = df[['mag', 'depth', 'latitude', 'longitude']].corr()
    print(corr)
    print()

    # Ensure IMAGENES directory exists
    os.makedirs("IMAGENES", exist_ok=True)

    # 5. Visualizaciones
    # Mapa scatter: Longitud vs Latitud coloreado por PROFUNDIDAD
    plt.figure(figsize=(10, 8))
    scatter_depth = plt.scatter(df['longitude'], df['latitude'], c=df['depth'], cmap='viridis_r', s=10, alpha=0.6)
    plt.colorbar(scatter_depth, label='Profundidad (km)')
    plt.title('Sismicidad en Colombia: Ubicación vs Profundidad')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.savefig('IMAGENES/scatter_depth.png', dpi=300)
    plt.close()
    print("Imagen guardada: IMAGENES/scatter_depth.png")

    # Mapa scatter: Longitud vs Latitud coloreado por MAGNITUD
    plt.figure(figsize=(10, 8))
    scatter_mag = plt.scatter(df['longitude'], df['latitude'], c=df['mag'], cmap='hot', s=10, alpha=0.6)
    plt.colorbar(scatter_mag, label='Magnitud')
    plt.title('Sismicidad en Colombia: Ubicación vs Magnitud')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.savefig('IMAGENES/scatter_magnitude.png', dpi=300)
    plt.close()
    print("Imagen guardada: IMAGENES/scatter_magnitude.png")

    # Histograma de Magnitudes
    plt.figure(figsize=(10, 5))
    sns.histplot(df['mag'], bins=30, kde=True, color='blue')
    plt.title('Distribución de Magnitudes')
    plt.savefig('IMAGENES/dist_magnitude.png')
    plt.close()

    # Histograma de Profundidades
    plt.figure(figsize=(10, 5))
    sns.histplot(df['depth'], bins=30, kde=True, color='green')
    plt.title('Distribución de Profundidades')
    plt.savefig('IMAGENES/dist_depth.png')
    plt.close()

if __name__ == "__main__":
    perform_eda()
